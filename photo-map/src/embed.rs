//! Gemma embeddings (words and photos) through ONNX Runtime, in this process (no Python).
//!
//! Everything it needs lives in the project folder: ONNX Runtime and the CUDA libraries in runtime/, the model and tokenizer in
//! .cache/onnx/embeddinggemma-2/ (the ONNX export of Gemma; vision_encoder_fp16_fused.onnx is for photos). The CUDA libraries are loaded by
//! full path before the first session is made, so nothing has to be installed or put on a search path.

use ort::ep;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use image::RgbaImage;
use tokenizers::Tokenizer;

/// What the model sees in front of a search query (the model's own prompt for queries).
pub const QUERY_PROMPT: &str = "task: search result | query: ";
/// The model's default for a text that is searched (a caption), when it has no title.
pub const DOCUMENT_PROMPT: &str = "title: none | text: ";
/// CUDA libraries to load, dependencies first (ONNX Runtime's CUDA provider needs these four).
const CUDA_LIBS: [&str; 4] = ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10"];
/// cuDNN (its softmax and attention helpers), also dependencies first.
const CUDNN_LIBS: [&str; 8] = [
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
];

pub struct Gemma {
    // the environment must outlive the sessions
    env: Environment,
    session: Session,
    /// The photo encoder, loaded when the first photo is searched (it costs video memory).
    vision: Option<Session>,
    tokenizer: Tokenizer,
    model_dir: PathBuf,
    pub on_gpu: bool,
}

// Gemma's special tokens around a photo's soft tokens, and how a photo is cut up (its image processor's settings).
const BOS: i64 = 2;
const BOI: i64 = 255999;
const SOFT: i64 = 258880;
const EOI: i64 = 258882;
const EOS: i64 = 1;
const PATCH: u32 = 16;
const POOL: u32 = 3;
/// A photo makes at most this many soft tokens: the same as when the maps' photos were embedded.
const MAX_SOFT: u32 = 140;
const MAX_PATCHES: u32 = MAX_SOFT * POOL * POOL;
/// The fused attention of the photo encoder needs a patch count that is a multiple of 8: masked patches fill up to it.
const PADDED: usize = (MAX_PATCHES as usize).div_ceil(8) * 8;
const PATCH_VALUES: usize = (PATCH * PATCH * 3) as usize;

/// A photo as the encoder takes it.
pub struct Patches {
    /// PADDED x 768: each patch's pixels 0..1 (rows, columns, channels), zeros after the real ones.
    pub pixels: Vec<f32>,
    /// PADDED x 2: each patch's column and row; -1 for the filler.
    pub positions: Vec<i64>,
    pub soft_tokens: usize,
}

/// The size a photo is scaled to: its shape kept, the most patches that fit the budget, both sides a multiple of 48.
fn scaled_size(height: u32, width: u32) -> Result<(u32, u32)> {
    let side = POOL * PATCH;
    let factor = ((MAX_PATCHES * PATCH * PATCH) as f64 / (height as f64 * width as f64)).sqrt();
    let (mut h, mut w) = (((factor * height as f64 / side as f64).floor() as u32) * side, ((factor * width as f64 / side as f64).floor() as u32) * side);
    let longest = (MAX_PATCHES / (POOL * POOL)) * side;
    match (h, w) {
        (0, 0) => return Err("photo too small".into()),
        (0, _) => {
            h = side;
            w = ((width / height) * side).min(longest);
        }
        (_, 0) => {
            w = side;
            h = ((height / width) * side).min(longest);
        }
        _ => {}
    }
    Ok((h, w))
}

/// Cut a photo into the encoder's patches (what Gemma's image processor does: scale with bicubic, 16 x 16 patches in rows).
pub fn patches(img: &RgbaImage) -> Result<Patches> {
    let rgb = image::DynamicImage::ImageRgba8(img.clone()).to_rgb8();
    let (h, w) = scaled_size(rgb.height(), rgb.width())?;
    let rgb = if (h, w) == (rgb.height(), rgb.width()) { rgb } else { image::imageops::resize(&rgb, w, h, image::imageops::FilterType::CatmullRom) };
    let (rows, cols) = ((h / PATCH) as usize, (w / PATCH) as usize);
    let mut pixels = vec![0f32; PADDED * PATCH_VALUES];
    let mut positions = vec![-1i64; PADDED * 2];
    for py in 0..rows {
        for px in 0..cols {
            let at = py * cols + px;
            positions[at * 2] = px as i64;
            positions[at * 2 + 1] = py as i64;
            let patch = &mut pixels[at * PATCH_VALUES..(at + 1) * PATCH_VALUES];
            for r in 0..PATCH as usize {
                for c in 0..PATCH as usize {
                    let p = rgb.get_pixel((px * PATCH as usize + c) as u32, (py * PATCH as usize + r) as u32);
                    for ch in 0..3 {
                        patch[(r * PATCH as usize + c) * 3 + ch] = p[ch] as f32 / 255.0;
                    }
                }
            }
        }
    }
    Ok(Patches { pixels, positions, soft_tokens: rows * cols / (POOL * POOL) as usize })
}

pub type Result<T> = std::result::Result<T, String>;

fn err(what: &str) -> impl Fn(ort::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

impl Gemma {
    /// Load the model. `root`: the project folder (with runtime/ and .cache/).
    pub fn load(root: &Path) -> Result<Self> {
        let onnx = root.join("runtime/ort/libonnxruntime.so");
        let model_dir = root.join(".cache/onnx/embeddinggemma-2");
        for need in [&onnx, &model_dir.join("onnx/model_fp16.onnx"), &model_dir.join("tokenizer.json")] {
            if !need.exists() {
                return Err(format!("missing {}", need.display()));
            }
        }
        // the CUDA libraries, by full path (a missing one just means no GPU)
        let cuda_dir: PathBuf = root.join("runtime/cuda");
        let on_gpu = CUDA_LIBS.iter().all(|l| ort::util::preload_dylib(cuda_dir.join(l)).is_ok())
            && CUDNN_LIBS.iter().all(|l| ort::util::preload_dylib(root.join("runtime/cudnn").join(l)).is_ok());
        let mut env = ort::init_from(&onnx).map_err(|e| format!("loading ONNX Runtime: {e}"))?.with_name("photo-map");
        if on_gpu {
            env = env.with_execution_providers([ep::CUDA::default().with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested).build()]); // allocate what is needed, not the next power of two (the card is shared with the map)
        }
        let env = env.build().map_err(err("starting ONNX Runtime"))?;
        let session = Session::builder(&env)
            .and_then(|mut b| b.commit_from_file(model_dir.join("onnx/model_fp16.onnx")))
            .map_err(err("loading the model"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| format!("loading the tokenizer: {e}"))?;
        Ok(Self { env, session, vision: None, tokenizer, model_dir, on_gpu })
    }

    /// The token ids of a text, as the model gets them (its special tokens included).
    pub fn tokens(&self, text: &str) -> Result<Vec<i64>> {
        let enc = self.tokenizer.encode(text, true).map_err(|e| format!("tokenizing: {e}"))?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    /// A search query's vector (768 numbers, length 1).
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokens(&format!("{QUERY_PROMPT}{text}"))?;
        self.run(ids, Vec::new())
    }

    /// A caption's vector (768 numbers, length 1), as the maps' captions were embedded.
    pub fn embed_document(&mut self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokens(&format!("{DOCUMENT_PROMPT}{text}"))?;
        self.run(ids, Vec::new())
    }

    /// A photo's vector (768 numbers, length 1), as the maps' photos were embedded: no prompt, at most 140 soft tokens.
    pub fn embed_image(&mut self, img: &RgbaImage) -> Result<Vec<f32>> {
        let cut = patches(img)?;
        self.embed_patches(vec![cut]).pop().unwrap()
    }

    /// Several photos at once (already cut up, see `patches`): the encoder runs on them together, which is faster.
    pub fn embed_patches(&mut self, cuts: Vec<Patches>) -> Vec<Result<Vec<f32>>> {
        let n = cuts.len();
        let soft: Vec<usize> = cuts.iter().map(|c| c.soft_tokens).collect();
        let features = match self.encode_patches(cuts) {
            Ok(f) => f,
            Err(e) => return (0..n).map(|_| Err(e.clone())).collect(),
        };
        let mut start = 0;
        soft.into_iter()
            .map(|tokens| {
                let part = features[start * 512..(start + tokens) * 512].to_vec();
                start += tokens;
                let mut ids = vec![BOS, BOI];
                ids.extend(std::iter::repeat_n(SOFT, tokens));
                ids.extend([EOI, EOS]);
                self.run(ids, part)
            })
            .collect()
    }

    /// The encoder's soft tokens for the photos together, one after the other (512 numbers each).
    fn encode_patches(&mut self, cuts: Vec<Patches>) -> Result<Vec<f32>> {
        let batch = cuts.len();
        if self.vision.is_none() {
            let file = self.model_dir.join("onnx/vision_encoder_fp16_fused.onnx");
            let session = Session::builder(&self.env).and_then(|mut b| b.commit_from_file(&file)).map_err(err("loading the photo encoder"))?;
            self.vision = Some(session);
        }
        let tokens: usize = cuts.iter().map(|c| c.soft_tokens).sum();
        let mut pixels = Vec::with_capacity(batch * PADDED * PATCH_VALUES);
        let mut positions = Vec::with_capacity(batch * PADDED * 2);
        for c in cuts {
            pixels.extend(c.pixels);
            positions.extend(c.positions);
        }
        let vision = self.vision.as_mut().unwrap();
        let inputs = ort::inputs![
            "pixel_values" => Tensor::from_array(([batch, PADDED, PATCH_VALUES], pixels)).map_err(err("tensor"))?,
            "pixel_position_ids" => Tensor::from_array(([batch, PADDED, 2], positions)).map_err(err("tensor"))?,
        ];
        let out = vision.run(inputs).map_err(err("running the photo encoder"))?;
        let (shape, data) = out["image_features"].try_extract_tensor::<f32>().map_err(err("reading the photo features"))?;
        if shape[0] as usize != tokens {
            return Err(format!("the photo encoder made {} tokens, expected {tokens}", shape[0]));
        }
        Ok(data.to_vec())
    }

    /// Run the text model on token ids, with the photo's features if there are any: the sentence embedding, unit length.
    fn run(&mut self, ids: Vec<i64>, image_features: Vec<f32>) -> Result<Vec<f32>> {
        let n = ids.len();
        let soft = image_features.len() / 512;
        let features = |rows: usize, data: Vec<f32>| Tensor::from_array(([rows, 512usize], data)).map_err(err("tensor"));
        let inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([1usize, n], ids)).map_err(err("tensor"))?,
            "attention_mask" => Tensor::from_array(([1usize, n], vec![1i64; n])).map_err(err("tensor"))?,
            "image_features" => features(soft, image_features)?,
            "video_features" => features(0, Vec::new())?,
            "audio_features" => features(0, Vec::new())?,
        ];
        let outputs = self.session.run(inputs).map_err(err("running the model"))?;
        let (_, data) = outputs["sentence_embedding"].try_extract_tensor::<f32>().map_err(err("reading the result"))?;
        let norm = data.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        Ok(data.iter().map(|x| x / norm).collect())
    }
}

/// The one model all the tests share: ONNX Runtime allows one environment per process, and a copy takes 1.7 GB of the card.
#[cfg(test)]
pub fn shared_model() -> std::sync::Arc<std::sync::Mutex<Option<Gemma>>> {
    static MODEL: std::sync::OnceLock<std::sync::Arc<std::sync::Mutex<Option<Gemma>>>> = std::sync::OnceLock::new();
    MODEL
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
            std::sync::Arc::new(std::sync::Mutex::new(Some(Gemma::load(root).unwrap())))
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same token ids and (nearly) the same vectors as the Python / PyTorch side (tests/query_vectors.json).
    #[test]
    fn queries_match_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("photo-map/tests/query_vectors.json")).unwrap()).unwrap();
        assert_eq!(fixture["prompt"].as_str().unwrap(), QUERY_PROMPT);
        let shared = shared_model();
        let mut guard = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap();
        eprintln!("on the GPU: {}", model.on_gpu);
        for (k, text) in fixture["texts"].as_array().unwrap().iter().enumerate() {
            let text = text.as_str().unwrap();
            let ids: Vec<i64> = fixture["ids"][k].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            assert_eq!(model.tokens(&format!("{QUERY_PROMPT}{text}")).unwrap(), ids, "token ids of {text:?}");
            let want: Vec<f32> = fixture["vectors"][k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let got = model.embed_query(text).unwrap();
            let cos: f32 = want.iter().zip(&got).map(|(a, b)| a * b).sum();
            eprintln!("{text:?}: cosine with Python {cos:.5}");
            assert!(cos > 0.998, "{text:?}: cosine {cos}");
        }
    }

    /// A photo through the Rust side gives (nearly) the vector Python made of it, and its patches are the processor's.
    #[test]
    fn photos_match_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("photo-map/tests/image_vectors.json")).unwrap()).unwrap();
        let shared = shared_model();
        let mut guard = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap();
        for (k, file) in fixture["files"].as_array().unwrap().iter().enumerate() {
            let path = root.join("maps/wikivoyage.photomap/originals").join(file.as_str().unwrap());
            let Some(img) = crate::photo::load(&path) else { return };
            if k == 0 {
                // the patches themselves, against the processor's (saved by hand, in /tmp)
                if let (Ok(px), Ok(pos)) = (std::fs::read("/tmp/img0_pixels.bin"), std::fs::read("/tmp/img0_pos.bin")) {
                    let theirs: Vec<f32> = px.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                    let their_pos: Vec<i64> = pos.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect();
                    let cut = patches(&img).unwrap();
                    assert_eq!(&cut.positions[..their_pos.len()], &their_pos[..]);
                    let diff: f32 = cut.pixels[..theirs.len()].iter().zip(&theirs).map(|(a, b)| (a - b).abs()).sum::<f32>() / theirs.len() as f32;
                    let worst = cut.pixels[..theirs.len()].iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                    eprintln!("patches: mean difference {diff:.5}, largest {worst:.3} (pixel values are 0..1)");
                    assert!(diff < 0.01);
                }
            }
            let want: Vec<f32> = fixture["vectors"][k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let started = std::time::Instant::now();
            let got = model.embed_image(&img).unwrap();
            let cos: f32 = want.iter().zip(&got).map(|(a, b)| a * b).sum();
            eprintln!("{file}: cosine with Python {cos:.5} ({:?})", started.elapsed());
            assert!(cos > 0.99, "{file}: cosine {cos}");
        }
    }

    /// Speed of embedding real photos at different batch sizes (run by hand: cargo test --release batch_sizes -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn batch_sizes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let dir = root.join("maps/wikivoyage.photomap/originals");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let cuts: Vec<Patches> = files.iter().take(96).filter_map(|f| crate::photo::load(f)).map(|i| patches(&i).unwrap()).collect();
        let shared = shared_model();
        let mut guard = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap();
        model.embed_patches(cuts[..4].iter().map(|c| Patches { pixels: c.pixels.clone(), positions: c.positions.clone(), soft_tokens: c.soft_tokens }).collect()); // warm up
        for batch in [1usize, 2, 4, 8, 12, 16] {
            let started = std::time::Instant::now();
            let mut done = 0;
            for chunk in cuts.chunks(batch) {
                let group: Vec<Patches> = chunk.iter().map(|c| Patches { pixels: c.pixels.clone(), positions: c.positions.clone(), soft_tokens: c.soft_tokens }).collect();
                for r in model.embed_patches(group) {
                    r.unwrap();
                    done += 1;
                }
            }
            eprintln!("batch {batch:2}: {:.1} photos/s", done as f32 / started.elapsed().as_secs_f32());
        }
    }
}
