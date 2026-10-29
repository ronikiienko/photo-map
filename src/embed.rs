//! Gemma embeddings (words and photos) through ONNX Runtime, in this process (no Python).
//!
//! Everything it needs lives in the project folder: ONNX Runtime and the CUDA libraries in runtime/, the model and tokenizer in
//! .cache/onnx/embeddinggemma-2/ (the publisher's ONNX files, and the patched copies made from them for CUDA and TensorRT: patch.rs). The CUDA libraries are loaded by
//! full path before the first session is made, so nothing has to be installed or put on a search path.

use crate::runtime::{self, Setup};
use ort::ep;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use image::RgbaImage;
use tokenizers::Tokenizer;

/// What the model sees in front of a search query (the model's own prompt for queries).
pub const QUERY_PROMPT: &str = "task: search result | query: ";
pub struct Gemma {
    // the environment must outlive the sessions
    env: Environment,
    session: Session,
    /// The photo encoder, loaded when the first photo is searched (it costs video memory). With TensorRT it is the transformer
    /// layers only, and `pool` is what comes after them.
    vision: Option<Session>,
    pool: Option<Session>,
    tokenizer: Tokenizer,
    root: PathBuf,
    model_dir: PathBuf,
    pub on_gpu: bool,
    /// Run the photo encoder's layers on TensorRT (about 1.5 times faster): its libraries and the two converted models are there.
    /// Switched off if TensorRT fails to start, and the encoder runs on the CUDA provider as before.
    tensorrt: bool,
    /// See `Runtime::apple`.
    apple: Option<ep::coreml::ComputeUnits>,
}

/// The most photos TensorRT's engine takes at once (it is built for 1 to this many).
const TENSORRT_BATCH: usize = 4;

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
/// How long the longer side of a photo of this size must at least be for `patches` (what it is scaled to), 0 if it can't be cut.
pub fn longest_side_needed(width: u32, height: u32) -> u32 {
    scaled_size(height, width).map_or(0, |(h, w)| h.max(w))
}

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

/// What every model shares: the ONNX Runtime environment (one per process), and whether the graphics card and TensorRT can be used.
pub struct Runtime {
    pub env: Environment,
    pub on_gpu: bool,
    pub tensorrt: bool,
    /// On a Mac: what the photo encoders run on besides the processor (CoreML); None: the processor alone.
    pub apple: Option<ep::coreml::ComputeUnits>,
}

/// The encoder on one group of at most `TENSORRT_BATCH` photos: their soft tokens one after the other (512 numbers each).
fn encode_group(vision: &mut Session, pool: &mut Option<Session>, apple: bool, mut cuts: Vec<Patches>) -> Result<Vec<f32>> {
    let tokens: usize = cuts.iter().map(|c| c.soft_tokens).sum();
    if apple {
        // CoreML's network is made for exactly this many photos: a short group is filled up with blank ones (all filler, no tokens)
        while cuts.len() < TENSORRT_BATCH {
            cuts.push(Patches { pixels: vec![0f32; PADDED * PATCH_VALUES], positions: vec![-1i64; PADDED * 2], soft_tokens: 0 });
        }
    }
    let batch = cuts.len();
    let mut pixels = Vec::with_capacity(batch * PADDED * PATCH_VALUES);
    let mut positions = Vec::with_capacity(batch * PADDED * 2);
    for c in cuts {
        pixels.extend(c.pixels);
        positions.extend(c.positions);
    }
    let pool_positions = pool.is_some().then(|| positions.clone());
    let inputs = ort::inputs![
        "pixel_values" => Tensor::from_array(([batch, PADDED, PATCH_VALUES], pixels)).map_err(err("tensor"))?,
        "pixel_position_ids" => Tensor::from_array(([batch, PADDED, 2], positions)).map_err(err("tensor"))?,
    ];
    let out = vision.run(inputs).map_err(err("running the photo encoder"))?;
    let features: Vec<f32> = if let (Some(pool), Some(positions)) = (pool.as_mut(), pool_positions) {
        // TensorRT ran the layers; the pooling after them runs on CUDA
        let (_, hidden) = out["hidden"].try_extract_tensor::<f32>().map_err(err("reading the hidden states"))?;
        let inputs = ort::inputs![
            "hidden" => Tensor::from_array(([batch, PADDED, 768usize], hidden.to_vec())).map_err(err("tensor"))?,
            "pixel_position_ids" => Tensor::from_array(([batch, PADDED, 2], positions)).map_err(err("tensor"))?,
        ];
        let pooled = pool.run(inputs).map_err(err("running the photo pooling"))?;
        let (shape, data) = pooled["image_features"].try_extract_tensor::<f32>().map_err(err("reading the photo features"))?;
        // CoreML's filler photos come last and may leave a token of their own: it is cut off
        if shape[0] as usize != tokens && !(apple && shape[0] as usize > tokens) {
            return Err(format!("the photo encoder made {} tokens, expected {tokens}", shape[0]));
        }
        data[..tokens * (data.len() / shape[0] as usize)].to_vec()
    } else {
        let (shape, data) = out["image_features"].try_extract_tensor::<f32>().map_err(err("reading the photo features"))?;
        if shape[0] as usize != tokens {
            return Err(format!("the photo encoder made {} tokens, expected {tokens}", shape[0]));
        }
        data.to_vec()
    };
    Ok(features)
}

/// The text model's vectors for a group of photos, from their soft tokens one after the other.
fn read_photos(session: &mut Session, soft: Vec<usize>, features: Vec<f32>) -> Vec<Result<Vec<f32>>> {
    let n = soft.len();
    // the text model takes the photos together: their token lists padded to the longest, the padding masked out
    let mut start = 0;
    let photos: Vec<(Vec<i64>, Vec<f32>)> = soft
        .into_iter()
        .map(|tokens| {
            let part = features[start * 512..(start + tokens) * 512].to_vec();
            start += tokens;
            let mut ids = vec![BOS, BOI];
            ids.extend(std::iter::repeat_n(SOFT, tokens));
            ids.extend([EOI, EOS]);
            (ids, part)
        })
        .collect();
    match Gemma::run_together(session, photos) {
        Ok(vectors) => vectors.into_iter().map(Ok).collect(),
        Err(e) => (0..n).map(|_| Err(e.clone())).collect(),
    }
}

/// A session builder for a photo encoder: with CoreML when the machine has it and `apple` says so.
/// `sizes`: the dimensions the model leaves open, fixed to these sizes (CoreML needs shapes that do not change).
pub fn vision_builder(env: &Environment, apple: Option<ep::coreml::ComputeUnits>, cache: &Path, sizes: &[(&str, i64)]) -> ort::Result<ort::session::builder::SessionBuilder> {
    let mut builder = Session::builder(env)?;
    let Some(units) = apple else { return Ok(builder) };
    for (name, size) in sizes {
        builder = builder.with_dimension_override(name, *size)?;
    }
    let _ = std::fs::create_dir_all(cache);
    let coreml = ep::CoreML::default().with_compute_units(units).with_model_format(if std::env::var("PHOTO_MAP_COREML_FORMAT").as_deref() == Ok("nn") { ep::coreml::ModelFormat::NeuralNetwork } else { ep::coreml::ModelFormat::MLProgram }).with_model_cache_dir(cache.display()).build();
    builder.with_execution_providers([coreml]).map_err(|e| e.into())
}

impl Runtime {
    /// Start ONNX Runtime from the libraries `runtime::Setup` finds, with the CUDA libraries loaded by full path (a missing one
    /// means no GPU).
    pub fn start(root: &Path) -> Result<Self> {
        let setup = Setup::detect(root);
        let Some(ort_place) = &setup.ort else {
            return Err(format!("ONNX Runtime has not been downloaded (looked in {})", root.join("runtime").display()));
        };
        let onnx = ort_place.dir.join(runtime::ORT_LIB);
        let chosen = crate::backends::Choice::load(root);
        let mut on_gpu = false;
        if let (Some(gpu), true, true) = (&setup.gpu, setup.ort_cuda, chosen.cuda(&setup)) {
            on_gpu = runtime::CUDA_LIBS.iter().all(|l| ort::util::preload_dylib(gpu.cuda.dir.join(l)).is_ok())
                && runtime::CUDNN_LIBS.iter().all(|l| ort::util::preload_dylib(gpu.cudnn.dir.join(l)).is_ok());
        }
        let mut env = ort::init_from(&onnx).map_err(|e| format!("loading ONNX Runtime: {e}"))?.with_name("photo-map");
        if on_gpu {
            env = env.with_execution_providers([ep::CUDA::default().with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested).build()]); // allocate what is needed, not the next power of two (the card is shared with the map)
        }
        let env = env.build().map_err(err("starting ONNX Runtime"))?;
        let tensorrt = on_gpu
            && chosen.tensorrt(&setup)
            && setup.ort_tensorrt
            && setup.tensorrt.as_ref().is_some_and(|t| runtime::TENSORRT_LIBS.iter().all(|l| ort::util::preload_dylib(t.dir.join(l)).is_ok()));
        let apple = chosen.apple(&setup);
        Ok(Self { env, on_gpu, tensorrt, apple })
    }
}

impl Gemma {
    /// Load the model into a running ONNX Runtime. `root`: the project folder (with runtime/ and .cache/).
    pub fn load_with(rt: &Runtime, root: &Path) -> Result<Self> {
        let model_dir = root.join(".cache/onnx/embeddinggemma-2");
        for need in [&model_dir.join("onnx/model_fp16.onnx"), &model_dir.join("tokenizer.json")] {
            if !need.exists() {
                return Err(format!("missing {}", need.display()));
            }
        }
        let env = rt.env.clone();
        // the text model reads every photo too: on the processor it runs in float32 (patch.rs), on a card as published
        let text = if rt.on_gpu { model_dir.join("onnx/model_fp16.onnx") } else { crate::patch::ensure_or(&model_dir, crate::patch::GEMMA_TEXT_PROCESSOR, "onnx/model_fp16.onnx") };
        let session = Session::builder(&env).and_then(|mut b| b.commit_from_file(text)).map_err(err("loading the model"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| format!("loading the tokenizer: {e}"))?;
        let on_gpu = rt.on_gpu || rt.apple.is_some();
        let tensorrt = rt.tensorrt;
        let mut model = Self { env, session, vision: None, pool: None, tokenizer, root: root.to_path_buf(), model_dir, on_gpu, tensorrt, apple: rt.apple };
        // CoreML and TensorRT prepare the network for the chip or card when it is opened (minutes the first time): here, where a map being made can still be paused
        if model.apple.is_some() || model.tensorrt {
            model.open_vision()?;
        }
        Ok(model)
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

    /// A photo's vector (768 numbers, length 1), as the maps' photos were embedded: no prompt, at most 140 soft tokens.
    pub fn embed_image(&mut self, img: &RgbaImage) -> Result<Vec<f32>> {
        let cut = patches(img)?;
        self.embed_patches(vec![cut]).pop().unwrap()
    }

    /// Several photos at once (already cut up, see `patches`): the encoder runs on them together, which is faster. The encoder works
    /// in groups; while the text model reads one group's tokens (on the processor on a Mac) the encoder is already on the next.
    pub fn embed_patches(&mut self, mut cuts: Vec<Patches>) -> Vec<Result<Vec<f32>>> {
        let n = cuts.len();
        if self.vision.is_none() {
            if let Err(e) = self.open_vision() {
                return (0..n).map(|_| Err(e.clone())).collect();
            }
        }
        let mut groups = Vec::new();
        while !cuts.is_empty() {
            let rest = cuts.split_off(TENSORRT_BATCH.min(cuts.len()));
            groups.push(std::mem::replace(&mut cuts, rest));
        }
        let Self { session, vision, pool, apple, tokenizer: _, .. } = self;
        let (vision, apple) = (vision.as_mut().unwrap(), apple.is_some());
        let (send, receive) = std::sync::mpsc::sync_channel::<(Vec<usize>, Result<Vec<f32>>)>(1);
        std::thread::scope(|scope| {
            let reader = scope.spawn(move || {
                let mut out = Vec::with_capacity(n);
                for (soft, features) in receive {
                    match features {
                        Ok(features) => out.extend(read_photos(session, soft, features)),
                        Err(e) => out.extend(soft.iter().map(|_| Err(e.clone()))),
                    }
                }
                out
            });
            for group in groups {
                let soft: Vec<usize> = group.iter().map(|c| c.soft_tokens).collect();
                let features = encode_group(vision, pool, apple, group);
                if send.send((soft, features)).is_err() {
                    break;
                }
            }
            drop(send);
            reader.join().unwrap_or_else(|_| (0..n).map(|_| Err("the text model stopped".to_string())).collect())
        })
    }

    /// Open the photo encoder: on TensorRT if it can, else on the CUDA provider. TensorRT builds its engine the first time (about a
    /// minute) and keeps it in .cache/trt/.
    fn open_vision(&mut self) -> Result<()> {
        if self.tensorrt {
            match self.open_tensorrt() {
                Ok(()) => return Ok(()),
                Err(e) => {
                    eprintln!("TensorRT is off, the photo encoder runs on CUDA: {e}");
                    self.tensorrt = false;
                }
            }
        }
        if self.apple.is_some() {
            return self.open_coreml();
        }
        // CUDA: the fused attention; the processor: float32 (patch.rs)
        let file = if self.on_gpu {
            crate::patch::ensure_or(&self.model_dir, crate::patch::GEMMA_CUDA, "onnx/vision_encoder_fp16.onnx")
        } else {
            crate::patch::ensure_or(&self.model_dir, crate::patch::GEMMA_PROCESSOR, "onnx/vision_encoder_fp16.onnx")
        };
        let session = Session::builder(&self.env).and_then(|mut b| b.commit_from_file(&file)).map_err(err("loading the photo encoder"))?;
        self.vision = Some(session);
        Ok(())
    }

    /// CoreML takes the layers (standard operators only, `GEMMA_COREML`), on the graphics chip whatever Apple option is chosen: the
    /// neural engine's compiler gives up on them (after 15 minutes). The pooling after them runs on the processor, as it runs on
    /// CUDA beside TensorRT. As published, ONNX Runtime broke the network into about 100 pieces (its own operators are not
    /// CoreML's), which took 0.1 photos/s and as many compilations.
    fn open_coreml(&mut self) -> Result<()> {
        let layers = crate::patch::ensure(&self.model_dir, crate::patch::GEMMA_COREML)?;
        let pool = crate::patch::ensure(&self.model_dir, crate::patch::GEMMA_POOL)?;
        let cache = self.root.join(".cache/coreml/embeddinggemma-2");
        let layers = vision_builder(&self.env, Some(ep::coreml::ComputeUnits::CPUAndGPU), &cache, &[("s11", TENSORRT_BATCH as i64), ("s35", PADDED as i64)])
            .and_then(|mut b| b.commit_from_file(&layers))
            .map_err(err("loading the photo encoder on CoreML"))?;
        let pool = Session::builder(&self.env).and_then(|mut b| b.commit_from_file(&pool)).map_err(err("loading the photo pooling"))?;
        self.vision = Some(layers);
        self.pool = Some(pool);
        Ok(())
    }

    fn open_tensorrt(&mut self) -> Result<()> {
        let layers = crate::patch::ensure(&self.model_dir, crate::patch::GEMMA_TENSORRT)?;
        let pool = crate::patch::ensure(&self.model_dir, crate::patch::GEMMA_POOL)?;
        let cache = self.root.join(".cache/trt");
        std::fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
        let shapes = |n: usize| format!("pixel_values:{n}x{PADDED}x{PATCH_VALUES},pixel_position_ids:{n}x{PADDED}x2");
        let trt = ep::TensorRT::default()
            .with_fp16(true)
            .with_max_workspace_size(1 << 30) // the card is shared with the map: keep the engine's scratch space small
            .with_engine_cache(true)
            .with_engine_cache_path(cache.display())
            .with_timing_cache(true)
            .with_timing_cache_path(cache.display())
            .with_profile_min_shapes(shapes(1))
            .with_profile_opt_shapes(shapes(TENSORRT_BATCH))
            .with_profile_max_shapes(shapes(TENSORRT_BATCH))
            .build();
        // (the CUDA provider is already registered for every session by the environment, and takes what TensorRT leaves)
        let layers = Session::builder(&self.env)
            .and_then(|b| b.with_execution_providers([trt]).map_err(|e| e.into()))
            .and_then(|mut b| b.commit_from_file(&layers))
            .map_err(err("loading the photo encoder on TensorRT"))?;
        let pool = Session::builder(&self.env).and_then(|mut b| b.commit_from_file(&pool)).map_err(err("loading the photo pooling"))?;
        self.vision = Some(layers);
        self.pool = Some(pool);
        Ok(())
    }

    /// The encoder's soft tokens for the photos together, one after the other (512 numbers each).
    fn encode_patches(&mut self, mut cuts: Vec<Patches>) -> Result<Vec<f32>> {
        if self.vision.is_none() {
            self.open_vision()?;
        }
        if (self.pool.is_some() || self.apple.is_some()) && cuts.len() > TENSORRT_BATCH {
            // more photos than the TensorRT engine takes at once: in turns
            let mut out = Vec::new();
            while !cuts.is_empty() {
                let rest = cuts.split_off(TENSORRT_BATCH.min(cuts.len()));
                out.extend(self.encode_patches(std::mem::replace(&mut cuts, rest))?);
            }
            return Ok(out);
        }
        let apple = self.apple.is_some();
        encode_group(self.vision.as_mut().unwrap(), &mut self.pool, apple, cuts)
    }

    /// Run the text model on several photos at once: (token ids, soft token features) each. The ids are padded on the right to the
    /// longest and the padding is masked, so each result is what `run` gives for that photo alone.
    fn run_together(session: &mut Session, photos: Vec<(Vec<i64>, Vec<f32>)>) -> Result<Vec<Vec<f32>>> {
        let batch = photos.len();
        let longest = photos.iter().map(|(ids, _)| ids.len()).max().unwrap_or(0);
        let mut ids = vec![0i64; batch * longest]; // 0 is the pad token
        let mut mask = vec![0i64; batch * longest];
        let mut features = Vec::new();
        for (i, (photo_ids, photo_features)) in photos.into_iter().enumerate() {
            ids[i * longest..i * longest + photo_ids.len()].copy_from_slice(&photo_ids);
            mask[i * longest..i * longest + photo_ids.len()].fill(1);
            features.extend(photo_features);
        }
        let soft = features.len() / 512;
        let inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([batch, longest], ids)).map_err(err("tensor"))?,
            "attention_mask" => Tensor::from_array(([batch, longest], mask)).map_err(err("tensor"))?,
            "image_features" => Tensor::from_array(([soft, 512usize], features)).map_err(err("tensor"))?,
            "video_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new())).map_err(err("tensor"))?,
            "audio_features" => Tensor::from_array(([0usize, 512], Vec::<f32>::new())).map_err(err("tensor"))?,
        ];
        let outputs = session.run(inputs).map_err(err("running the model"))?;
        let (_, data) = outputs["sentence_embedding"].try_extract_tensor::<f32>().map_err(err("reading the result"))?;
        Ok(data
            .chunks_exact(768)
            .map(|row| {
                let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                row.iter().map(|x| x / norm).collect()
            })
            .collect())
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
pub fn shared_model() -> std::sync::Arc<crate::models::Hub> {
    crate::models::test_hub()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Where the time goes per photo once everything is warm: the layers, the pooling, the text model. PHOTO_MAP_PHOTOS=<folder>.
    #[test]
    #[ignore]
    fn gemma_where_time_goes() {
        let folder = std::env::var("PHOTO_MAP_PHOTOS").expect("PHOTO_MAP_PHOTOS");
        let mut files: Vec<_> = std::fs::read_dir(folder).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "jpg")).collect();
        files.sort();
        let cuts = || -> Vec<Patches> { files.iter().take(8).map(|f| patches(&image::open(f).unwrap().into_rgba8()).unwrap()).collect() };
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
        model.embed_patches(cuts().into_iter().take(4).collect());
        for _ in 0..2 {
            let t = std::time::Instant::now();
            let features = model.encode_patches(cuts()).unwrap();
            let encode = t.elapsed().as_secs_f32();
            let t = std::time::Instant::now();
            // a group as the map builder sends it
            let results = model.embed_patches(cuts());
            let n = results.len();
            if let Some(Err(e)) = results.iter().find(|r| r.is_err()) {
                panic!("a photo failed: {}", &e[..e.len().min(300)]);
            }
            let all = t.elapsed().as_secs_f32();
            eprintln!("GEMMA {n} photos: encoder {encode:.2} s, encoder + text model {all:.2} s => {:.2} photos/s ({} tokens)", n as f32 / all, features.len() / 512);
        }
    }

    /// Same token ids and (nearly) the same vectors as the Python / PyTorch side (tests/query_vectors.json).
    #[test]
    fn queries_match_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("tests/query_vectors.json")).unwrap()).unwrap();
        assert_eq!(fixture["prompt"].as_str().unwrap(), QUERY_PROMPT);
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
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
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("tests/image_vectors.json")).unwrap()).unwrap();
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
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

    /// Photos embedded together (their token lists padded) give the same vectors as each one alone.
    #[test]
    fn together_is_the_same_as_alone() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("maps/wikivoyage.photomap/originals");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let imgs: Vec<_> = files.iter().take(8).filter_map(|f| crate::photo::load(f)).collect();
        let copy = |c: &Patches| Patches { pixels: c.pixels.clone(), positions: c.positions.clone(), soft_tokens: c.soft_tokens };
        let cuts: Vec<Patches> = imgs.iter().map(|i| patches(i).unwrap()).collect();
        assert!(cuts.iter().map(|c| c.soft_tokens).collect::<std::collections::HashSet<_>>().len() > 1, "the photos should differ in length");
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
        let together = model.embed_patches(cuts.iter().take(4).map(&copy).collect());
        for (i, got) in together.into_iter().enumerate() {
            let alone = model.embed_patches(vec![copy(&cuts[i])]).pop().unwrap().unwrap();
            let cos: f32 = alone.iter().zip(got.unwrap()).map(|(a, b)| a * b).sum();
            eprintln!("photo {i}: {} soft tokens, cosine together/alone {cos:.6}", cuts[i].soft_tokens);
            assert!(cos > 0.9995, "photo {i}: {cos}");
        }
    }

    /// How fast the CPU alone can decode photos and cut them into patches (what feeds the GPU), one at a time and on all cores
    /// (run by hand: cargo test --release cpu_side_speed -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn cpu_side_speed() {
        use rayon::prelude::*;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for name in ["data/wikiart/images", "maps/wikivoyage.photomap/originals"] {
            let mut files: Vec<_> = std::fs::read_dir(root.join(name)).unwrap().flatten().map(|e| e.path()).collect();
            files.sort();
            let files: Vec<_> = files.into_iter().step_by(37).take(64).collect();
            let bytes: u64 = files.iter().map(|f| std::fs::metadata(f).unwrap().len()).sum();
            let work = |f: &PathBuf| crate::photo::load(f).map(|i| (i.width(), patches(&i).is_ok()));
            let started = std::time::Instant::now();
            let sizes: Vec<_> = files.iter().filter_map(work).collect();
            let one = started.elapsed().as_secs_f32();
            let started = std::time::Instant::now();
            let _: Vec<_> = files.par_iter().filter_map(work).collect();
            let all = started.elapsed().as_secs_f32();
            let width = sizes.iter().map(|s| s.0 as f32).sum::<f32>() / sizes.len() as f32;
            eprintln!(
                "{name}: {} photos, {:.0} KB and {width:.0} px wide on average; one core {:.1} photos/s ({:.0} ms each), all cores {:.1} photos/s",
                sizes.len(),
                bytes as f32 / 1024.0 / files.len() as f32,
                sizes.len() as f32 / one,
                one * 1000.0 / sizes.len() as f32,
                sizes.len() as f32 / all
            );
        }
    }

    /// The same for a 24-megapixel camera JPEG (made from a painting, scaled up), one core and all cores.
    #[test]
    #[ignore]
    fn cpu_side_speed_camera_photo() {
        use rayon::prelude::*;
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files: Vec<_> = std::fs::read_dir(root.join("data/wikiart/images")).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let base = crate::photo::load(&files[100]).unwrap();
        let big = image::imageops::resize(&base, 6000, 4000, image::imageops::FilterType::Triangle);
        let dir = root.join(".cache/cpu_test");
        std::fs::create_dir_all(&dir).unwrap();
        let rgb = image::DynamicImage::ImageRgba8(big).to_rgb8();
        let mut paths = Vec::new();
        for i in 0..12 {
            let path = dir.join(format!("{i}.jpg"));
            image::codecs::jpeg::JpegEncoder::new_with_quality(std::fs::File::create(&path).unwrap(), 90).encode_image(&rgb).unwrap();
            paths.push(path);
        }
        let kb = std::fs::metadata(&paths[0]).unwrap().len() / 1024;
        let full = |f: &PathBuf| crate::photo::load(f).map(|i| patches(&i).is_ok());
        let scaled = |f: &PathBuf| crate::photo::load_for(f, longest_side_needed).map(|i| patches(&i).is_ok());
        for (name, work) in [("full decode", &full as &(dyn Fn(&PathBuf) -> Option<bool> + Sync)), ("scaled decode", &scaled)] {
            let started = std::time::Instant::now();
            let _: Vec<_> = paths.iter().take(3).filter_map(work).collect();
            let one = started.elapsed().as_secs_f32() / 3.0;
            let started = std::time::Instant::now();
            let _: Vec<_> = paths.par_iter().filter_map(work).collect();
            let all = started.elapsed().as_secs_f32();
            eprintln!("24 MP JPEG ({kb} KB), {name}: one core {:.1} photos/s ({:.0} ms each), all cores {:.1} photos/s", 1.0 / one, one * 1000.0, paths.len() as f32 / all);
        }
        // the vectors from both ways
        let small = crate::photo::load_for(&paths[0], longest_side_needed).unwrap();
        eprintln!("scaled decode gave {} x {} (the full photo is 6000 x 4000)", small.width(), small.height());
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
        let a = model.embed_image(&crate::photo::load(&paths[0]).unwrap()).unwrap();
        let b = model.embed_image(&small).unwrap();
        let cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        eprintln!("cosine between the vectors of the full and the scaled decode: {cos:.5}");
        assert!(cos > 0.998, "{cos}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// How long freeing the model takes, and whether the process ends cleanly after it (run by hand, alone:
    /// cargo test --release closing_is_clean -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn closing_is_clean() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("maps/wikivoyage.photomap/originals");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let cuts: Vec<Patches> = files.iter().take(8).filter_map(|f| crate::photo::load(f)).map(|i| patches(&i).unwrap()).collect();
        let mut model = Gemma::load_with(&Runtime::start(root).unwrap(), root).unwrap();
        for r in model.embed_patches(cuts) {
            r.unwrap();
        }
        eprintln!("model used; freeing it");
        let started = std::time::Instant::now();
        drop(model);
        eprintln!("model freed in {:.2} s", started.elapsed().as_secs_f32());
    }

    /// Where the time goes for real photos in groups of 4: the photo encoder, or the text model after it (run by hand:
    /// cargo test --release where_the_time_goes -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn where_the_time_goes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("maps/wikivoyage.photomap/originals");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let cuts: Vec<Patches> = files.iter().take(96).filter_map(|f| crate::photo::load(f)).map(|i| patches(&i).unwrap()).collect();
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
        let copy = |chunk: &[Patches]| -> Vec<Patches> { chunk.iter().map(|c| Patches { pixels: c.pixels.clone(), positions: c.positions.clone(), soft_tokens: c.soft_tokens }).collect() };
        model.embed_patches(copy(&cuts[..4])); // warm up
        let (mut encoder, mut text) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for chunk in cuts.chunks(4) {
            let soft: Vec<usize> = chunk.iter().map(|c| c.soft_tokens).collect();
            let started = std::time::Instant::now();
            let features = model.encode_patches(copy(chunk)).unwrap();
            encoder += started.elapsed();
            let started = std::time::Instant::now();
            let mut start = 0;
            let photos: Vec<(Vec<i64>, Vec<f32>)> = soft
                .into_iter()
                .map(|tokens| {
                    let mut ids = vec![BOS, BOI];
                    ids.extend(std::iter::repeat_n(SOFT, tokens));
                    ids.extend([EOI, EOS]);
                    let part = features[start * 512..(start + tokens) * 512].to_vec();
                    start += tokens;
                    (ids, part)
                })
                .collect();
            Gemma::run_together(&mut model.session, photos).unwrap();
            text += started.elapsed();
        }
        let n = cuts.len() as f32;
        eprintln!("per photo: encoder {:.1} ms, text model {:.1} ms, together {:.1} photos/s", encoder.as_secs_f32() * 1000.0 / n, text.as_secs_f32() * 1000.0 / n, n / (encoder + text).as_secs_f32());
    }

    /// Speed of embedding real photos at different batch sizes (run by hand: cargo test --release batch_sizes -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn batch_sizes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("maps/wikivoyage.photomap/originals");
        let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().flatten().map(|e| e.path()).collect();
        files.sort();
        let cuts: Vec<Patches> = files.iter().take(96).filter_map(|f| crate::photo::load(f)).map(|i| patches(&i).unwrap()).collect();
        let shared = shared_model();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap().gemma();
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
