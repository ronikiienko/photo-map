//! SigLIP 2 (photo encoder and text encoder, two separate networks that were trained to put a photo near the words that describe it)
//! through ONNX Runtime: .cache/onnx/<model>/ has the publisher's files (onnx-community: onnx/vision_model_fp16.onnx,
//! onnx/text_model_fp16.onnx, tokenizer.json; outputs pooler_output). Our own earlier export (image_features, text_features) works too.
//!
//! A photo is squeezed into a square (its shape is not kept), scaled to -1..1; the photo encoder runs on TensorRT when it can (its
//! libraries are in runtime/tensorrt/) and on the CUDA provider otherwise. A search runs on the processor (words in ~60 ms, a photo in
//! ~150 ms on a Ryzen 5 5600H), so that browsing a map never starts CUDA and the card's memory stays the map's. The text goes in lower case, cut to 64 tokens with an
//! end mark and padded to 64 by the tokenizer's own settings. The text encoder is fed the tokens' rows, read here from its token table
//! in the publisher's file (mapped, so only the rows read are loaded), not token numbers: the table never goes on the card
//! (patch.rs SIGLIP_TEXT_ROWS).

use crate::embed::{Result, Runtime};
use crate::models::Spec;
use image::RgbaImage;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

const TEXT_FILE: &str = crate::patch::SIGLIP_TEXT;
const VISION_FILE: &str = "onnx/vision_model_fp16.onnx";

/// How many tokens the text encoder takes (what it was trained with).
const TOKENS: usize = 64;
/// The most photos TensorRT's engine takes at once (it is built for 1 to this many).
const BATCH: usize = 4;

pub struct Siglip {
    id: &'static str,
    // the environment must outlive the sessions
    env: ort::environment::Environment,
    /// A build's photo encoder, opened with the first photo (it costs video memory, and TensorRT builds its engine then).
    vision: Option<Session>,
    /// A search's photo encoder, on the processor, when the build's is on the card: a search never starts CUDA.
    vision_search: Option<Session>,
    /// The text encoder on the processor: a search's words.
    text: Session,
    /// The text encoder on the card, opened for a build's vocabulary (thousands of phrases).
    text_build: Option<Session>,
    text_file: PathBuf,
    /// The token table the text encoder's rows come from.
    rows: Rows,
    tokenizer: Tokenizer,
    dim: usize,
    size: u32,
    model_dir: PathBuf,
    cache: PathBuf,
    tensorrt: bool,
    apple: Option<ort::ep::coreml::ComputeUnits>,
    root: PathBuf,
    /// Does a build run on the card (or Apple's chip)?
    pub on_gpu: bool,
    /// Can a build's sessions use CUDA?
    cuda: bool,
}

/// A token table inside the publisher's text encoder file, mapped: a row is `width` numbers (float16) per token.
struct Rows {
    file: memmap2::Mmap,
    at: std::ops::Range<usize>,
    width: usize,
}

impl Rows {
    /// The table that the patched network `patched` names, in the publisher's file `source`.
    fn open(patched: &Path, source: &Path, width: usize) -> Result<Self> {
        let model = crate::patch::load(patched)?;
        let name = model.metadata_props.iter().find(|p| p.key == crate::patch::ROWS_TABLE).ok_or("the patched text encoder doesn't name its token table")?.value.clone();
        let file = std::fs::File::open(source).map_err(|e| format!("reading {}: {e}", source.display()))?;
        // SAFETY: the model's files are not changed while the program runs
        let file = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("reading {}: {e}", source.display()))?;
        let at = crate::patch::weight_ranges(&file)?.remove(&name).ok_or_else(|| format!("{} has no {name}", source.display()))?;
        if at.len() % (width * 2) != 0 {
            return Err(format!("the token table is {} bytes, not rows of {width} float16 numbers", at.len()));
        }
        Ok(Self { file, at, width })
    }

    /// The tokens' rows one after the other, as float32.
    fn of(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let row_bytes = self.width * 2;
        let mut out = Vec::with_capacity(ids.len() * self.width);
        for &id in ids {
            let start = self.at.start + id as usize * row_bytes;
            if start + row_bytes > self.at.end {
                return Err(format!("token {id} is past the end of the token table"));
            }
            out.extend(self.file[start..start + row_bytes].chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()));
        }
        Ok(out)
    }
}

fn err(what: &str) -> impl Fn(ort::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// How long the longer side of a photo must at least be for `prepare`: the shorter side has to be a full `size`, since the
/// photo is squeezed into a square and a smaller one would be stretched.
pub fn longest_side_needed(size: u32, width: u32, height: u32) -> u32 {
    let (long, short) = (width.max(height) as u64, width.min(height).max(1) as u64);
    ((size as u64 * long).div_ceil(short)) as u32
}

/// The photo as the encoder takes it: `size` x `size`, red plane then green then blue, each pixel 0..255 scaled to -1..1.
pub fn prepare(img: &RgbaImage, size: u32) -> Result<Vec<f32>> {
    if img.width() == 0 || img.height() == 0 {
        return Err("empty photo".into());
    }
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::{Image, ImageRef}};
    let from = ImageRef::new(img.width(), img.height(), img.as_raw(), PixelType::U8x4).map_err(|e| e.to_string())?;
    let mut to = Image::new(size, size, PixelType::U8x4);
    // the bilinear filter, widened when the photo shrinks, is the triangle filter: every source pixel counts
    Resizer::new().resize(&from, &mut to, &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear))).map_err(|e| e.to_string())?;
    let plane = (size * size) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (i, p) in to.buffer().chunks_exact(4).enumerate() {
        for c in 0..3 {
            out[c * plane + i] = p[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(out)
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

impl Siglip {
    pub fn load(rt: &Runtime, root: &Path, spec: &'static Spec, size: u32) -> Result<Self> {
        let model_dir = spec.dir(root);
        if !spec.installed(root) {
            return Err(format!("{} is not downloaded (its files go in {})", spec.label, model_dir.display()));
        }
        let env = rt.env.clone();
        // the text encoder without its token table, fed rows read here (patch.rs)
        let text_file = crate::patch::ensure(&model_dir, crate::patch::SIGLIP_TEXT_ROWS)?;
        let rows = Rows::open(&text_file, &model_dir.join(TEXT_FILE), spec.dim)?;
        let text = Session::builder(&env).and_then(|mut b| b.commit_from_file(&text_file)).map_err(err("loading the text encoder"))?;
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| format!("loading the tokenizer: {e}"))?;
        tokenizer.with_truncation(Some(TruncationParams { max_length: TOKENS, ..Default::default() })).map_err(|e| e.to_string())?;
        // padded to 64 as the model was trained, whatever the tokenizer file says
        let pad_id = tokenizer.token_to_id("<pad>").ok_or("the tokenizer has no <pad>")?;
        tokenizer.with_padding(Some(PaddingParams { strategy: PaddingStrategy::Fixed(TOKENS), pad_id, pad_token: "<pad>".into(), ..Default::default() }));
        let mut model = Self {
            id: spec.id,
            env,
            vision: None,
            vision_search: None,
            text,
            text_build: None,
            text_file,
            rows,
            tokenizer,
            dim: spec.dim,
            size,
            cache: root.join(".cache/trt").join(spec.folder),
            model_dir,
            tensorrt: rt.tensorrt,
            apple: rt.apple,
            root: root.to_path_buf(),
            on_gpu: rt.on_gpu || rt.apple.is_some(),
            cuda: rt.on_gpu,
        };
        // CoreML prepares the network for the chip when it is opened (minutes the first time): here, where a map being made can still be paused
        if model.apple.is_some() {
            model.open_vision()?;
        }
        Ok(model)
    }

    pub fn id(&self) -> &'static str {
        self.id
    }

    /// A search's words: their vector, unit length (on the processor).
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_text(text, false)
    }

    /// A build's phrase: its vector, unit length (on the card if a build runs there).
    pub fn embed_phrase(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_text(text, self.cuda)
    }

    fn embed_text(&mut self, text: &str, on_card: bool) -> Result<Vec<f32>> {
        if on_card && self.text_build.is_none() {
            self.text_build = Some(Session::builder(&self.env).and_then(|b| b.with_execution_providers([crate::embed::cuda()]).map_err(|e| e.into())).and_then(|mut b| b.commit_from_file(&self.text_file)).map_err(err("loading the text encoder"))?);
        }
        let encoding = self.tokenizer.encode(text.to_lowercase(), true).map_err(|e| e.to_string())?;
        let ids = encoding.get_ids();
        if ids.len() != TOKENS {
            return Err(format!("the tokenizer made {} tokens, expected {TOKENS}", ids.len()));
        }
        let rows = Tensor::from_array(([1usize, TOKENS, self.rows.width], self.rows.of(ids)?)).map_err(err("tensor"))?;
        let session = if on_card { self.text_build.as_mut().unwrap() } else { &mut self.text };
        let outputs = session.run(ort::inputs![crate::patch::ROWS_INPUT => rows]).map_err(err("running the text encoder"))?;
        let (_, data) = (outputs.get("text_features").or_else(|| outputs.get("pooler_output"))).ok_or("the text encoder has no text_features or pooler_output")?.try_extract_tensor::<f32>().map_err(err("reading the result"))?;
        Ok(unit(data.to_vec()))
    }

    /// Is the text longer than the 64 tokens the encoder reads?
    pub fn query_cut(&self, text: &str) -> bool {
        self.tokenizer.encode(text.to_lowercase(), true).is_ok_and(|e| !e.get_overflowing().is_empty())
    }

    /// A search's photo: its vector, unit length (on the processor when a build would use CUDA).
    pub fn embed_image(&mut self, img: &RgbaImage) -> Result<Vec<f32>> {
        let pixels = prepare(img, self.size)?;
        if !self.cuda {
            return self.embed_prepared(vec![pixels]).pop().unwrap();
        }
        if self.vision_search.is_none() {
            let file = crate::patch::ensure_or(&self.model_dir, crate::patch::SIGLIP_PROCESSOR, VISION_FILE);
            self.vision_search = Some(Session::builder(&self.env).and_then(|mut b| b.commit_from_file(&file)).map_err(err("loading the photo encoder"))?);
        }
        let side = self.size as usize;
        let outputs = self.vision_search.as_mut().unwrap().run(ort::inputs!["pixel_values" => Tensor::from_array(([1, 3, side, side], pixels)).map_err(err("tensor"))?]).map_err(err("running the photo encoder"))?;
        let (_, data) = (outputs.get("image_features").or_else(|| outputs.get("pooler_output"))).ok_or("the photo encoder has no image_features or pooler_output")?.try_extract_tensor::<f32>().map_err(err("reading the result"))?;
        if data.len() != self.dim {
            return Err(format!("the photo encoder made {} numbers for a photo", data.len()));
        }
        Ok(unit(data.to_vec()))
    }

    /// A build's photos, prepared (see `prepare`), a few at a time.
    pub fn embed_prepared(&mut self, photos: Vec<Vec<f32>>) -> Vec<Result<Vec<f32>>> {
        let n = photos.len();
        let mut out: Vec<Result<Vec<f32>>> = Vec::with_capacity(n);
        let mut photos = photos.into_iter().peekable();
        while photos.peek().is_some() {
            let group: Vec<Vec<f32>> = photos.by_ref().take(BATCH).collect();
            let count = group.len();
            match self.encode(group) {
                Ok(rows) => out.extend(rows.into_iter().map(Ok)),
                Err(e) => out.extend((0..count).map(|_| Err(e.clone()))),
            }
        }
        out
    }

    fn encode(&mut self, group: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
        if self.vision.is_none() {
            self.open_vision()?;
        }
        let batch = group.len();
        let side = self.size as usize;
        let mut pixels: Vec<f32> = group.into_iter().flatten().collect();
        // with CoreML the network is made for exactly BATCH photos: a short group is filled up with blank ones
        let run_batch = if self.apple.is_some() { BATCH } else { batch };
        pixels.resize(run_batch * 3 * side * side, 0.0);
        let vision = self.vision.as_mut().unwrap();
        let outputs = vision.run(ort::inputs!["pixel_values" => Tensor::from_array(([run_batch, 3, side, side], pixels)).map_err(err("tensor"))?]).map_err(err("running the photo encoder"))?;
        let (_, data) = (outputs.get("image_features").or_else(|| outputs.get("pooler_output"))).ok_or("the photo encoder has no image_features or pooler_output")?.try_extract_tensor::<f32>().map_err(err("reading the result"))?;
        if data.len() != run_batch * self.dim {
            return Err(format!("the photo encoder made {} numbers for {run_batch} photos", data.len()));
        }
        Ok(data.chunks_exact(self.dim).take(batch).map(|row| unit(row.to_vec())).collect())
    }

    /// Let go of a build's sessions on the card (a search doesn't use them). CoreML's stay: a search uses them, and they are slow to open.
    pub fn release_build(&mut self) {
        if self.cuda {
            self.vision = None;
            self.text_build = None;
        }
    }

    /// Open a build's photo encoder now, if it isn't (TensorRT builds its engine then: minutes the first time).
    pub fn prepare_build(&mut self) -> Result<()> {
        if self.vision.is_none() {
            self.open_vision()?;
        }
        Ok(())
    }

    /// A build's photo encoder: on TensorRT if it can (its engine is built the first time, about a minute, and kept in .cache/trt/),
    /// else on the CUDA provider.
    fn open_vision(&mut self) -> Result<()> {
        // CoreML can't take the publisher's first layer as it is: it gets a patched copy
        let file = if self.apple.is_some() {
            crate::patch::ensure(&self.model_dir, crate::patch::SIGLIP_COREML)?
        } else if !self.cuda {
            // float32 is faster on the processor
            crate::patch::ensure_or(&self.model_dir, crate::patch::SIGLIP_PROCESSOR, VISION_FILE)
        } else {
            self.model_dir.join(VISION_FILE)
        };
        if self.tensorrt {
            // TensorRT gets its norms in one piece (patch.rs)
            match crate::patch::ensure(&self.model_dir, crate::patch::SIGLIP_TENSORRT).and_then(|f| self.open_tensorrt(&f)) {
                Ok(session) => {
                    self.vision = Some(session);
                    return Ok(());
                }
                Err(e) => {
                    eprintln!("TensorRT is off, the photo encoder runs on CUDA: {e}");
                    self.tensorrt = false;
                }
            }
        }
        let cache = self.root.join(".cache/coreml").join(self.model_dir.file_name().unwrap_or_default());
        let builder = crate::embed::vision_builder(&self.env, self.apple, &cache, &[("batch_size", BATCH as i64), ("num_channels", 3), ("height", self.size as i64), ("width", self.size as i64)]);
        let builder = if self.cuda { builder.and_then(|b| b.with_execution_providers([crate::embed::cuda()]).map_err(|e| e.into())) } else { builder };
        self.vision = Some(builder.and_then(|mut b| b.commit_from_file(&file)).map_err(err("loading the photo encoder"))?);
        Ok(())
    }

    fn open_tensorrt(&self, file: &Path) -> Result<Session> {
        std::fs::create_dir_all(&self.cache).map_err(|e| e.to_string())?;
        let side = self.size;
        let shapes = |n: usize| format!("pixel_values:{n}x3x{side}x{side}");
        let trt = ep::TensorRT::default()
            .with_fp16(true)
            .with_max_workspace_size(1 << 30) // the card is shared with the map: keep the engine's scratch space small
            .with_engine_cache(true)
            .with_engine_cache_path(self.cache.display())
            .with_timing_cache(true)
            .with_timing_cache_path(self.cache.display())
            .with_profile_min_shapes(shapes(1))
            .with_profile_opt_shapes(shapes(BATCH))
            .with_profile_max_shapes(shapes(BATCH))
            .build();
        // the CUDA provider takes what TensorRT leaves
        Session::builder(&self.env)
            .and_then(|b| b.with_execution_providers([trt, crate::embed::cuda()]).map_err(|e| e.into()))
            .and_then(|mut b| b.commit_from_file(file))
            .map_err(err("loading the photo encoder on TensorRT"))
    }
}

#[cfg(test)]
mod tests {
    use crate::models::{self, Model, SIGLIP_BASE_512};

    /// The text encoder fed rows from the mapped table (patch.rs SIGLIP_TEXT_ROWS) gives the vectors the publisher's network gives
    /// (SigLIP 256 has to be downloaded; the patch is made in its folder): cargo test --release text_rows -- --ignored --nocapture
    #[test]
    #[ignore]
    fn text_rows_give_the_publishers_vectors() {
        let root = crate::data_root();
        let spec = models::spec(models::SIGLIP_BASE_256).unwrap();
        let rt = crate::embed::Runtime::shared(&root).unwrap();
        let mut model = super::Siglip::load(&rt, &root, spec, spec.size).unwrap();
        let mut publisher = ort::session::Session::builder(&rt.env).unwrap().commit_from_file(spec.dir(&root).join(super::TEXT_FILE)).unwrap();
        for text in ["a dog on a beach", "street at night NOT cars", "собака без повідка", "東京の夜景", ""] {
            let ours = model.embed_query(text).unwrap();
            let ids: Vec<i64> = model.tokenizer.encode(text.to_lowercase(), true).unwrap().get_ids().iter().map(|&i| i as i64).collect();
            let out = publisher.run(ort::inputs!["input_ids" => ort::value::Tensor::from_array(([1usize, super::TOKENS], ids)).unwrap()]).unwrap();
            let theirs = super::unit(out["pooler_output"].try_extract_tensor::<f32>().unwrap().1.to_vec());
            let worst = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            eprintln!("{text:?}: largest difference {worst:e}");
            assert!(worst < 1e-3, "{text:?}");
        }
    }

    /// Models can be swapped in the hub and each works after its turn.
    #[test]
    #[ignore]
    fn models_swap() {
        let hub = models::test_hub();
        for id in [SIGLIP_BASE_512, models::SIGLIP_BASE_256, SIGLIP_BASE_512] {
            assert_eq!(hub.wait_for(id, &|| false), Ok(true));
            let dim = hub.with(id, |m: &mut Model| m.embed_query("a dog on a beach").unwrap().len()).unwrap();
            assert_eq!(dim, models::spec(id).unwrap().dim, "{id}");
            assert!(hub.with("some/other", |_| ()).is_err(), "a model that isn't loaded is refused");
        }
    }

    /// How fast the processor alone can read photos and make them ready for the model (what feeds the card), one at a time and on all
    /// cores (run by hand: cargo test --release siglip_cpu_side_speed -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn siglip_cpu_side_speed() {
        use rayon::prelude::*;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        for (id, name) in [SIGLIP_BASE_512, models::SIGLIP_BASE_256].into_iter().flat_map(|id| ["data/rare-species/images", "data/wikivoyage/images_zim"].map(|n| (id, n))) {
            let spec = models::spec(id).unwrap();
            let mut files: Vec<_> = std::fs::read_dir(root.join(name)).unwrap().flatten().map(|e| e.path()).collect();
            files.sort();
            let files: Vec<_> = files.into_iter().step_by(23).take(96).collect();
            let bytes: u64 = files.iter().map(|f| std::fs::metadata(f).unwrap().len()).sum();
            let (mut read, mut prepare, mut small) = (0f32, 0f32, 0f32);
            let one = std::time::Instant::now();
            let mut done = 0;
            let mut width = 0u32;
            for f in &files {
                let t = std::time::Instant::now();
                let Some(img) = crate::photo::load_for(f, |w, h| spec.longest_side_needed(w, h)) else { continue };
                read += t.elapsed().as_secs_f32();
                width += img.width();
                let t = std::time::Instant::now();
                spec.prepare(&img).unwrap();
                prepare += t.elapsed().as_secs_f32();
                let t = std::time::Instant::now();
                crate::build::thumbnail(&img);
                small += t.elapsed().as_secs_f32();
                done += 1;
            }
            let one = one.elapsed().as_secs_f32();
            let started = std::time::Instant::now();
            let n: usize = files.par_iter().filter_map(|f| crate::photo::load_for(f, |w, h| spec.longest_side_needed(w, h)).map(|i| spec.prepare(&i).unwrap())).count();
            let all = started.elapsed().as_secs_f32();
            eprintln!(
                "{}, {name}: {done} photos, {:.0} KB each, {:.0} px wide after reading; one core {:.1} photos/s (read {:.0} ms, prepare {:.0} ms, 64 px copy {:.0} ms each), all cores {:.1} photos/s",
                spec.label,
                bytes as f32 / 1024.0 / files.len() as f32,
                width as f32 / done as f32,
                done as f32 / one,
                read * 1000.0 / done as f32,
                prepare * 1000.0 / done as f32,
                small * 1000.0 / done as f32,
                n as f32 / all
            );
        }
    }

    /// How fast the card embeds real photos already made ready, in groups of 4 as the build does (run by hand with the viewer closed:
    /// cargo test --release siglip_embed_speed -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn siglip_embed_speed() {
        use rayon::prelude::*;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let hub = models::test_hub();
        for (id, name) in [SIGLIP_BASE_512, models::SIGLIP_BASE_256].into_iter().flat_map(|id| ["data/rare-species/images", "data/wikivoyage/images_zim"].map(|n| (id, n))) {
            let spec = models::spec(id).unwrap();
            if !spec.installed(&crate::data_root()) {
                continue;
            }
            assert_eq!(hub.wait_for(id, &|| false), Ok(true));
            let mut files: Vec<_> = std::fs::read_dir(root.join(name)).unwrap().flatten().map(|e| e.path()).collect();
            files.sort();
            let files: Vec<_> = files.into_iter().step_by(11).take(160).collect();
            let started = std::time::Instant::now();
            let mut prepared: Vec<_> = files.par_iter().filter_map(|f| crate::photo::load_for(f, |w, h| spec.longest_side_needed(w, h)).map(|i| spec.prepare(&i).unwrap())).collect();
            let prep = started.elapsed().as_secs_f32();
            let n = prepared.len();
            hub.with(id, |m| {
                let warm: Vec<_> = prepared.drain(..4).collect();
                m.embed_prepared(warm); // the engine loads on the first photos
                let rest = prepared.len();
                let started = std::time::Instant::now();
                while !prepared.is_empty() {
                    let group: Vec<_> = prepared.drain(..4.min(prepared.len())).collect();
                    for r in m.embed_prepared(group) {
                        r.unwrap();
                    }
                }
                eprintln!("{}, {name}: {n} photos made ready in {prep:.1} s ({:.0} photos/s); the card alone, groups of 4: {:.1} photos/s", spec.label, n as f32 / prep, rest as f32 / started.elapsed().as_secs_f32());
            })
            .unwrap();
        }
    }
}


