//! The models a map can be made with, and the one that is loaded.
//!
//! A map is made with one model and searched with the same one: its manifest says which (`embedding.model`), and vectors of one
//! model mean nothing next to another's, so nothing is shared between models - not the vectors an earlier map holds (reuse.rs), not
//! the vocabulary's (naming.rs). Only one model is on the graphics card at a time; the `Hub` loads the one that is asked for.

use crate::embed::{self, Gemma, Runtime};
use crate::siglip::Siglip;
use image::RgbaImage;
use std::path::Path;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

pub const GEMMA: &str = "google/embeddinggemma-2";
pub const SIGLIP_BASE_512: &str = "google/siglip2-base-patch16-512";
pub const SIGLIP_BASE_256: &str = "google/siglip2-base-patch16-256";
/// What a map or a build made before models could be chosen was made with.
pub const DEFAULT: &str = GEMMA;

pub enum Kind {
    Gemma,
    /// Photos are squeezed into a square of this many pixels a side.
    Siglip { size: u32 },
}

pub struct Spec {
    pub id: &'static str,
    pub label: &'static str,
    /// A line on what it is good for, for the model choice.
    pub note: &'static str,
    /// How many numbers a photo's vector has.
    pub dim: usize,
    pub kind: Kind,
    /// The most tokens of a search text it reads (about 3 or 4 words in 5); None: more than any search is.
    pub max_tokens: Option<usize>,
    /// What the search box says for a map made with it.
    pub query_hint: &'static str,
    /// Where its files come from: a Hugging Face repository at a fixed revision. They are kept under .cache/onnx/<folder>/ with
    /// the paths they have there.
    pub repo: &'static str,
    pub revision: &'static str,
    pub folder: &'static str,
    pub files: &'static [File],
}

/// A file of a model as its publisher has it: refused if it is not exactly this.
pub struct File {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

const fn file(path: &'static str, size: u64, sha256: &'static str) -> File {
    File { path, size, sha256 }
}

pub const ALL: [Spec; 3] = [
    Spec {
        id: GEMMA,
        label: "EmbeddingGemma 2",
        note: "Best quality. Words, photos and long texts share one space.",
        dim: 768,
        kind: Kind::Gemma,
        max_tokens: None,
        query_hint: "Search the map: describe what you look for, or drop in a photo",
        repo: "onnx-community/embeddinggemma-2-ONNX",
        revision: "daa72c51243991dfcaf9f9137d2c573d8f7790c0",
        folder: "embeddinggemma-2",
        files: &[
            file("tokenizer.json", 32170510, "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"),
            file("onnx/model_fp16.onnx", 425780, "a49e227d7e0e5f4ee606d79879d084264366367bc6318f344ab179e35d4fd6a4"),
            file("onnx/model_fp16.onnx_data", 542085120, "b9dbe09415d77c8686ba6e216a244086f4ada1608238c13f2bfaa587ba627207"),
            file("onnx/vision_encoder_fp16.onnx", 112253, "3e2b6d5648a3ae61b572cc3ec0a23b13ebfda3b63e3c9c343c593636392a879c"),
            file("onnx/vision_encoder_fp16.onnx_data", 335513088, "9f374070808b8b96252f7d04c9c4eae34df60283473fde5a3fbf4d9205495e19"),
        ],
    },
    Spec {
        id: SIGLIP_BASE_512,
        label: "SigLIP 2 Base (512 px)",
        note: "About 2 to 3 times faster than Gemma, a little less accurate. Short words and photos only.",
        dim: 768,
        kind: Kind::Siglip { size: 512 },
        max_tokens: Some(64),
        query_hint: "Search the map: a few words on what you look for, or drop in a photo",
        repo: "onnx-community/siglip2-base-patch16-512-ONNX",
        revision: "323fe16b35d8ced8182f8d9ff5e5ec935dd79bd7",
        folder: "siglip2-base-patch16-512",
        files: &[
            file("tokenizer.json", 34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
            file("onnx/text_model_fp16.onnx", 564862230, "0026e285ac559977f36afae61dde214c75475007822eb8a4dd544ffc22f5634b"),
            file("onnx/vision_model_fp16.onnx", 187312043, "70cad97773fbda5d8edbd7df3abd6023346e3ce4bf18747557d28dde570bdec8"),
        ],
    },
    Spec {
        id: SIGLIP_BASE_256,
        label: "SigLIP 2 Base (256 px)",
        note: "The fastest: photos are looked at in a quarter of the detail, so it misses more. Short words and photos only.",
        dim: 768,
        kind: Kind::Siglip { size: 256 },
        max_tokens: Some(64),
        query_hint: "Search the map: a few words on what you look for, or drop in a photo",
        repo: "onnx-community/siglip2-base-patch16-256-ONNX",
        revision: "d1114256522a37ffa257a0a58017348ab0058db2",
        folder: "siglip2-base-patch16-256",
        files: &[
            file("tokenizer.json", 34363039, "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322"),
            file("onnx/text_model_fp16.onnx", 564862230, "80954edffdc689599e5d5bc6a1738380bc9e8139a18e5c8892485f248b6b4890"),
            file("onnx/vision_model_fp16.onnx", 186131676, "fe9ad8020a6d3d98d394c9be8f07064066135fc2f87ec11692de0b677c0ac4db"),
        ],
    },
];

pub fn spec(id: &str) -> Option<&'static Spec> {
    ALL.iter().find(|s| s.id == id)
}

impl Spec {
    /// The folder its files are kept in.
    pub fn dir(&self, root: &Path) -> std::path::PathBuf {
        root.join(".cache/onnx").join(self.folder)
    }

    /// Are its files there? (A file is only put there once it was checked whole, so its size is enough.)
    pub fn installed(&self, root: &Path) -> bool {
        let dir = self.dir(root);
        self.files.iter().all(|f| std::fs::metadata(dir.join(f.path)).is_ok_and(|m| m.len() == f.size))
    }

    /// What its files take to download, in bytes.
    pub fn download_size(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// How long the longer side of a photo of this size must at least be for `prepare`, 0 if it can't be prepared.
    pub fn longest_side_needed(&self, width: u32, height: u32) -> u32 {
        match self.kind {
            Kind::Gemma => embed::longest_side_needed(width, height),
            Kind::Siglip { size } => crate::siglip::longest_side_needed(size, width, height),
        }
    }

    /// A photo as the model takes it.
    pub fn prepare(&self, img: &RgbaImage) -> Result<Prepared, String> {
        match self.kind {
            Kind::Gemma => embed::patches(img).map(Prepared::Gemma),
            Kind::Siglip { size } => crate::siglip::prepare(img, size).map(Prepared::Siglip),
        }
    }
}

/// A photo made ready for its model on the processor (the slow part of that is cheap to do on several cores while the card works).
pub enum Prepared {
    Gemma(embed::Patches),
    Siglip(Vec<f32>),
}

pub enum Model {
    Gemma(Gemma),
    Siglip(Siglip),
}

impl Model {
    pub fn id(&self) -> &'static str {
        match self {
            Model::Gemma(_) => GEMMA,
            Model::Siglip(s) => s.id(),
        }
    }

    pub fn on_gpu(&self) -> bool {
        match self {
            Model::Gemma(m) => m.on_gpu,
            Model::Siglip(m) => m.on_gpu,
        }
    }

    pub fn embed_query(&mut self, text: &str) -> embed::Result<Vec<f32>> {
        match self {
            Model::Gemma(m) => m.embed_query(text),
            Model::Siglip(m) => m.embed_text(text),
        }
    }

    /// Did the model cut this text short (it reads no more than its limit)? How many tokens it reads is in `Spec::max_tokens`.
    pub fn query_cut(&self, text: &str) -> bool {
        match self {
            Model::Gemma(_) => false,
            Model::Siglip(m) => m.cuts(text),
        }
    }

    /// A caption, as the text that is searched.
    pub fn embed_document(&mut self, text: &str) -> embed::Result<Vec<f32>> {
        match self {
            Model::Gemma(m) => m.embed_document(text),
            Model::Siglip(m) => m.embed_text(text),
        }
    }

    pub fn embed_image(&mut self, img: &RgbaImage) -> embed::Result<Vec<f32>> {
        match self {
            Model::Gemma(m) => m.embed_image(img),
            Model::Siglip(m) => m.embed_image(img),
        }
    }

    /// Several prepared photos at once; each result is that photo's vector, or why it failed.
    pub fn embed_prepared(&mut self, photos: Vec<Prepared>) -> Vec<embed::Result<Vec<f32>>> {
        match self {
            Model::Gemma(m) => m.embed_patches(photos.into_iter().filter_map(|p| if let Prepared::Gemma(p) = p { Some(p) } else { None }).collect()),
            Model::Siglip(m) => m.embed_pixels(photos.into_iter().filter_map(|p| if let Prepared::Siglip(p) = p { Some(p) } else { None }).collect()),
        }
    }

    /// The Gemma inside (for tests that look at Gemma's own parts).
    #[cfg(test)]
    pub fn gemma(&mut self) -> &mut Gemma {
        match self {
            Model::Gemma(m) => m,
            Model::Siglip(_) => panic!("not Gemma"),
        }
    }
}

/// The loaded model, and a thread that loads the one asked for. Everything that needs a model asks the hub for it by id, waits
/// until it is there, and then uses it under the lock of `slot`.
pub struct Hub {
    pub slot: Mutex<Option<Model>>,
    state: Mutex<State>,
    load: Mutex<Sender<String>>,
}

#[derive(Default, Clone)]
pub struct State {
    /// The id of the model in the slot (kept here too: the slot is locked for as long as a model is working).
    pub loaded: Option<String>,
    pub on_gpu: bool,
    /// Why the model asked for last couldn't be loaded.
    pub failed: Option<String>,
}

impl Hub {
    /// Start the loader thread. `wake`: called when a model has loaded (or failed), to redraw. Nothing is loaded until `want`.
    pub fn start(root: std::path::PathBuf, wake: impl Fn() + Send + 'static) -> Arc<Hub> {
        let (load, requests) = channel::<String>();
        let hub = Arc::new(Hub { slot: Mutex::new(None), state: Mutex::new(State::default()), load: Mutex::new(load) });
        let shared = hub.clone();
        std::thread::spawn(move || {
            let mut runtime: Option<Runtime> = None;
            while let Ok(id) = requests.recv() {
                if shared.state.lock().unwrap().loaded.as_deref() == Some(id.as_str()) {
                    continue;
                }
                let started = std::time::Instant::now();
                // the old one goes first: two do not fit on the card
                *shared.slot.lock().unwrap() = None;
                *shared.state.lock().unwrap() = State::default();
                let loaded = (|| {
                    let spec = spec(&id).ok_or_else(|| format!("unknown model {id}"))?;
                    if runtime.is_none() {
                        runtime = Some(Runtime::start(&root)?);
                    }
                    let rt = runtime.as_ref().unwrap();
                    match spec.kind {
                        Kind::Gemma => Gemma::load_with(rt, &root).map(Model::Gemma),
                        Kind::Siglip { size } => Siglip::load(rt, &root, spec, size).map(Model::Siglip),
                    }
                })();
                match loaded {
                    Ok(model) => {
                        eprintln!("{} ready ({}) after {:.1}s", id, if model.on_gpu() { "GPU" } else { "CPU" }, started.elapsed().as_secs_f32());
                        let on_gpu = model.on_gpu();
                        *shared.slot.lock().unwrap() = Some(model);
                        *shared.state.lock().unwrap() = State { loaded: Some(id), on_gpu, failed: None };
                    }
                    Err(e) => *shared.state.lock().unwrap() = State { failed: Some(e), ..State::default() },
                }
                wake();
            }
        });
        hub
    }

    /// Ask for a model to be loaded (nothing happens if it is). Returns at once.
    pub fn want(&self, id: &str) {
        {
            // an earlier failure was another model's (or an earlier try's): it must not be taken for this one's
            let mut state = self.state.lock().unwrap();
            if state.loaded.as_deref() != Some(id) {
                state.failed = None;
            }
        }
        let _ = self.load.lock().unwrap().send(id.to_string());
    }

    /// Let go of the loaded model (it holds memory), unless it is in use or being loaded. It is loaded again by the next `want`.
    pub fn unload(&self) {
        let Ok(mut slot) = self.slot.try_lock() else { return };
        if slot.take().is_some() {
            *self.state.lock().unwrap() = State::default();
        }
    }

    pub fn state(&self) -> State {
        self.state.lock().unwrap().clone()
    }

    /// Ask for the model and wait until it is loaded: Ok(true), or Ok(false) if `paused` said to stop waiting, or why it failed.
    pub fn wait_for(&self, id: &str, paused: &dyn Fn() -> bool) -> Result<bool, String> {
        self.want(id);
        let started = std::time::Instant::now();
        loop {
            {
                let state = self.state.lock().unwrap();
                if state.loaded.as_deref() == Some(id) {
                    return Ok(true);
                }
                // a failure is only this model's if the thread has got to it (the state is cleared when it starts on another)
                if let Some(why) = &state.failed {
                    return Err(why.clone());
                }
            }
            if paused() {
                return Ok(false);
            }
            if started.elapsed().as_secs() > 3600 {
                return Err("the model didn't load".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    /// Use the loaded model, if it is `id`.
    pub fn with<T>(&self, id: &str, f: impl FnOnce(&mut Model) -> T) -> Result<T, String> {
        match self.slot.lock().unwrap().as_mut() {
            Some(m) if m.id() == id => Ok(f(m)),
            _ => Err("the model is gone".into()),
        }
    }
}

/// A hub holding Gemma, loaded once for the tests (ONNX Runtime allows one environment in a process).
#[cfg(test)]
pub fn test_hub() -> Arc<Hub> {
    test_hub_with(GEMMA)
}

/// The tests' hub, holding this model.
#[cfg(test)]
pub fn test_hub_with(id: &str) -> Arc<Hub> {
    static HUB: std::sync::OnceLock<Arc<Hub>> = std::sync::OnceLock::new();
    let hub = HUB
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
            Hub::start(root, || {})
        })
        .clone();
    assert_eq!(hub.wait_for(id, &|| false), Ok(true));
    hub
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (dot, na, nb) = a.iter().zip(b).fold((0f32, 0f32, 0f32), |(d, x, y), (p, q)| (d + p * q, x + p * p, y + q * q));
        dot / (na.sqrt() * nb.sqrt())
    }

    /// The publisher's files, as patched for the backend in use (patch.rs), give the vectors that maps already made with the model
    /// hold, so those maps can still be searched. Models that are not downloaded are skipped. Run alone, once per backend:
    /// PHOTO_MAP_BACKEND=cpu|cuda|tensorrt cargo test --release match_the_maps -- --ignored --test-threads=1 --nocapture
    #[test]
    #[ignore]
    fn models_match_the_maps() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut hub = None;
        for (id, map) in [(SIGLIP_BASE_256, "rare-species 2"), (SIGLIP_BASE_256, "nasa"), (GEMMA, "rare-species"), (GEMMA, "everything from usb")] {
            let spec = spec(id).unwrap();
            if !spec.installed(root) {
                eprintln!("{} is not downloaded: skipped", spec.label);
                continue;
            }
            let hub = hub.get_or_insert_with(|| test_hub_with(id)).clone();
            let dir = root.join("maps").join(format!("{map}.photomap"));
            let photos: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(dir.join("photos.json")).unwrap()).unwrap();
            let stored = std::fs::read(dir.join("embeddings.bin")).unwrap();
            let rows: Vec<usize> = (0..photos.len()).step_by(photos.len() / 24).take(24).collect();
            let prepared: Vec<_> = rows
                .iter()
                .map(|&r| {
                    let source = std::path::PathBuf::from(photos[r]["source"].as_str().unwrap());
                    let path = if source.exists() { source } else { dir.join(photos[r]["file"].as_str().unwrap()) };
                    spec.prepare(&crate::photo::load_for(&path, |w, h| spec.longest_side_needed(w, h)).unwrap()).unwrap()
                })
                .collect();
            assert_eq!(hub.wait_for(id, &|| false), Ok(true));
            let got = hub.with(id, |m| m.embed_prepared(prepared)).unwrap();
            let mut all: Vec<f32> = rows
                .iter()
                .zip(got)
                .map(|(&r, got)| {
                    let want: Vec<f32> = stored[r * spec.dim * 2..(r + 1) * spec.dim * 2].chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
                    cosine(&got.unwrap(), &want)
                })
                .collect();
            all.sort_by(f32::total_cmp);
            eprintln!("{} on {map}: cosine to the map's vectors over {} photos: lowest {:.5}, median {:.5}", spec.label, rows.len(), all[0], all[all.len() / 2]);
            assert!(all[0] > 0.99, "{}", all[0]);
        }
    }

    /// For machines without the maps above (a Mac): every downloaded model's vectors for 24 photos of the folder PHOTO_MAP_PHOTOS
    /// and a few search words, on the backend in use, kept in .cache/test/agree/; on any backend but the processor they are compared
    /// with the processor's. Run with cpu first, then each other backend:
    /// PHOTO_MAP_PHOTOS=~/Pictures PHOTO_MAP_BACKEND=cpu cargo test --release backends_agree -- --ignored --test-threads=1 --nocapture
    #[test]
    #[ignore]
    fn backends_agree() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let folder = std::path::PathBuf::from(std::env::var("PHOTO_MAP_PHOTOS").expect("PHOTO_MAP_PHOTOS: a folder of photos"));
        let backend = format!("{:?}", crate::backends::Choice::load(root).selected(&crate::runtime::Setup::detect(root)));
        let mut files: Vec<_> = std::fs::read_dir(&folder).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| ["jpg", "jpeg", "png", "webp"].contains(&e.to_string_lossy().to_lowercase().as_str()))).collect();
        files.sort();
        let files: Vec<_> = files.iter().step_by((files.len() / 24).max(1)).take(24).cloned().collect();
        let words = ["a dog on a beach", "mountains at sunset", "a plate of food", "a crowded street at night"];
        let keep = root.join(".cache/test/agree");
        std::fs::create_dir_all(&keep).unwrap();
        let mut hub = None;
        for spec in ALL.iter().filter(|s| s.installed(root)) {
            let hub = hub.get_or_insert_with(|| test_hub_with(spec.id)).clone();
            assert_eq!(hub.wait_for(spec.id, &|| false), Ok(true));
            let prepared: Vec<_> = files.iter().map(|f| spec.prepare(&crate::photo::load_for(f, |w, h| spec.longest_side_needed(w, h)).unwrap()).unwrap()).collect();
            let started = std::time::Instant::now();
            let photos: Vec<Vec<f32>> = hub.with(spec.id, |m| m.embed_prepared(prepared)).unwrap().into_iter().map(|v| v.unwrap()).collect();
            let speed = files.len() as f32 / started.elapsed().as_secs_f32();
            let texts: Vec<Vec<f32>> = words.iter().map(|w| hub.with(spec.id, |m| m.embed_query(w)).unwrap().unwrap()).collect();
            let name = |b: &str| keep.join(format!("{}.{b}.json", crate::naming::slug(spec.id)));
            std::fs::write(name(&backend), serde_json::to_vec(&serde_json::json!({ "files": files, "photos": photos, "words": texts })).unwrap()).unwrap();
            eprintln!("{} on {backend}: {speed:.1} photos/s (with the first run's preparing)", spec.label);
            if backend == "Cpu" {
                continue;
            }
            let Ok(cpu) = std::fs::read(name("Cpu")) else {
                eprintln!("  no processor vectors to compare with: run with PHOTO_MAP_BACKEND=cpu first");
                continue;
            };
            let cpu: serde_json::Value = serde_json::from_slice(&cpu).unwrap();
            let rows = |v: &serde_json::Value| -> Vec<Vec<f32>> { v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect() };
            for (what, ours, theirs) in [("photos", &photos, rows(&cpu["photos"])), ("words", &texts, rows(&cpu["words"]))] {
                let lowest = ours.iter().zip(&theirs).map(|(a, b)| cosine(a, b)).fold(1f32, f32::min);
                eprintln!("  {what}: lowest cosine to the processor's {lowest:.5}");
                assert!(lowest > 0.99, "{} {what}: {lowest}", spec.label);
            }
        }
    }
}
