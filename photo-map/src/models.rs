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
    /// Its files, under .cache/onnx/: the folder, and the files that must be in it.
    pub folder: &'static str,
    pub files: &'static [&'static str],
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
        folder: "embeddinggemma-2",
        files: &["onnx/model_fp16.onnx", "tokenizer.json"],
    },
    Spec {
        id: SIGLIP_BASE_512,
        label: "SigLIP 2 Base (512 px)",
        note: "About 2 to 3 times faster than Gemma, a little less accurate. Short words and photos only.",
        dim: 768,
        kind: Kind::Siglip { size: 512 },
        max_tokens: Some(64),
        query_hint: "Search the map: a few words on what you look for, or drop in a photo",
        folder: "siglip2-base-patch16-512",
        files: &["vision_fp16.onnx", "text_fp16.onnx", "tokenizer.json"],
    },
    Spec {
        id: SIGLIP_BASE_256,
        label: "SigLIP 2 Base (256 px)",
        note: "The fastest: photos are looked at in a quarter of the detail, so it misses more. Short words and photos only.",
        dim: 768,
        kind: Kind::Siglip { size: 256 },
        max_tokens: Some(64),
        query_hint: "Search the map: a few words on what you look for, or drop in a photo",
        folder: "siglip2-base-patch16-256",
        files: &["vision_fp16.onnx", "text_fp16.onnx", "tokenizer.json"],
    },
];

pub fn spec(id: &str) -> Option<&'static Spec> {
    ALL.iter().find(|s| s.id == id)
}

impl Spec {
    /// Are its files there? (The photo encoder is checked when it is first needed.)
    pub fn installed(&self, root: &Path) -> bool {
        let dir = root.join(".cache/onnx").join(self.folder);
        self.files.iter().all(|f| dir.join(f).exists())
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
            if started.elapsed().as_secs() > 300 {
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
    static HUB: std::sync::OnceLock<Arc<Hub>> = std::sync::OnceLock::new();
    let hub = HUB
        .get_or_init(|| {
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
            Hub::start(root, || {})
        })
        .clone();
    assert_eq!(hub.wait_for(GEMMA, &|| false), Ok(true));
    hub
}
