//! Search without Python: the text model (models.rs) runs on a thread of its own, and a map's photo vectors are compared with the
//! query's vector here. A map's vectors are two files in its bundle: embeddings.bin (the photos' images, float16, as many numbers
//! each as the map's model makes) and captions.bin (the captions', same layout; the photos without a caption have zeros). A map is
//! searched with the model it was made with.

use crate::models::{self, Hub, Model};
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

/// What the text model thread says.
pub enum Event {
    /// The vector of a query (`id`: the number `ask` returned): the words' and the photo's vectors together.
    /// `note`: something to tell the user about the search (a text cut short).
    Vector { id: u64, vector: Result<Vec<f32>, String>, note: Option<String> },
}

/// What is searched for: words, or a photo, never both. (Adding their two vectors would mean something only for a model whose words
/// are about as close to its photos as photos are to each other; for SigLIP they are not - a caption is 0.17 like its own photo, a
/// photo 0.78 like its nearest photo - and one model without it would make a feature that works on some maps and not on others.)
#[derive(Clone)]
pub enum Query {
    Words(String),
    Photo(PathBuf),
}

struct Request {
    id: u64,
    model: String,
    query: Query,
}

pub struct Engine {
    to_model: Sender<Request>,
    from_model: Receiver<Event>,
    next: u64,
    /// The loaded model, shared with map building, which embeds photos with it (one copy on the card).
    pub hub: Arc<Hub>,
}

impl Engine {
    /// Start the model thread. Nothing is loaded until a model is asked for (`want`, `ask`).
    pub fn start(root: PathBuf, wake: impl Fn() + Send + Sync + 'static) -> Self {
        let wake = Arc::new(wake);
        let hub = Hub::start(root, {
            let wake = wake.clone();
            move || wake()
        });
        let (to_model, requests) = channel::<Request>();
        let (events, from_model) = channel();
        let shared = hub.clone();
        std::thread::spawn(move || {
            while let Ok(Request { id, model, query }) = requests.recv() {
                let (vector, note) = match shared.wait_for(&model, &|| false) {
                    Ok(_) => match shared.with(&model, |m| query_vector(m, &model, &query)) {
                        Ok(Ok((v, note))) => (Ok(v), note),
                        Ok(Err(e)) | Err(e) => (Err(e), None),
                    },
                    Err(e) => (Err(e), None),
                };
                let _ = events.send(Event::Vector { id, vector, note });
                wake();
            }
        });
        Self { to_model, from_model, next: 0, hub }
    }

    /// Start loading the model for a map (so that the first search is quick). Returns at once.
    pub fn want(&self, model: &str) {
        self.hub.want(model);
    }

    /// Is this model loaded? Some(on the graphics card).
    pub fn ready(&self, model: &str) -> Option<bool> {
        let state = self.hub.state();
        (state.loaded.as_deref() == Some(model)).then_some(state.on_gpu)
    }

    /// Why the model couldn't be loaded, if it couldn't.
    pub fn failed(&self) -> Option<String> {
        self.hub.state().failed
    }

    /// Ask for the vector of some words or a photo, made by this model.
    pub fn ask(&mut self, model: &str, query: Query) -> u64 {
        self.next += 1;
        let _ = self.to_model.send(Request { id: self.next, model: model.to_string(), query });
        self.next
    }

    /// Whatever the thread has said since the last call.
    pub fn poll(&mut self) -> Vec<Event> {
        self.from_model.try_iter().collect()
    }
}

/// The vector of a search (unit length), and a note if the model could not read all of it.
fn query_vector(model: &mut Model, id: &str, query: &Query) -> Result<(Vec<f32>, Option<String>), String> {
    let spec = models::spec(id).ok_or("unknown model")?;
    match query {
        Query::Words(text) => {
            let text = text.trim();
            if text.is_empty() {
                return Err("nothing to search for".into());
            }
            let note = spec.max_tokens.filter(|_| model.query_cut(text)).map(|n| format!("{} reads only the first {n} tokens of a text (roughly {} words); the rest was ignored", spec.label, n * 2 / 3));
            Ok((model.embed_query(text)?, note))
        }
        Query::Photo(path) => {
            let img = crate::photo::load_for(path, |w, h| spec.longest_side_needed(w, h)).ok_or("can't read that photo")?;
            Ok((model.embed_image(&img)?, None))
        }
    }
}

/// A map's vectors, memory-mapped.
pub struct Vectors {
    images: Mmap,
    captions: Option<Mmap>,
    /// Which photos have a caption vector (a row of zeros is none).
    n: usize,
    /// How many numbers each vector has.
    dim: usize,
}

impl Vectors {
    /// None if the map has no vectors (embeddings.bin), or they are the wrong size for its photos.
    pub fn open(bundle: &Path, n: usize, dim: usize) -> Option<Self> {
        let map = |name: &str| {
            let file = std::fs::File::open(bundle.join(name)).ok()?;
            let map = unsafe { Mmap::map(&file) }.ok()?;
            (map.len() == n * dim * 2).then_some(map)
        };
        Some(Self { images: map("embeddings.bin")?, captions: map("captions.bin"), n, dim })
    }

    pub fn has_captions(&self) -> bool {
        self.captions.is_some()
    }

    /// The query's cosine with every photo's image, then with every photo's caption, as the viewer keeps them (u16: 0..1 as 0..65535;
    /// 0 for a photo with no caption). The query is a unit vector, so is each photo's.
    pub fn cosines(&self, query: &[f32]) -> Vec<u16> {
        let quantise = |c: f32| (c.clamp(0.0, 1.0) * 65535.0).round() as u16;
        let mut out: Vec<u16> = self.image_cosines(query).into_iter().map(quantise).collect();
        match &self.captions {
            Some(captions) => {
                let has = |i: usize| captions[i * self.dim * 2..(i + 1) * self.dim * 2].iter().any(|&b| b != 0);
                out.extend(self.dots(captions, query).into_iter().enumerate().map(|(i, c)| if has(i) { quantise(c) } else { 0 }));
            }
            None => out.extend(std::iter::repeat_n(0, self.n)),
        }
        out
    }

    /// The query's cosine with every photo's image, as it is (negative ones too).
    pub fn image_cosines(&self, query: &[f32]) -> Vec<f32> {
        self.dots(&self.images, query)
    }

    fn dots(&self, data: &Mmap, query: &[f32]) -> Vec<f32> {
        data.par_chunks_exact(self.dim * 2)
            .map(|row| row.chunks_exact(2).zip(query).map(|(b, q)| f16::from_le_bytes([b[0], b[1]]).to_f32() * q).sum::<f32>())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real map and Python's answer for the same words (/tmp/wv_ref.*, made by hand; the test is skipped without them): the
    /// cosines agree to within the noise of fp16 vs bf16, and the best photos are the same.
    #[test]
    fn wikivoyage_matches_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let bundle = root.join("maps/wikivoyage.photomap");
        let Ok(answer) = std::fs::read(std::path::Path::new("/tmp/wv_ref.json")) else { return };
        let query = serde_json::from_slice::<serde_json::Value>(&answer).unwrap()["query"].as_str().unwrap().to_string();
        let n = 88133;
        let Some(vectors) = Vectors::open(&bundle, n, 768) else { return };
        let python: Vec<u16> = std::fs::read(std::path::Path::new("/tmp/wv_ref.u16")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let shared = crate::models::test_hub();
        let mut guard = shared.slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let model = guard.as_mut().unwrap();
        let started = std::time::Instant::now();
        let vector = model.embed_query(&query).unwrap();
        let embedded = started.elapsed();
        let started = std::time::Instant::now();
        let mine = vectors.cosines(&vector);
        eprintln!("{query:?}: query {embedded:?}, {n} photos compared in {:?}", started.elapsed());
        assert_eq!(mine.len(), python.len());
        let worst = mine.iter().zip(&python).map(|(&a, &b)| (a as i32 - b as i32).abs()).max().unwrap();
        eprintln!("largest difference {:.4} in cosine", worst as f32 / 65535.0);
        let top = |v: &[u16]| {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by_key(|&i| std::cmp::Reverse(v[i].max(v[n + i])));
            idx.truncate(30);
            idx
        };
        let (a, b) = (top(&mine), top(&python));
        let shared = a.iter().filter(|i| b.contains(i)).count();
        eprintln!("{shared} of the best 30 photos are the same");
        assert!(worst < 1500, "cosines differ by up to {}", worst as f32 / 65535.0);
        assert!(shared >= 24);
    }
}
