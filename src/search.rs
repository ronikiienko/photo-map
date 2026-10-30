//! Search without Python: the text model (models.rs) runs on a thread of its own, and a map's photo vectors are compared with the
//! query's vector here. A map's vectors are embeddings.bin in its bundle (the photos' pictures, float16, as many numbers each as the
//! map's model makes). A map is searched with the model it was made with.
//!
//! Only the pictures are searched, never the captions (captions.bin, which older maps have, is not read). Measured on Wikivoyage
//! (4,139 photos with their real captions): with SigLIP, words are much closer to any caption than to any picture (a typical
//! caption 0.70 like "church", the best picture 0.15), so a caption decided every match, and SigLIP judges text against text
//! poorly ("seal": Carp, shore, Woodpecker).

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
        Some(Self { images: map("embeddings.bin")?, dim })
    }

    /// The query's cosine with every photo's picture (negative ones too). The query is a unit vector, so is each photo's.
    pub fn cosines(&self, query: &[f32]) -> Vec<f32> {
        self.images
            .par_chunks_exact(self.dim * 2)
            .map(|row| row.chunks_exact(2).zip(query).map(|(b, q)| f16::from_le_bytes([b[0], b[1]]).to_f32() * q).sum::<f32>())
            .collect()
    }

    /// One photo's vector.
    pub fn row(&self, i: usize) -> Vec<f32> {
        self.images[i * self.dim * 2..(i + 1) * self.dim * 2].chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()
    }
}
