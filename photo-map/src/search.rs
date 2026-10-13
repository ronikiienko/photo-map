//! Search without Python: the text model (embed.rs) runs on a thread of its own, and a map's photo vectors are compared with the
//! query's vector here. A map's vectors are two files in its bundle: embeddings.bin (the photos' images, float16, 768 numbers each) and
//! captions.bin (the captions', same layout; the photos without a caption have zeros). Gemma is the only model.

use crate::embed::Gemma;
use half::f16;
use memmap2::Mmap;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender, channel};

const DIM: usize = 768;

/// What the text model thread says.
pub enum Event {
    /// The model is loaded (on the GPU or not).
    Ready { gpu: bool },
    /// It couldn't be loaded.
    Failed(String),
    /// The vector of a query (`id`: the number `ask` returned): the words' and the photo's vectors together.
    Vector { id: u64, vector: Result<Vec<f32>, String> },
}

pub struct Engine {
    to_model: Sender<(u64, String, Option<PathBuf>)>,
    from_model: Receiver<Event>,
    next: u64,
    pub ready: Option<bool>,
    pub failed: Option<String>,
    /// The model, once loaded: shared with map building, which embeds photos with it (one copy on the card).
    pub model: Arc<Mutex<Option<Gemma>>>,
}

impl Engine {
    /// Start loading the model on a thread (a few seconds); `ask` can be called at once, it waits its turn.
    pub fn start(root: PathBuf, wake: impl Fn() + Send + 'static) -> Self {
        let (to_model, requests) = channel::<(u64, String, Option<PathBuf>)>();
        let (events, from_model) = channel();
        let model = Arc::new(Mutex::new(None));
        let shared = model.clone();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let loaded = match Gemma::load(&root) {
                Ok(m) => m,
                Err(e) => {
                    let _ = events.send(Event::Failed(e));
                    wake();
                    return;
                }
            };
            eprintln!("search model ready ({}) after {:.1}s", if loaded.on_gpu { "GPU" } else { "CPU" }, started.elapsed().as_secs_f32());
            let _ = events.send(Event::Ready { gpu: loaded.on_gpu });
            *shared.lock().unwrap() = Some(loaded);
            wake();
            while let Ok((id, text, photo)) = requests.recv() {
                let vector = match shared.lock().unwrap().as_mut() {
                    Some(m) => query_vector(m, &text, photo.as_deref()),
                    None => Err("the model is not loaded".into()),
                };
                let _ = events.send(Event::Vector { id, vector });
                wake();
            }
        });
        Self { to_model, from_model, next: 0, ready: None, failed: None, model }
    }

    /// Ask for the vector of some words and/or a photo (their vectors added: the one between them).
    pub fn ask(&mut self, text: &str, photo: Option<&Path>) -> u64 {
        self.next += 1;
        let _ = self.to_model.send((self.next, text.to_string(), photo.map(Path::to_path_buf)));
        self.next
    }

    /// Whatever the thread has said since the last call.
    pub fn poll(&mut self) -> Vec<Event> {
        let events: Vec<Event> = self.from_model.try_iter().collect();
        for e in &events {
            match e {
                Event::Ready { gpu } => self.ready = Some(*gpu),
                Event::Failed(why) => self.failed = Some(why.clone()),
                Event::Vector { .. } => {}
            }
        }
        events
    }
}

fn query_vector(model: &mut Gemma, text: &str, photo: Option<&Path>) -> Result<Vec<f32>, String> {
    let mut sum = vec![0f32; DIM];
    let mut parts = 0;
    if !text.trim().is_empty() {
        for (s, v) in sum.iter_mut().zip(model.embed_query(text.trim())?) {
            *s += v;
        }
        parts += 1;
    }
    if let Some(path) = photo {
        let img = crate::photo::load_for(path, crate::embed::longest_side_needed).ok_or("can't read that photo")?;
        for (s, v) in sum.iter_mut().zip(model.embed_image(&img)?) {
            *s += v;
        }
        parts += 1;
    }
    if parts == 0 {
        return Err("nothing to search for".into());
    }
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    Ok(sum.into_iter().map(|x| x / norm).collect())
}

/// A map's vectors, memory-mapped.
pub struct Vectors {
    images: Mmap,
    captions: Option<Mmap>,
    /// Which photos have a caption vector (a row of zeros is none).
    n: usize,
}

impl Vectors {
    /// None if the map has no vectors (embeddings.bin), or they are the wrong size for its photos.
    pub fn open(bundle: &Path, n: usize) -> Option<Self> {
        let map = |name: &str| {
            let file = std::fs::File::open(bundle.join(name)).ok()?;
            let map = unsafe { Mmap::map(&file) }.ok()?;
            (map.len() == n * DIM * 2).then_some(map)
        };
        Some(Self { images: map("embeddings.bin")?, captions: map("captions.bin"), n })
    }

    pub fn has_captions(&self) -> bool {
        self.captions.is_some()
    }

    /// The query's cosine with every photo's image, then with every photo's caption, as the viewer keeps them (u16: 0..1 as 0..65535;
    /// 0 for a photo with no caption). The query is a unit vector, so is each photo's.
    pub fn cosines(&self, query: &[f32]) -> Vec<u16> {
        let dots = |data: &Mmap| -> Vec<f32> {
            data.par_chunks_exact(DIM * 2)
                .map(|row| {
                    row.chunks_exact(2).zip(query).map(|(b, q)| f16::from_le_bytes([b[0], b[1]]).to_f32() * q).sum::<f32>()
                })
                .collect()
        };
        let quantise = |c: f32| (c.clamp(0.0, 1.0) * 65535.0).round() as u16;
        let mut out: Vec<u16> = dots(&self.images).into_iter().map(quantise).collect();
        match &self.captions {
            Some(captions) => {
                let has = |i: usize| captions[i * DIM * 2..(i + 1) * DIM * 2].iter().any(|&b| b != 0);
                out.extend(dots(captions).into_iter().enumerate().map(|(i, c)| if has(i) { quantise(c) } else { 0 }));
            }
            None => out.extend(std::iter::repeat_n(0, self.n)),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real map and Python's answer for the same words (/tmp/wv_ref.*, made by hand; the test is skipped without them): the
    /// cosines agree to within the noise of fp16 vs bf16, and the best photos are the same.
    #[test]
    fn wikivoyage_matches_python() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let bundle = root.join("maps/wikivoyage.photomap");
        let Ok(answer) = std::fs::read(std::path::Path::new("/tmp/wv_ref.json")) else { return };
        let query = serde_json::from_slice::<serde_json::Value>(&answer).unwrap()["query"].as_str().unwrap().to_string();
        let n = 88133;
        let Some(vectors) = Vectors::open(&bundle, n) else { return };
        let python: Vec<u16> = std::fs::read(std::path::Path::new("/tmp/wv_ref.u16")).unwrap().chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let shared = crate::embed::shared_model();
        let mut guard = shared.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
