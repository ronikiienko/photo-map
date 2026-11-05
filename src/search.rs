//! Search without Python: the text model (models.rs) runs on a thread of its own, and a map's photo vectors are compared with the
//! query's vector here. A map's vectors are look.bin in its folder (its pictures', float16, as many numbers each as the
//! map's model makes). A map is searched with the model it was made with.
//!
//! A video's sound is searched too, when the map has it (sound.bin, listen.rs): words only, through the sound model's text model,
//! which the search thread loads on the processor when a map with sound opens. Its scores mean nothing next to the pictures', so a
//! search is by look or by sound, never both.
//!
//! Only the pictures are searched, never words that come with them. Measured on Wikivoyage
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
    /// Words said or heard in the videos (the sound model's: `model` is ignored).
    Sound(String),
}

struct Request {
    id: u64,
    model: String,
    query: Query,
}

enum Message {
    Ask(Request),
    /// Load the sound model's text model now (a map with sound opened): the first search by sound is then quick.
    WantSound,
}

/// The sound model's text model in the search thread: not loaded, loaded, or why it couldn't be.
#[derive(Clone, Default)]
pub enum SoundState {
    #[default]
    Off,
    Loading,
    Ready,
    Failed(String),
}

pub struct Engine {
    to_model: Sender<Message>,
    sound: Arc<std::sync::Mutex<SoundState>>,
    from_model: Receiver<Event>,
    next: u64,
    /// The loaded model, shared with map building, which embeds photos with it (one copy on the card).
    pub hub: Arc<Hub>,
}

impl Engine {
    /// Start the model thread. Nothing is loaded until a model is asked for (`want`, `ask`).
    pub fn start(root: PathBuf, wake: impl Fn() + Send + Sync + 'static) -> Self {
        let wake = Arc::new(wake);
        let hub = Hub::start(root.clone(), {
            let wake = wake.clone();
            move || wake()
        });
        let (to_model, requests) = channel::<Message>();
        let (events, from_model) = channel();
        let shared = hub.clone();
        let sound = Arc::new(std::sync::Mutex::new(SoundState::Off));
        let sound_state = sound.clone();
        std::thread::spawn(move || {
            let mut listener: Option<crate::listen::Listener> = None;
            while let Ok(message) = requests.recv() {
                let Request { id, model, query } = match message {
                    Message::WantSound => {
                        let _ = listen(&mut listener, &sound_state, &root, &*wake);
                        continue;
                    }
                    Message::Ask(r) => r,
                };
                let (vector, note) = match &query {
                    Query::Sound(text) if text.trim().is_empty() => (Err("nothing to search for".into()), None),
                    Query::Sound(text) => (listen(&mut listener, &sound_state, &root, &*wake).and_then(|l| l.embed_query(text.trim())), None),
                    _ => match shared.wait_for(&model, &|| false) {
                        Ok(_) => match shared.with(&model, |m| query_vector(m, &model, &query)) {
                            Ok(Ok((v, note))) => (Ok(v), note),
                            Ok(Err(e)) | Err(e) => (Err(e), None),
                        },
                        Err(e) => (Err(e), None),
                    },
                };
                let _ = events.send(Event::Vector { id, vector, note });
                wake();
            }
        });
        Self { to_model, sound, from_model, next: 0, hub }
    }

    /// Start loading the model for a map (so that the first search is quick). Returns at once.
    pub fn want(&self, model: &str) {
        self.hub.want(model);
    }

    /// Start loading the sound model's text model (a map with sound opened). Returns at once.
    pub fn want_sound(&self) {
        let _ = self.to_model.send(Message::WantSound);
    }

    pub fn sound(&self) -> SoundState {
        self.sound.lock().unwrap().clone()
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
        let _ = self.to_model.send(Message::Ask(Request { id: self.next, model: model.to_string(), query }));
        self.next
    }

    /// Whatever the thread has said since the last call.
    pub fn poll(&mut self) -> Vec<Event> {
        self.from_model.try_iter().collect()
    }
}

/// The sound model's text model in the search thread, loaded the first time it is wanted (a failure is kept: it would fail again).
fn listen<'a>(slot: &'a mut Option<crate::listen::Listener>, state: &std::sync::Mutex<SoundState>, root: &Path, wake: &dyn Fn()) -> Result<&'a mut crate::listen::Listener, String> {
    if slot.is_none() {
        if let SoundState::Failed(why) = &*state.lock().unwrap() {
            return Err(why.clone());
        }
        *state.lock().unwrap() = SoundState::Loading;
        wake();
        let loaded = crate::embed::Runtime::shared(root).and_then(|rt| crate::listen::Listener::for_search(rt, root, models::sound()));
        *state.lock().unwrap() = match &loaded {
            Ok(_) => SoundState::Ready,
            Err(e) => SoundState::Failed(e.clone()),
        };
        wake();
        *slot = Some(loaded?);
    }
    Ok(slot.as_mut().unwrap())
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
        Query::Sound(_) => Err("a search by sound goes to the sound model".into()),
    }
}

/// Each `dim`-long row of `halves` times `query`.
fn cosines(halves: &[f16], dim: usize, query: &[f32]) -> Vec<f32> {
    use half::slice::HalfFloatSliceExt;
    const ROWS: usize = 512;
    let mut out = vec![0.0; halves.len() / dim];
    out.par_chunks_mut(ROWS).zip(halves.par_chunks(ROWS * dim)).for_each(|(out, block)| {
        let mut row_f32 = vec![0.0f32; dim];
        for (o, row) in out.iter_mut().zip(block.chunks_exact(dim)) {
            row.convert_to_f32_slice(&mut row_f32);
            *o = dot(&row_f32, query);
        }
    });
    out
}

/// Eight sums side by side, so the compiler makes it vector instructions (one running sum can't be reordered).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let (a8, b8) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = a8.remainder().iter().zip(b8.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in a8.zip(b8) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    lanes.iter().sum::<f32>() + tail
}

/// A video's score is its best moment's (a frame, or a window of its sound): `scores` (per photo) get their videos' best of
/// `moment_scores` (moment k is of video `videos[k]`; a video's moments come one after another). Returns each photo's best moment:
/// none for a photo, which keeps its own score.
pub fn best_moments(scores: &mut [f32], moment_scores: &[f32], videos: impl IntoIterator<Item = usize>) -> Vec<Option<usize>> {
    let mut best = vec![None; scores.len()];
    let mut last = None;
    for (k, (&c, video)) in moment_scores.iter().zip(videos).enumerate() {
        if last != Some(video) || c > scores[video] {
            scores[video] = c;
            best[video] = Some(k);
        }
        last = Some(video);
    }
    best
}

/// A map's vectors, memory-mapped.
pub struct Vectors {
    /// look.bin: the items' rows, then the videos' frames' (a video's own row is its poster frame's, which places it on the map;
    /// a search scores it by its best frame: best_moments).
    look: Mmap,
    /// How many items (the rows before the frames').
    n: usize,
    /// The windows of the videos' sound (sound.bin; None if none) and how many numbers each has (the sound model's).
    sounds: Option<(Mmap, usize)>,
    /// How many numbers each vector has.
    dim: usize,
}

impl Vectors {
    /// None if the map has no vectors (look.bin), or they are the wrong size for its `n` items and `frames` frames.
    /// `sounds`: the sound windows and their vectors' size (sound.bin is left out if it is the wrong size).
    pub fn open(bundle: &Path, n: usize, frames: usize, dim: usize, sounds: (usize, usize)) -> Option<Self> {
        let map = |name: &str, rows: usize, dim: usize| {
            let file = std::fs::File::open(bundle.join(name)).ok()?;
            // mapped in whole now, while the map opens: the first search would otherwise wait for the disk and the page faults (~50 ms)
            let map = unsafe { memmap2::MmapOptions::new().populate().map(&file) }.ok()?;
            (map.len() == rows * dim * 2).then_some(map)
        };
        let sounds = if sounds.0 > 0 { map("sound.bin", sounds.0, sounds.1).map(|m| (m, sounds.1)) } else { None };
        Some(Self { look: map("look.bin", n + frames, dim)?, n, sounds, dim })
    }

    /// Has the map its videos' sound?
    pub fn has_sound(&self) -> bool {
        self.sounds.is_some()
    }

    /// The query's cosine (a vector of the sound model) with every window of the videos' sound (empty if there are none).
    pub fn sound_cosines(&self, query: &[f32]) -> Vec<f32> {
        self.sounds.as_ref().filter(|s| s.1 == query.len()).map_or(Vec::new(), |(m, dim)| cosines(bytemuck::cast_slice(m), *dim, query))
    }

    /// The query's cosine with every photo's picture (negative ones too). The query is a unit vector, so is each photo's.
    /// Blocks of rows on each thread, each row turned into floats a whole slice at a time (with the CPU's own f16 instructions,
    /// which one number at a time can't use) and multiplied in eight lanes: 88k photos in ~10 ms instead of ~50. A video's is its
    /// average frame's (`frame_cosines` has each frame's).
    pub fn cosines(&self, query: &[f32]) -> Vec<f32> {
        cosines(&bytemuck::cast_slice::<u8, f16>(&self.look)[..self.n * self.dim], self.dim, query)
    }

    /// The query's cosine with every frame of the videos (empty if there are none).
    pub fn frame_cosines(&self, query: &[f32]) -> Vec<f32> {
        cosines(&bytemuck::cast_slice::<u8, f16>(&self.look)[self.n * self.dim..], self.dim, query)
    }

    /// One photo's vector.
    pub fn row(&self, i: usize) -> Vec<f32> {
        self.look[i * self.dim * 2..(i + 1) * self.dim * 2].chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_video_scores_as_its_best_moment() {
        let mut out = vec![0.5, 0.9, 0.5, 0.2];
        let best = best_moments(&mut out, &[0.1, 0.3, 0.2, 0.7, 0.6], [1, 1, 1, 3, 3]);
        assert_eq!(out, vec![0.5, 0.3, 0.5, 0.7], "the video's own row (its poster) doesn't count, its best moment does");
        assert_eq!(best, vec![None, Some(1), None, Some(3)], "a photo has no moments");
    }

    #[test]
    fn cosines_are_each_rows_dot_product() {
        let dim = 13; // not a multiple of 8: the tail counts too
        let rows: Vec<Vec<f32>> = (0..1100).map(|r| (0..dim).map(|c| ((r * 7 + c * 3) % 11) as f32 / 11.0 - 0.5).collect()).collect();
        let halves: Vec<f16> = rows.iter().flatten().map(|&x| f16::from_f32(x)).collect();
        let query: Vec<f32> = (0..dim).map(|c| c as f32 / dim as f32).collect();
        let got = cosines(&halves, dim, &query);
        assert_eq!(got.len(), rows.len());
        for (row, g) in rows.iter().zip(got) {
            let want: f32 = row.iter().zip(&query).map(|(x, q)| f16::from_f32(*x).to_f32() * q).sum();
            assert!((want - g).abs() < 1e-4, "{want} {g}");
        }
    }
}
