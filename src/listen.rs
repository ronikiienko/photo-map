//! Videos searched by their sound: EmbeddingGemma 2's audio encoder and text model (models::EMBEDDINGGEMMA_2), through ONNX Runtime.
//! A video's sound is cut into 10 s windows, each turned into a vector in the same space as search words (docs/audio.md: 10 s
//! windows found best, with no overlap as good as any).
//!
//! A window goes in as sound features (`features`: 128 log-mel bands, 100 frames a second, as the publisher's feature extractor
//! makes them, Gemma4AudioFeatureExtractor); the audio encoder turns them into 250 tokens, which take the place of the 250 audio
//! marks in the text model's input (`window_ids`); the text model gives the pooled vector. Search words go through the same text
//! model alone, after the publisher's search prompt.
//!
//! A build runs it on TensorRT when it can (the audio encoder in float16, the text model in bfloat16: patch.rs), else on CUDA,
//! else on the processor, all from the publisher's float32 files. A search's words run on the processor only, as SigLIP's do.

use crate::embed::{Result, Runtime};
use crate::models::Spec;
use ort::ep;
use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// The sound is read at this rate, one channel.
pub const RATE: usize = 16_000;
/// A window's length, seconds and samples.
pub const WINDOW: f32 = 10.0;
pub const WINDOW_SAMPLES: usize = 160_000;
/// A window's feature frames (10 ms apart) and the mel bands of each.
pub const FRAMES: usize = 999;
pub const MELS: usize = 128;
/// The audio encoder's tokens for a window (25 a second).
const TOKENS: usize = 250;
/// The audio encoder's token width (its projection into the text model).
const TOKEN_WIDTH: usize = 512;
/// Windows handed to the card at once (TensorRT's engines are built for 1 to this many).
pub const BATCH: usize = 8;
/// What the publisher's sentence-transformers config puts before a search ("SearchQuery").
const SEARCH_PROMPT: &str = "task: search result | query: ";

// the feature extractor's settings (the publisher's preprocessor_config.json)
const FRAME_LENGTH: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const MEL_FLOOR: f64 = 0.001;
const MAX_FREQUENCY: f64 = 8000.0;

/// The analysis window (periodic Hann) and the mel filters (HTK scale, no norm), made once. Each filter is a triangle over a few
/// bins: kept as its first bin and its weights there (the zeros elsewhere would only add nothing).
struct Bank {
    window: Vec<f32>,
    filters: Vec<(usize, Vec<f64>)>,
}

fn bank() -> &'static Bank {
    static BANK: std::sync::OnceLock<Bank> = std::sync::OnceLock::new();
    BANK.get_or_init(|| {
        let window = (0..FRAME_LENGTH).map(|n| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME_LENGTH as f64).cos()) as f32).collect();
        let bins = FFT / 2 + 1;
        let mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let hertz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);
        let (lo, hi) = (mel(0.0), mel(MAX_FREQUENCY));
        let centres: Vec<f64> = (0..MELS + 2).map(|i| hertz(lo + (hi - lo) * i as f64 / (MELS + 1) as f64)).collect();
        let filters = (0..MELS)
            .map(|m| {
                let weight = |b: usize| {
                    let f = (RATE / 2) as f64 * b as f64 / (bins - 1) as f64;
                    let down = (f - centres[m]) / (centres[m + 1] - centres[m]);
                    let up = (centres[m + 2] - f) / (centres[m + 2] - centres[m + 1]);
                    down.min(up).max(0.0)
                };
                let used: Vec<usize> = (0..bins).filter(|&b| weight(b) > 0.0).collect();
                match (used.first(), used.last()) {
                    (Some(&first), Some(&last)) => (first, (first..=last).map(weight).collect()),
                    _ => (0, Vec::new()),
                }
            })
            .collect();
        Bank { window, filters }
    })
}

/// A window's sound features: FRAMES x MELS, log-mel. `samples`: WINDOW_SAMPLES of sound at RATE (a shorter one is padded with
/// silence). As the publisher's extractor: frames of 320 samples every 160, the first centred at 0 (160 zeros before), a Hann
/// window, a 512-point FFT's magnitudes through the mel filters, the log of that plus 0.001.
pub fn features(samples: &[f32]) -> Vec<f32> {
    use realfft::RealFftPlanner;
    let bank = bank();
    let mut padded = vec![0f32; HOP + WINDOW_SAMPLES];
    let n = samples.len().min(WINDOW_SAMPLES);
    padded[HOP..HOP + n].copy_from_slice(&samples[..n]);
    let fft = RealFftPlanner::<f64>::new().plan_fft_forward(FFT);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut magnitude = vec![0f64; FFT / 2 + 1];
    let mut out = vec![0f32; FRAMES * MELS];
    for (i, row) in out.chunks_exact_mut(MELS).enumerate() {
        let frame = &padded[i * HOP..i * HOP + FRAME_LENGTH];
        input.fill(0.0);
        for (x, (s, w)) in input.iter_mut().zip(frame.iter().zip(&bank.window)) {
            *x = (s * w) as f64; // multiplied in float32, as numpy does with two float32 arrays
        }
        fft.process(&mut input, &mut spectrum).expect("the plan's own buffers");
        for (m, c) in magnitude.iter_mut().zip(&spectrum) {
            *m = c.norm();
        }
        for (v, (first, weights)) in row.iter_mut().zip(&bank.filters) {
            let sum: f64 = magnitude[*first..].iter().zip(weights).map(|(m, w)| m * w).sum();
            *v = (sum + MEL_FLOOR).ln() as f32;
        }
    }
    out
}

/// Where a video's windows start (seconds), the rule `read_windows` follows: every WINDOW from the start, and one ending at the end
/// if more than a second is left after the last whole one (so every window is a whole one; it overlaps the one before). A video
/// shorter than a window has one, from the start (its features padded with silence); one under half a second, none.
#[cfg(test)]
pub fn window_starts(duration: f32) -> Vec<f32> {
    if duration < 0.5 {
        return Vec::new();
    }
    if duration <= WINDOW {
        return vec![0.0];
    }
    let whole = (duration / WINDOW).floor() as usize;
    let mut starts: Vec<f32> = (0..whole).map(|k| k as f32 * WINDOW).collect();
    if duration - whole as f32 * WINDOW > 1.0 {
        starts.push(duration - WINDOW);
    }
    starts
}

fn err(what: &str) -> impl Fn(ort::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// What the sound runs on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Card {
    TensorRt,
    Cuda,
    Processor,
}

pub struct Listener {
    // the environment must outlive the sessions
    _env: ort::environment::Environment,
    /// The audio encoder: None for a search (words only).
    audio: Option<Session>,
    text: Session,
    tokenizer: Tokenizer,
    /// A window's token ids: <bos> <|audio> 250 x <|audio|> <audio|> <eos>.
    window_ids: Vec<i64>,
    pub card: Card,
}

impl Listener {
    /// For a build: the audio encoder and the text model, on the card if this computer has one (TensorRT builds its engines the
    /// first time: a few minutes, kept in .cache/trt/).
    pub fn for_build(rt: &Runtime, root: &Path, spec: &'static Spec) -> Result<Self> {
        let dir = installed(root, spec)?;
        let cache = root.join(".cache/trt").join(spec.folder);
        if rt.tensorrt {
            let opened = (|| {
                let audio = crate::patch::ensure(&dir, crate::patch::SOUND_AUDIO_TENSORRT)?;
                let text = crate::patch::ensure(&dir, crate::patch::SOUND_TEXT_TENSORRT)?;
                let window = window_ids(&tokenizer(&dir)?)?.len();
                let shapes = |n: usize| format!("input_features:{n}x{FRAMES}x{MELS},input_features_mask:{n}x{FRAMES}");
                let audio = tensorrt(rt, &cache, &audio, true, shapes(1), shapes(BATCH))?;
                let shapes = |n: usize| format!("input_ids:{n}x{window},attention_mask:{n}x{window},audio_features:{}x{TOKEN_WIDTH}", n * TOKENS);
                let text = tensorrt(rt, &cache, &text, false, shapes(1), shapes(BATCH))?;
                Self::new(rt, &dir, Some(audio), text, Card::TensorRt)
            })();
            match opened {
                Ok(l) => return Ok(l),
                Err(e) => eprintln!("TensorRT is off, the sound model runs on CUDA: {e}"),
            }
        }
        let card = if rt.on_gpu { Card::Cuda } else { Card::Processor };
        let builder = || {
            let b = Session::builder(&rt.env);
            if card == Card::Cuda { b.and_then(|b| b.with_execution_providers([crate::embed::cuda()]).map_err(|e| e.into())) } else { b }
        };
        let audio = builder().and_then(|mut b| b.commit_from_file(dir.join(crate::patch::SOUND_AUDIO))).map_err(err("loading the audio encoder"))?;
        let text = builder().and_then(|mut b| b.commit_from_file(dir.join(crate::patch::SOUND_TEXT))).map_err(err("loading the sound model's text model"))?;
        Self::new(rt, &dir, Some(audio), text, card)
    }

    /// For a search: the text model alone, on the processor (a search never starts CUDA: the card's memory stays the map's).
    pub fn for_search(rt: &Runtime, root: &Path, spec: &'static Spec) -> Result<Self> {
        let dir = installed(root, spec)?;
        let text = Session::builder(&rt.env).and_then(|mut b| b.commit_from_file(dir.join(crate::patch::SOUND_TEXT))).map_err(err("loading the sound model's text model"))?;
        Self::new(rt, &dir, None, text, Card::Processor)
    }

    fn new(rt: &Runtime, dir: &Path, audio: Option<Session>, text: Session, card: Card) -> Result<Self> {
        let tokenizer = tokenizer(dir)?;
        let window_ids = window_ids(&tokenizer)?;
        Ok(Self { _env: rt.env.clone(), audio, text, tokenizer, window_ids, card })
    }

    /// Windows' features (`features`, at most BATCH): their vectors, unit length.
    pub fn embed_windows(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n = windows.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let audio = self.audio.as_mut().ok_or("the sound model was opened for searching only")?;
        let flat: Vec<f32> = windows.iter().flatten().copied().collect();
        if flat.len() != n * FRAMES * MELS {
            return Err(format!("{} sound features for {n} windows", flat.len()));
        }
        let features = Tensor::from_array(([n, FRAMES, MELS], flat)).map_err(err("tensor"))?;
        let mask = Tensor::from_array(([n, FRAMES], vec![true; n * FRAMES])).map_err(err("tensor"))?;
        let out = audio.run(ort::inputs!["input_features" => features, "input_features_mask" => mask]).map_err(err("running the audio encoder"))?;
        let (_, tokens) = out["audio_features"].try_extract_tensor::<f32>().map_err(err("reading the audio tokens"))?;
        if tokens.len() != n * TOKENS * TOKEN_WIDTH {
            return Err(format!("the audio encoder made {} numbers for {n} windows", tokens.len()));
        }
        let tokens = tokens.to_vec();
        drop(out);
        let len = self.window_ids.len();
        let ids: Vec<i64> = (0..n).flat_map(|_| self.window_ids.iter().copied()).collect();
        self.run_text(n, len, ids, tokens, n * TOKENS)
    }

    /// A search's words: their vector, unit length (after the publisher's search prompt).
    pub fn embed_query(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(format!("{SEARCH_PROMPT}{text}"), true).map_err(|e| e.to_string())?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        let len = ids.len();
        Ok(self.run_text(1, len, ids, Vec::new(), 0)?.remove(0))
    }

    /// The text model on `n` rows of `len` token ids, with `audio` tokens (rows of TOKEN_WIDTH) in place of the audio marks.
    fn run_text(&mut self, n: usize, len: usize, ids: Vec<i64>, audio: Vec<f32>, audio_rows: usize) -> Result<Vec<Vec<f32>>> {
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array(([n, len], ids)).map_err(err("tensor"))?,
            "attention_mask" => Tensor::from_array(([n, len], vec![1i64; n * len])).map_err(err("tensor"))?,
            "audio_features" => Tensor::from_array(([audio_rows, TOKEN_WIDTH], audio)).map_err(err("tensor"))?,
        ];
        // the publisher's file takes photos and videos too (none here); the TensorRT patch has them taken out
        if self.text.inputs().iter().any(|i| i.name() == "image_features") {
            for name in ["image_features", "video_features"] {
                inputs.push((name.into(), Tensor::from_array(([0usize, TOKEN_WIDTH], Vec::<f32>::new())).map_err(err("tensor"))?.into()));
            }
        }
        let out = self.text.run(inputs).map_err(err("running the sound model's text model"))?;
        let (_, data) = out["sentence_embedding"].try_extract_tensor::<f32>().map_err(err("reading the result"))?;
        if data.len() % n != 0 {
            return Err(format!("the text model made {} numbers for {n}", data.len()));
        }
        Ok(data.chunks_exact(data.len() / n).map(|r| unit(r.to_vec())).collect())
    }
}

fn installed(root: &Path, spec: &'static Spec) -> Result<PathBuf> {
    let dir = spec.dir(root);
    if !spec.installed(root) {
        return Err(format!("{} is not downloaded (its files go in {})", spec.label, dir.display()));
    }
    Ok(dir)
}

fn tokenizer(dir: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("loading the sound model's tokenizer: {e}"))
}

/// A sound window's token ids, as the publisher's processor makes them: <bos> <|audio> 250 x <|audio|> <audio|> <eos>.
fn window_ids(tokenizer: &Tokenizer) -> Result<Vec<i64>> {
    let id = |t: &str| tokenizer.token_to_id(t).map(|i| i as i64).ok_or_else(|| format!("the tokenizer has no {t}"));
    let mut ids = vec![id("<bos>")?, id("<|audio>")?];
    ids.extend(std::iter::repeat_n(id("<|audio|>")?, TOKENS));
    ids.extend([id("<audio|>")?, id("<eos>")?]);
    Ok(ids)
}

/// A session on TensorRT (float16 or bfloat16) with CUDA under it, its engine kept in `cache`.
fn tensorrt(rt: &Runtime, cache: &Path, file: &Path, fp16: bool, min: String, max: String) -> Result<Session> {
    std::fs::create_dir_all(cache).map_err(|e| e.to_string())?;
    let trt = ep::TensorRT::default()
        .with_fp16(fp16)
        .with_bf16(!fp16)
        .with_max_workspace_size(1 << 30) // the card is shared with the map: keep the engine's scratch space small
        .with_engine_cache(true)
        .with_engine_cache_path(cache.display())
        .with_timing_cache(true)
        .with_timing_cache_path(cache.display())
        .with_profile_min_shapes(min)
        .with_profile_opt_shapes(max.clone())
        .with_profile_max_shapes(max)
        .build();
    Session::builder(&rt.env)
        .and_then(|b| b.with_execution_providers([trt, crate::embed::cuda()]).map_err(|e| e.into()))
        .and_then(|mut b| b.commit_from_file(file))
        .map_err(err(&format!("loading {} on TensorRT", file.file_name().unwrap_or_default().to_string_lossy())))
}

/// A video's sound at RATE, one channel, float32, read by ffmpeg as it plays out: `each` gets every window (its start in seconds
/// and its samples) in turn, by the rule of `window_starts` applied to the sound really read (a header's length can be a little
/// off), and says whether to go on. Ok(false): stopped by `each`. A video without sound has no windows.
pub fn read_windows(ffmpeg: &Path, video: &Path, each: &mut dyn FnMut(f32, Vec<f32>) -> bool) -> Result<bool> {
    use std::io::Read;
    let mut child = std::process::Command::new(ffmpeg)
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(video)
        .args(["-map", "0:a:0", "-vn", "-ac", "1", "-ar", &RATE.to_string(), "-f", "f32le", "pipe:1"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("{}: {e}", ffmpeg.display()))?;
    let mut out = child.stdout.take().expect("piped");
    let stop = |mut child: std::process::Child| {
        let _ = child.kill();
        let _ = child.wait();
    };
    // the sound read so far, from sample `read_from` on: never less than the last WINDOW of it (a window ending at the end)
    let (mut kept, mut read_from) = (Vec::<f32>::with_capacity(2 * WINDOW_SAMPLES), 0usize);
    let mut whole = 0usize; // the whole windows handed on
    let (mut buf, mut carry) = (vec![0u8; 1 << 16], Vec::<u8>::new());
    loop {
        let got = match out.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                stop(child);
                return Err(e.to_string());
            }
        };
        carry.extend_from_slice(&buf[..got]);
        let bytes = carry.len() / 4 * 4;
        kept.extend(carry[..bytes].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
        carry.drain(..bytes);
        while (whole + 1) * WINDOW_SAMPLES <= read_from + kept.len() {
            let at = whole * WINDOW_SAMPLES - read_from;
            if !each(whole as f32 * WINDOW, kept[at..at + WINDOW_SAMPLES].to_vec()) {
                stop(child);
                return Ok(false);
            }
            whole += 1;
        }
        // what the next whole window has so far is less than a window: keeping the last WINDOW keeps it
        if kept.len() > 2 * WINDOW_SAMPLES {
            let drop = kept.len() - WINDOW_SAMPLES;
            kept.drain(..drop);
            read_from += drop;
        }
    }
    stop(child);
    let total = read_from + kept.len();
    let rest = total - whole * WINDOW_SAMPLES;
    if whole == 0 && total > RATE / 2 {
        // shorter than a window: all of it (padded with silence by `features`)
        return Ok(each(0.0, kept));
    }
    if whole > 0 && rest > RATE {
        let at = kept.len() - WINDOW_SAMPLES;
        return Ok(each((total - WINDOW_SAMPLES) as f32 / RATE as f32, kept[at..].to_vec()));
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_whole_and_reach_the_end() {
        assert_eq!(window_starts(0.2), Vec::<f32>::new());
        assert_eq!(window_starts(4.0), vec![0.0]);
        assert_eq!(window_starts(10.0), vec![0.0]);
        assert_eq!(window_starts(30.5), vec![0.0, 10.0, 20.0], "half a second left over is not worth a window");
        assert_eq!(window_starts(35.0), vec![0.0, 10.0, 20.0, 25.0], "the last one ends at the end");
    }

    #[test]
    fn mel_filters_are_triangles_up_to_8_khz() {
        let b = bank();
        let bins = FFT / 2 + 1;
        assert_eq!(b.filters.len(), MELS);
        assert!(b.filters.iter().all(|(first, w)| w.iter().all(|f| (0.0..=1.0).contains(f)) && first + w.len() <= bins));
        // the lowest and the top filters are narrower than a bin and can fall between two (the publisher's note): most pick some up
        let picking = b.filters.iter().filter(|(_, w)| !w.is_empty()).count();
        assert!(picking > MELS - 10, "{picking}");
        assert!((b.window[160] - 1.0).abs() < 1e-6 && b.window[0] == 0.0);
    }

    fn reference(name: &str) -> Vec<f32> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/sound");
        std::fs::read(dir.join(name)).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (a.iter().map(|x| x * x).sum::<f32>().sqrt() * b.iter().map(|x| x * x).sum::<f32>().sqrt())
    }

    /// Search words on the processor give the publisher's vectors (tools/sound_reference.py first; the model downloaded):
    /// cargo test --release query_vectors -- --ignored --nocapture
    #[test]
    #[ignore]
    fn query_vectors_match_the_publishers() {
        let root = crate::data_root();
        let mut l = Listener::for_search(crate::embed::Runtime::shared(&root).unwrap(), &root, crate::models::sound()).unwrap();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/sound");
        let queries: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(dir.join("queries.json")).unwrap()).unwrap();
        for q in queries {
            let want: Vec<f32> = q["vector"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let started = std::time::Instant::now();
            let got = l.embed_query(q["text"].as_str().unwrap()).unwrap();
            let c = cosine(&got, &want);
            eprintln!("{:?}: cosine {c:.6} ({:.0} ms)", q["text"], started.elapsed().as_secs_f32() * 1000.0);
            assert!(c > 0.9999, "{c}");
        }
    }

    /// A recording's first windows, read and embedded as a build does on the backend in use, against the publisher's vectors
    /// (float32 on the processor): cargo test --release window_vectors -- --ignored --nocapture
    /// (PHOTO_MAP_BACKEND=cpu|cuda|tensorrt picks the backend).
    #[test]
    #[ignore]
    fn window_vectors_match_the_publishers() {
        let root = crate::data_root();
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/sound");
        let about: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("windows.json")).unwrap()).unwrap();
        let count = about["windows"].as_u64().unwrap() as usize;
        let want = reference("windows.f32");
        let ffmpeg = crate::runtime::ffmpeg(&root).expect("ffmpeg");
        let mut windows = Vec::new();
        read_windows(&ffmpeg, Path::new(about["source"].as_str().unwrap()), &mut |start, samples| {
            windows.push((start, samples));
            windows.len() < count
        })
        .unwrap();
        assert_eq!(windows.iter().map(|w| w.0).collect::<Vec<_>>(), (0..count).map(|k| k as f32 * WINDOW).collect::<Vec<_>>());
        let started = std::time::Instant::now();
        let features: Vec<Vec<f32>> = windows.iter().map(|w| features(&w.1)).collect();
        let feature_ms = started.elapsed().as_secs_f32() * 1000.0 / count as f32;
        let started = std::time::Instant::now();
        let mut l = Listener::for_build(crate::embed::Runtime::shared(&root).unwrap(), &root, crate::models::sound()).unwrap();
        l.embed_windows(&features[..BATCH.min(count)]).unwrap(); // TensorRT's engines load (or are built) on the first
        eprintln!("{:?}: ready in {:.1} s", l.card, started.elapsed().as_secs_f32());
        let started = std::time::Instant::now();
        let got: Vec<Vec<f32>> = features.chunks(BATCH).flat_map(|c| l.embed_windows(c).unwrap()).collect();
        let took = started.elapsed().as_secs_f32();
        let cos: Vec<f32> = got.iter().zip(want.chunks_exact(768)).map(|(a, b)| cosine(a, b)).collect();
        let (mean, worst) = (cos.iter().sum::<f32>() / cos.len() as f32, cos.iter().copied().fold(1.0, f32::min));
        eprintln!(
            "{:?}: {count} windows, features {feature_ms:.1} ms a window (one thread), model {:.1} ms a window ({:.0}x real time); cosine to the publisher's: mean {mean:.5}, worst {worst:.5}",
            l.card,
            took * 1000.0 / count as f32,
            count as f32 * WINDOW / took
        );
        assert!(worst > 0.99, "{worst}");
    }

    /// Where a folder's listening time goes: reading, features, the card by video's batches and by full ones, each batch size.
    /// PHOTO_MAP_VIDEOS=list.txt (one video a line) cargo test --release listening_speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn listening_speed() {
        use rayon::prelude::*;
        let root = crate::data_root();
        let ffmpeg = crate::runtime::ffmpeg(&root).expect("ffmpeg");
        let list = std::fs::read_to_string(std::env::var("PHOTO_MAP_VIDEOS").unwrap()).unwrap();
        let t = std::time::Instant::now();
        let mut videos: Vec<Vec<Vec<f32>>> = Vec::new();
        for v in list.lines() {
            let mut w = Vec::new();
            read_windows(&ffmpeg, Path::new(v), &mut |_, s| {
                w.push(s);
                true
            })
            .unwrap();
            videos.push(w);
        }
        let read = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        pool.install(|| {
            list.lines().collect::<Vec<_>>().par_iter().with_max_len(1).for_each(|v| {
                read_windows(&ffmpeg, Path::new(v), &mut |_, _| true).unwrap();
            })
        });
        eprintln!("reading 4 at once: {:.2} s", t.elapsed().as_secs_f32());
        let count: usize = videos.iter().map(|v| v.len()).sum();
        let sound = count as f32 * WINDOW;
        let t = std::time::Instant::now();
        let feats: Vec<Vec<Vec<f32>>> = videos.par_iter().map(|v| v.par_iter().map(|s| features(s)).collect()).collect();
        let feat_all = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        for v in &videos {
            for c in v.chunks(BATCH) {
                let _: Vec<_> = c.par_iter().map(|s| features(s)).collect();
            }
        }
        let feat_by8 = t.elapsed().as_secs_f32();
        eprintln!("{} videos, {count} windows; reading one by one {read:.2} s; features all at once {feat_all:.2} s, 8 at a time {feat_by8:.2} s", videos.len());
        let t = std::time::Instant::now();
        let mut l = Listener::for_build(crate::embed::Runtime::shared(&root).unwrap(), &root, crate::models::sound()).unwrap();
        let one: Vec<Vec<f32>> = feats.iter().flatten().take(BATCH).cloned().collect();
        l.embed_windows(&one).unwrap();
        eprintln!("{:?}: model ready in {:.1} s", l.card, t.elapsed().as_secs_f32());
        for n in [1, 2, 4, 8] {
            l.embed_windows(&one[..n]).unwrap();
            let t = std::time::Instant::now();
            for _ in 0..5 {
                l.embed_windows(&one[..n]).unwrap();
            }
            eprintln!("  batch of {n}: {:.1} ms", t.elapsed().as_secs_f32() * 1000.0 / 5.0);
        }
        let t = std::time::Instant::now();
        for v in &feats {
            for c in v.chunks(BATCH) {
                l.embed_windows(c).unwrap();
            }
        }
        let by_video = t.elapsed().as_secs_f32();
        let all: Vec<Vec<f32>> = feats.into_iter().flatten().collect();
        let t = std::time::Instant::now();
        for c in all.chunks(BATCH) {
            l.embed_windows(c).unwrap();
        }
        let full = t.elapsed().as_secs_f32();
        eprintln!("card: batches by video {by_video:.2} s ({:.0}x), full batches {full:.2} s ({:.0}x)", sound / by_video, sound / full);
        eprintln!("as the step runs (reading + 8-at-a-time features, then the card, overlapping): about {:.0}x at best", sound / (read + feat_by8).max(by_video));
    }

    /// The features match the publisher's extractor (Python, saved to .cache/test/sound/: tools/sound_reference.py).
    #[test]
    #[ignore]
    fn features_match_the_publishers() {
        let (wave, want) = (reference("wave.f32"), reference("features.f32"));
        let got = features(&wave);
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("largest difference {worst:e}");
        assert!(worst < 1e-3, "{worst}");
    }
}
