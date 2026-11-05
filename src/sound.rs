//! A playing video's sound (player.rs): ffmpeg decodes it to samples at the speakers' rate, they wait in a queue, and the
//! speakers (cpal) take them as they play. That is also the clock the frames go by, as players do: a frame shows when the sound
//! has reached its time. A video without sound, or a computer without speakers, plays silence the same way, so the clock is
//! always the sound's.

use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

/// The speakers, open while a video plays: they play whichever `Sound` is current (silence when none is).
pub struct Speakers {
    pub rate: u32,
    pub channels: u16,
    current: Arc<Mutex<Option<Arc<Sound>>>>,
    pub muted: Arc<AtomicBool>,
    _stream: Option<cpal::Stream>,
    /// Without speakers: a thread taking the samples in their time instead; it stops when this goes.
    alive: Arc<AtomicBool>,
}

impl Speakers {
    pub fn open(muted: bool) -> Self {
        let current: Arc<Mutex<Option<Arc<Sound>>>> = Arc::default();
        let muted = Arc::new(AtomicBool::new(muted));
        let alive = Arc::new(AtomicBool::new(true));
        if let Some((stream, rate, channels)) = open_stream(current.clone(), muted.clone()) {
            return Self { rate, channels, current, muted, _stream: Some(stream), alive };
        }
        // no speakers: the samples are taken 10 ms at a time, so the frames still go by the clock
        let (rate, channels) = (48_000, 2);
        let (taken, on) = (current.clone(), alive.clone());
        std::thread::spawn(move || {
            let mut out = vec![0.0f32; rate as usize / 100 * channels as usize];
            let mut next = Instant::now();
            while on.load(Ordering::Relaxed) {
                if let Some(sound) = taken.lock().unwrap().clone() {
                    sound.fill(&mut out, 0.0, true);
                }
                next += Duration::from_millis(10);
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
        });
        Self { rate, channels, current, muted, _stream: None, alive }
    }

    /// Play `sound` (none: silence).
    pub fn play(&self, sound: Option<Arc<Sound>>) {
        *self.current.lock().unwrap() = sound;
    }
}

impl Drop for Speakers {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
    }
}

/// The default speakers, in whatever kind of sample they take.
fn open_stream(current: Arc<Mutex<Option<Arc<Sound>>>>, muted: Arc<AtomicBool>) -> Option<(cpal::Stream, u32, u16)> {
    let device = cpal::default_host().default_output_device()?;
    let config = device.default_output_config().ok()?;
    let (rate, channels) = (config.sample_rate(), config.channels());
    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => build::<f32>(&device, config.into(), current, muted),
        cpal::SampleFormat::I16 => build::<i16>(&device, config.into(), current, muted),
        cpal::SampleFormat::I32 => build::<i32>(&device, config.into(), current, muted),
        cpal::SampleFormat::U16 => build::<u16>(&device, config.into(), current, muted),
        _ => None,
    }?;
    stream.play().ok()?;
    Some((stream, rate, channels))
}

fn build<T: SizedSample + FromSample<f32>>(device: &cpal::Device, config: cpal::StreamConfig, current: Arc<Mutex<Option<Arc<Sound>>>>, muted: Arc<AtomicBool>) -> Option<cpal::Stream> {
    let mut samples = Vec::new();
    let write = move |out: &mut [T], info: &cpal::OutputCallbackInfo| {
        let t = info.timestamp();
        let latency = t.playback.duration_since(t.callback).as_secs_f32();
        samples.clear();
        samples.resize(out.len(), 0.0f32);
        if let Some(sound) = current.lock().unwrap().clone() {
            sound.fill(&mut samples, latency, muted.load(Ordering::Relaxed));
        }
        for (o, s) in out.iter_mut().zip(&samples) {
            *o = T::from_sample(*s);
        }
    };
    device.build_output_stream(config, write, |_| {}, None).ok()
}

/// One video's sound from one time (going to another time is a new one, as the frames' ffmpeg is).
pub struct Sound {
    rate: u32,
    channels: usize,
    /// Samples (the speakers' channels, interleaved) from ffmpeg, not yet played.
    queue: Mutex<VecDeque<f32>>,
    /// ffmpeg has sent all it has (maybe nothing: no sound); past it plays silence, the clock going on.
    done: AtomicBool,
    /// Playing (from the video's first frame shown, until paused), else silent with the clock standing.
    pub run: AtomicBool,
    clock: Mutex<Clock>,
    stopped: AtomicBool,
}

/// What the speakers have taken: by the last time they asked, and before it; when that was, and how long their samples take
/// to be heard.
#[derive(Default)]
struct Clock {
    taken: u64,
    before: u64,
    at: Option<Instant>,
    latency: f32,
}

/// Up to half a second of samples waits in the queue (ffmpeg is held back beyond it).
const AHEAD: f32 = 0.5;

impl Sound {
    /// `video`'s sound from `at` seconds, at the speakers' rate and channels: ffmpeg is started (the caller stops it).
    pub fn start(ffmpeg: &Path, video: &Path, at: f32, speakers: &Speakers) -> (Arc<Self>, Option<Child>) {
        let sound = Arc::new(Self {
            rate: speakers.rate,
            channels: speakers.channels as usize,
            queue: Mutex::default(),
            done: AtomicBool::new(false),
            run: AtomicBool::new(false),
            clock: Mutex::default(),
            stopped: AtomicBool::new(false),
        });
        let child = Command::new(ffmpeg)
            .args(["-nostdin", "-hide_banner", "-nostats", "-loglevel", "error", "-ss"])
            .arg(format!("{at:.3}"))
            .arg("-i")
            .arg(video)
            // its first sound track, if it has one (else ffmpeg stops at once: silence)
            .args(["-vn", "-map", "0:a:0?", "-ac"])
            .arg(sound.channels.to_string())
            .arg("-ar")
            .arg(sound.rate.to_string())
            .args(["-f", "f32le", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = child else {
            sound.done.store(true, Ordering::Relaxed);
            return (sound, None);
        };
        let mut out = child.stdout.take().expect("piped");
        let reader = sound.clone();
        std::thread::spawn(move || {
            let most = (AHEAD * reader.rate as f32) as usize * reader.channels;
            let mut bytes = vec![0u8; 16384];
            let mut odd = Vec::new(); // a sample's bytes split between two reads
            while let Ok(n) = out.read(&mut bytes) {
                if n == 0 || reader.stopped.load(Ordering::Relaxed) {
                    break;
                }
                odd.extend_from_slice(&bytes[..n]);
                let whole = odd.len() / 4 * 4;
                reader.queue.lock().unwrap().extend(odd[..whole].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                odd.drain(..whole);
                while reader.queue.lock().unwrap().len() > most && !reader.stopped.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
            reader.done.store(true, Ordering::Relaxed);
        });
        (sound, Some(child))
    }

    /// No more of it (ffmpeg was stopped).
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
        self.run.store(false, Ordering::Relaxed);
    }

    /// The speakers' turn: `out` filled (silent when muted, or held), heard in `latency` seconds. Waiting for ffmpeg, it gives
    /// silence and the clock stands, so the frames wait too.
    fn fill(&self, out: &mut [f32], latency: f32, muted: bool) {
        let frames = (out.len() / self.channels) as u64;
        let mut queue = self.queue.lock().unwrap();
        let ready = queue.len() >= out.len() || self.done.load(Ordering::Relaxed);
        let take = self.run.load(Ordering::Relaxed) && ready;
        for o in out.iter_mut() {
            let s = if take { queue.pop_front().unwrap_or(0.0) } else { 0.0 };
            *o = if muted { 0.0 } else { s };
        }
        drop(queue);
        let mut clock = self.clock.lock().unwrap();
        clock.before = clock.taken;
        if take {
            clock.taken += frames;
        }
        clock.at = Some(Instant::now());
        clock.latency = latency;
    }

    /// Where the sound heard is (seconds from its start): between the speakers' turns it goes on with the time, up to what
    /// they took.
    pub fn clock(&self) -> f32 {
        let c = self.clock.lock().unwrap();
        let Some(at) = c.at else { return 0.0 };
        let moved = (at.elapsed().as_secs_f64() * self.rate as f64) as u64;
        let heard = (c.before + moved).min(c.taken);
        heard as f32 / self.rate as f32 - c.latency
    }
}
