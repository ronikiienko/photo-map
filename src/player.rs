//! Playing a video in its tile (no sound): ffmpeg decodes it from a time at its own speed (`-re`) to raw RGBA frames of the
//! size it is shown at, a thread keeps the newest one, and the window uploads it when it is new. Pausing stops ffmpeg; playing
//! again, or going to another time, starts it there.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The longest side frames are decoded at (px).
pub const MAX_SIDE: u32 = 1280;

/// What the reader thread hands over: the newest frame, and whether the video has run out.
#[derive(Default)]
struct Shared {
    frame: Mutex<Option<Vec<u8>>>,
    ended: AtomicBool,
}

pub struct Player {
    /// The video (a photo of the map), and where it plays: on the map, or in a match's copy beside it (its position).
    pub photo: usize,
    pub copy: Option<usize>,
    pub duration: f32,
    ffmpeg: PathBuf,
    video: PathBuf,
    /// The frames' size (px).
    pub size: (u32, u32),
    /// Where it was when it last started or stopped, and since when it plays (None: paused).
    at: f32,
    since: Option<Instant>,
    child: Option<Child>,
    shared: Arc<Shared>,
    wake: Arc<dyn Fn() + Send + Sync>,
    texture: Option<egui::TextureHandle>,
}

impl Player {
    /// Plays `video` from `at` seconds, in frames of about `long` px on its longer side (its shape: `width` x `height`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(ffmpeg: &Path, video: &Path, photo: usize, copy: Option<usize>, duration: f32, at: f32, (width, height): (u32, u32), long: f32, wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        let side = (long.round() as u32).clamp(64, MAX_SIDE);
        let scale = side as f32 / width.max(height).max(1) as f32;
        let even = |v: f32| ((v / 2.0).round() as u32).max(1) * 2;
        let size = (even(width as f32 * scale), even(height as f32 * scale));
        let mut player = Self { photo, copy, duration, ffmpeg: ffmpeg.to_path_buf(), video: video.to_path_buf(), size, at: at.clamp(0.0, duration), since: None, child: None, shared: Arc::new(Shared::default()), wake, texture: None };
        player.play();
        player
    }

    /// Where it is (seconds).
    pub fn position(&self) -> f32 {
        let t = self.at + self.since.map_or(0.0, |s| s.elapsed().as_secs_f32());
        if self.shared.ended.load(Ordering::Relaxed) { self.duration } else { t.min(self.duration) }
    }

    pub fn playing(&self) -> bool {
        self.since.is_some() && !self.shared.ended.load(Ordering::Relaxed)
    }

    /// Start ffmpeg where it is (from the start again once it has run out).
    pub fn play(&mut self) {
        self.stop_ffmpeg();
        if self.position() >= self.duration - 0.05 {
            self.at = 0.0;
        }
        self.shared = Arc::new(Shared::default());
        let (w, h) = self.size;
        let child = Command::new(&self.ffmpeg)
            .args(["-nostdin", "-loglevel", "error", "-re", "-ss"])
            .arg(format!("{:.3}", self.at))
            .arg("-i")
            .arg(&self.video)
            .args(["-an", "-vf"])
            .arg(format!("scale={w}:{h}"))
            .args(["-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = child else {
            self.shared.ended.store(true, Ordering::Relaxed);
            return;
        };
        let mut out = child.stdout.take().expect("piped");
        let (shared, wake) = (self.shared.clone(), self.wake.clone());
        let bytes = (w * h * 4) as usize;
        std::thread::spawn(move || {
            loop {
                let mut frame = vec![0u8; bytes];
                if out.read_exact(&mut frame).is_err() {
                    break;
                }
                *shared.frame.lock().unwrap() = Some(frame);
                wake();
            }
            shared.ended.store(true, Ordering::Relaxed);
            wake();
        });
        self.child = Some(child);
        self.since = Some(Instant::now());
    }

    pub fn pause(&mut self) {
        self.at = self.position();
        self.since = None;
        self.stop_ffmpeg();
        // the stopped ffmpeg's reader sees its output end: that is not the video running out (play would start it over)
        self.shared = Arc::new(Shared::default());
    }

    /// Go to `t` seconds, playing on if it was playing.
    pub fn seek(&mut self, t: f32) {
        let playing = self.playing();
        self.pause();
        self.shared.ended.store(false, Ordering::Relaxed);
        self.at = t.clamp(0.0, self.duration);
        if playing {
            self.play();
        }
    }

    fn stop_ffmpeg(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if self.since.is_some() && self.shared.ended.load(Ordering::Relaxed) {
            self.at = self.duration;
        }
    }

    /// The newest frame as a texture (uploaded when a new one came); None until the first one has.
    pub fn texture(&mut self, ctx: &egui::Context) -> Option<egui::TextureId> {
        if let Some(frame) = self.shared.frame.lock().unwrap().take() {
            let image = egui::ColorImage::from_rgba_unmultiplied([self.size.0 as usize, self.size.1 as usize], &frame);
            match &mut self.texture {
                Some(t) => t.set(image, egui::TextureOptions::LINEAR),
                None => self.texture = Some(ctx.load_texture("video", image, egui::TextureOptions::LINEAR)),
            }
        }
        self.texture.as_ref().map(|t| t.id())
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop_ffmpeg();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames come at the size asked for, from the time asked for, and stop when paused. Needs ffmpeg; no window.
    #[test]
    fn plays_and_pauses() {
        let Some(ffmpeg) = crate::runtime::ffmpeg(&crate::data_root()) else { return };
        let temp = std::env::temp_dir().join(format!("photo-map-player-test-{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();
        let video = temp.join("v.mp4");
        assert!(Command::new(&ffmpeg).args(["-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x180:rate=25"]).arg(&video).status().unwrap().success());
        let mut p = Player::new(&ffmpeg, &video, 0, None, 3.0, 1.0, (320, 180), 160.0, Arc::new(|| {}));
        assert_eq!(p.size, (160, 90));
        let started = Instant::now();
        while p.shared.frame.lock().unwrap().is_none() && started.elapsed().as_secs() < 5 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(p.shared.frame.lock().unwrap().as_ref().map(Vec::len), Some(160 * 90 * 4));
        assert!(p.playing() && p.position() >= 1.0);
        p.pause();
        let at = p.position();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(p.position(), at, "paused: it stays");
        assert!(p.child.is_none());
        std::thread::sleep(std::time::Duration::from_millis(200)); // the stopped ffmpeg's output has ended by now
        p.play();
        assert!(p.position() >= at && p.position() < at + 0.1, "played again: on from where it was, {at} -> {}", p.position());
        p.pause();
        p.seek(2.5);
        assert_eq!(p.position(), 2.5);
        std::fs::remove_dir_all(&temp).unwrap();
    }
}
