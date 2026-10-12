//! Playing a photo's sound (meta "audio": a file in the bundle): one clip at a time, streamed from disk. The output
//! device and the decoder exist only while a clip is loaded; stop() frees both.

use rodio::decoder::DecoderBuilder;
use rodio::{DeviceSinkBuilder, MixerDeviceSink, Player, Source};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Duration;

struct Clip {
    photo: usize,
    player: Player,
    length: Duration,
    /// Keeps the device open while the clip plays.
    _device: MixerDeviceSink,
}

#[derive(Default)]
pub struct Audio {
    clip: Option<Clip>,
}

/// Where the loaded clip is, in seconds.
pub struct Position {
    pub now: f32,
    pub length: f32,
    pub paused: bool,
}

/// The file's decoder. Told its size and that it can seek: otherwise going back in the clip doesn't work.
fn decoder(file: &Path) -> Result<rodio::Decoder<BufReader<File>>, String> {
    let err = |e: &dyn std::fmt::Display| format!("{}: {e}", file.display());
    let f = File::open(file).map_err(|e| err(&e))?;
    let len = f.metadata().map_err(|e| err(&e))?.len();
    DecoderBuilder::new().with_data(BufReader::new(f)).with_byte_len(len).with_seekable(true).build().map_err(|e| err(&e))
}

impl Audio {
    /// The photo whose clip is loaded (playing or paused). A clip that has played to its end counts as gone.
    pub fn loaded(&mut self) -> Option<usize> {
        if self.clip.as_ref().is_some_and(|c| c.player.empty()) {
            self.clip = None;
        }
        self.clip.as_ref().map(|c| c.photo)
    }

    /// Start `photo`'s clip from the beginning, replacing any other; if it is the loaded one: pause / resume.
    pub fn toggle(&mut self, photo: usize, file: &Path) -> Result<(), String> {
        if self.loaded() == Some(photo) {
            let c = self.clip.as_ref().unwrap();
            if c.player.is_paused() { c.player.play() } else { c.player.pause() }
            return Ok(());
        }
        self.stop();
        let mut device = DeviceSinkBuilder::open_default_sink().map_err(|e| format!("no sound output: {e}"))?;
        device.log_on_drop(false);
        let source = decoder(file)?;
        let length = source.total_duration().unwrap_or_default();
        let player = Player::connect_new(device.mixer());
        player.append(source);
        self.clip = Some(Clip { photo, player, length, _device: device });
        Ok(())
    }

    pub fn position(&self) -> Option<Position> {
        let c = self.clip.as_ref()?;
        Some(Position { now: c.player.get_pos().as_secs_f32(), length: c.length.as_secs_f32(), paused: c.player.is_paused() })
    }

    pub fn seek(&mut self, seconds: f32) {
        if let Some(c) = &self.clip {
            let _ = c.player.try_seek(Duration::from_secs_f32(seconds.max(0.0)));
        }
    }

    /// Drop the clip: its decoder and the output device.
    pub fn stop(&mut self) {
        self.clip = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_has_length_and_seeks_both_ways() {
        let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("../maps/esc50.photomap/audio/000000.wav");
        let mut source = decoder(&file).unwrap();
        let length = source.total_duration().unwrap();
        assert!((length.as_secs_f32() - 5.0).abs() < 0.1, "{length:?}");
        let rate = source.sample_rate().get() as f32 * source.channels().get() as f32;
        source.try_seek(Duration::from_secs(4)).unwrap();
        source.try_seek(Duration::from_secs(1)).unwrap(); // back
        let left = source.count() as f32 / rate;
        assert!((left - 4.0).abs() < 0.1, "{left} s left after seeking back to 1 s");
    }
}
