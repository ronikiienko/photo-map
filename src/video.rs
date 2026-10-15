//! Frames of a video: one every so many seconds, as pictures.
//!
//! MP4 and MOV files (the container is read by `re_mp4`) holding H.264 (`openh264`) or H.265 (`rusty_h265`, pure Rust). A video is
//! cut into its groups of pictures (a key frame and what follows it, which can be decoded without anything before it); a group with
//! no wanted frame in it is not decoded at all, so a long video sampled every few seconds costs a fraction of playing it.
//! Rotation (a phone held upright) is applied. Other containers and codecs are refused by name.

use image::RgbaImage;
use re_mp4::{Mp4, Track, TrackKind};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

type Result<T> = std::result::Result<T, String>;

/// A picture of the video and where it is in it.
pub struct VideoFrame {
    pub seconds: f64,
    pub image: RgbaImage,
}

enum Codec {
    H264,
    H265,
}

/// What `frames` found out about a video before decoding any of it.
pub struct VideoInfo {
    pub seconds: f64,
    pub width: u32,
    pub height: u32,
}

/// One sample (a coded picture) of the video track: where it is in the file and when it shows.
struct Coded {
    offset: u64,
    size: u64,
    seconds: f64,
    sync: bool,
}

struct Opened {
    path: std::path::PathBuf,
    codec: Codec,
    /// The parameter sets, with start codes, to feed ahead of every group.
    headers: Vec<u8>,
    /// How many bytes long each NAL unit's length is in the samples.
    length_size: usize,
    samples: Vec<Coded>,
    /// Quarter turns clockwise to show it upright.
    turns: u8,
    seconds: f64,
    width: u32,
    height: u32,
}

const START: [u8; 4] = [0, 0, 0, 1];

fn open(path: &Path) -> Result<Opened> {
    let mut file = File::open(path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mp4 = Mp4::read(&mut file, size).map_err(|e| format!("reading {} as MP4: {e}", path.display()))?;
    let track: &Track = mp4.tracks().values().find(|t| t.kind == Some(TrackKind::Video)).ok_or("no video in the file")?;
    let codec_name = track.codec_string(&mp4).unwrap_or_default();
    let codec = match codec_name.split('.').next().unwrap_or("") {
        "avc1" | "avc3" => Codec::H264,
        "hvc1" | "hev1" => Codec::H265,
        other => return Err(format!("this video's codec ({}) can't be read yet", if other.is_empty() { "unknown" } else { other })),
    };
    let config = track.raw_codec_config(&mp4).ok_or("no codec configuration")?;
    let (headers, length_size) = match codec {
        Codec::H264 => avc_headers(&config)?,
        Codec::H265 => hevc_headers(&config)?,
    };
    let timescale = track.timescale.max(1) as f64;
    let samples: Vec<Coded> = track.samples.iter().map(|s| Coded { offset: s.offset, size: s.size, seconds: s.composition_timestamp as f64 / timescale, sync: s.is_sync }).collect();
    if samples.is_empty() {
        return Err("the video has no pictures".into());
    }
    let m = &track.trak(&mp4).tkhd.matrix;
    let turns = match (m.a, m.b, m.c, m.d) {
        (0, b, c, 0) if b > 0 && c < 0 => 1,
        (a, 0, 0, d) if a < 0 && d < 0 => 2,
        (0, b, c, 0) if b < 0 && c > 0 => 3,
        _ => 0,
    };
    let (width, height) = (track.width as u32, track.height as u32);
    let (width, height) = if turns % 2 == 1 { (height, width) } else { (width, height) };
    let end = track.samples.last().map_or(0, |s| s.composition_timestamp as u64 + s.duration);
    Ok(Opened { path: path.to_path_buf(), codec, headers, length_size, samples, turns, seconds: end.max(track.duration) as f64 / timescale, width, height })
}

/// Length and size of a video without decoding it.
pub fn info(path: &Path) -> Result<VideoInfo> {
    let v = open(path)?;
    Ok(VideoInfo { seconds: v.seconds, width: v.width, height: v.height })
}

/// The SPS and PPS of an `avcC` record, with start codes, and the size of the length prefix.
fn avc_headers(c: &[u8]) -> Result<(Vec<u8>, usize)> {
    let bad = || "the H.264 configuration is cut short".to_string();
    if c.len() < 7 {
        return Err(bad());
    }
    let length_size = (c[4] & 3) as usize + 1;
    let mut out = Vec::new();
    let mut at = 5;
    let sets = (c[at] & 0x1f) as usize;
    at += 1;
    for round in 0..2 {
        let count = if round == 0 { sets } else { *c.get(at).ok_or_else(bad)? as usize };
        if round == 1 {
            at += 1;
        }
        for _ in 0..count {
            let len = u16::from_be_bytes(c.get(at..at + 2).ok_or_else(bad)?.try_into().unwrap()) as usize;
            out.extend(START);
            out.extend(c.get(at + 2..at + 2 + len).ok_or_else(bad)?);
            at += 2 + len;
        }
    }
    Ok((out, length_size))
}

/// The parameter sets (video, sequence, picture) of an `hvcC` record, with start codes, and the size of the length prefix.
fn hevc_headers(c: &[u8]) -> Result<(Vec<u8>, usize)> {
    let bad = || "the H.265 configuration is cut short".to_string();
    if c.len() < 23 {
        return Err(bad());
    }
    let length_size = (c[21] & 3) as usize + 1;
    let mut out = Vec::new();
    let mut at = 23;
    for _ in 0..c[22] {
        let count = u16::from_be_bytes(c.get(at + 1..at + 3).ok_or_else(bad)?.try_into().unwrap()) as usize;
        at += 3;
        for _ in 0..count {
            let len = u16::from_be_bytes(c.get(at..at + 2).ok_or_else(bad)?.try_into().unwrap()) as usize;
            out.extend(START);
            out.extend(c.get(at + 2..at + 2 + len).ok_or_else(bad)?);
            at += 2 + len;
        }
    }
    Ok((out, length_size))
}

/// A sample as the decoders read it: the length-prefixed NAL units with start codes in their place.
fn annex_b(sample: &[u8], length_size: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(sample.len() + 16);
    let mut at = 0;
    while at < sample.len() {
        let prefix = sample.get(at..at + length_size).ok_or("a picture is cut short")?;
        let len = prefix.iter().fold(0usize, |n, b| n << 8 | *b as usize);
        out.extend(START);
        out.extend(sample.get(at + length_size..at + length_size + len).ok_or("a picture is cut short")?);
        at += length_size + len;
    }
    Ok(out)
}

fn turn(image: RgbaImage, quarter_turns: u8) -> RgbaImage {
    match quarter_turns {
        1 => image::imageops::rotate90(&image),
        2 => image::imageops::rotate180(&image),
        3 => image::imageops::rotate270(&image),
        _ => image,
    }
}

/// Which of the group's pictures (counted in the order they show) are wanted: for each time, the one showing then.
fn wanted(times: &[f64], targets: impl Iterator<Item = f64>) -> Vec<usize> {
    let mut out: Vec<usize> = targets.map(|t| times.partition_point(|s| *s <= t).saturating_sub(1)).collect();
    out.dedup();
    out
}

/// One group of pictures with something wanted in it.
struct Job {
    from: usize,
    to: usize,
    /// The pictures wanted, counted in the order they show, and when each shows.
    keep: Vec<(usize, f64)>,
}

/// The video's pictures, one every `every` seconds (the first at 0), in order, each handed to `each`, which says whether to go on.
/// The groups are decoded side by side, a few at a time.
pub fn frames(path: &Path, every: f64, mut each: impl FnMut(VideoFrame) -> bool) -> Result<()> {
    use rayon::prelude::*;
    if every.is_nan() || every <= 0.0 {
        return Err("the interval between frames must be above zero".into());
    }
    let video = open(path)?;
    // the groups: from a key frame to the next one
    let mut starts: Vec<usize> = (0..video.samples.len()).filter(|i| video.samples[*i].sync || *i == 0).collect();
    starts.push(video.samples.len());
    let mut next = 0u64; // the next wanted time is next * every
    let mut jobs = Vec::new();
    for g in 0..starts.len() - 1 {
        let (from, to) = (starts[g], starts[g + 1]);
        let mut times: Vec<f64> = video.samples[from..to].iter().map(|s| s.seconds).collect();
        times.sort_by(f64::total_cmp);
        // this group shows from its first picture to the next group's first
        let until = if g + 2 < starts.len() { video.samples[starts[g + 1]..starts[g + 2]].iter().map(|s| s.seconds).fold(f64::INFINITY, f64::min) } else { video.seconds };
        let mut targets = Vec::new();
        while (next as f64 * every) < until {
            targets.push(next as f64 * every);
            next += 1;
        }
        if !targets.is_empty() {
            let keep = wanted(&times, targets.into_iter()).into_iter().map(|k| (k, times[k])).collect();
            jobs.push(Job { from, to, keep });
        }
    }
    // a few groups at a time: their pictures wait in memory until they are handed over
    for chunk in jobs.chunks(rayon::current_num_threads().max(1) * 2) {
        let done: Vec<Result<Vec<VideoFrame>>> = chunk.par_iter().map(|job| decode_group(&video, job)).collect();
        for frames in done {
            for frame in frames? {
                if !each(frame) {
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// The wanted pictures of one group.
fn decode_group(video: &Opened, job: &Job) -> Result<Vec<VideoFrame>> {
    let mut file = File::open(&video.path).map_err(|e| format!("opening {}: {e}", video.path.display()))?;
    let mut packets = Vec::with_capacity(job.to - job.from);
    for s in &video.samples[job.from..job.to] {
        let mut bytes = vec![0u8; s.size as usize];
        file.seek(SeekFrom::Start(s.offset)).and_then(|_| file.read_exact(&mut bytes)).map_err(|e| format!("reading a picture: {e}"))?;
        packets.push(annex_b(&bytes, video.length_size)?);
    }
    let last = job.keep.last().map_or(0, |k| k.0);
    let mut out = Vec::with_capacity(job.keep.len());
    let mut count = 0usize;
    let mut give = |image: RgbaImage, n: usize| {
        if let Some((_, seconds)) = job.keep.iter().find(|k| k.0 == n) {
            out.push(VideoFrame { seconds: *seconds, image: turn(image, video.turns) });
        }
    };
    match video.codec {
        Codec::H264 => decode_h264(video, &packets, last, &mut count, &mut give)?,
        Codec::H265 => decode_h265(video, &packets, last, &mut count, &mut give)?,
    }
    Ok(out)
}

/// Decode one group with OpenH264, giving each picture (in the order they show) until `last`.
fn decode_h264(video: &Opened, packets: &[Vec<u8>], last: usize, count: &mut usize, give: &mut dyn FnMut(RgbaImage, usize)) -> Result<()> {
    use openh264::decoder::{Decoder, DecoderConfig, Flush};
    use openh264::formats::YUVSource;
    let mut decoder = Decoder::with_api_config(openh264::OpenH264API::from_source(), DecoderConfig::new().flush_after_decode(Flush::NoFlush)).map_err(|e| format!("starting the H.264 decoder: {e}"))?;
    let handle = |yuv: &openh264::decoder::DecodedYUV, count: &mut usize, give: &mut dyn FnMut(RgbaImage, usize)| {
        let n = *count;
        *count += 1;
        // only the wanted pictures are made into colours
        let (w, h) = yuv.dimensions();
        let mut rgba = vec![0u8; w * h * 4];
        yuv.write_rgba8(&mut rgba);
        if let Some(image) = RgbaImage::from_raw(w as u32, h as u32, rgba) {
            give(image, n);
        }
    };
    decoder.decode(&video.headers).map_err(|e| format!("the H.264 headers: {e}"))?;
    for p in packets {
        if let Some(yuv) = decoder.decode(p).map_err(|e| format!("decoding H.264: {e}"))? {
            handle(&yuv, count, give);
            if *count > last {
                return Ok(());
            }
        }
    }
    for yuv in decoder.flush_remaining().map_err(|e| format!("decoding H.264: {e}"))? {
        handle(&yuv, count, give);
        if *count > last {
            break;
        }
    }
    Ok(())
}

/// Decode one group with the pure-Rust H.265 decoder.
fn decode_h265(video: &Opened, packets: &[Vec<u8>], last: usize, count: &mut usize, give: &mut dyn FnMut(RgbaImage, usize)) -> Result<()> {
    let mut decoder = rusty_h265::Decoder::new();
    let drain = |decoder: &mut rusty_h265::Decoder, count: &mut usize, give: &mut dyn FnMut(RgbaImage, usize)| -> bool {
        while let Ok(frame) = decoder.next_frame() {
            let n = *count;
            *count += 1;
            give(h265_rgba(&frame.picture), n);
            if *count > last {
                return true;
            }
        }
        false
    };
    decoder.push_annexb(&video.headers, None).map_err(|e| format!("the H.265 headers: {e}"))?;
    for p in packets {
        decoder.push_annexb(p, None).map_err(|e| format!("decoding H.265: {e}"))?;
        if drain(&mut decoder, count, give) {
            return Ok(());
        }
    }
    decoder.flush();
    drain(&mut decoder, count, give);
    Ok(())
}

/// A decoded H.265 picture (4:2:0, 8 or 10 bits, limited range) as RGBA, with the HD colour matrix.
fn h265_rgba(picture: &rusty_h265::frame::Picture) -> RgbaImage {
    let (left, top, w, h) = picture.crop;
    let [y_plane, cb_plane, cr_plane] = &picture.planes;
    let shift = picture.bit_depth_luma.saturating_sub(8);
    let chroma_shift = picture.bit_depth_chroma.saturating_sub(8);
    let mut out = RgbaImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (left + x, top + y);
            let luma = (y_plane.get(px, py) >> shift) as f32;
            let (cx, cy) = (px / 2, py / 2);
            let cb = (cb_plane.get(cx, cy) >> chroma_shift) as f32 - 128.0;
            let cr = (cr_plane.get(cx, cy) >> chroma_shift) as f32 - 128.0;
            let l = 1.164 * (luma - 16.0);
            let r = l + 1.793 * cr;
            let g = l - 0.213 * cb - 0.533 * cr;
            let b = l + 2.112 * cb;
            out.put_pixel(x as u32, y as u32, image::Rgba([r.clamp(0.0, 255.0) as u8, g.clamp(0.0, 255.0) as u8, b.clamp(0.0, 255.0) as u8, 255]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wanted_time_is_the_picture_showing_then() {
        let times = [0.0, 0.04, 0.08, 0.12, 0.16];
        assert_eq!(wanted(&times, [0.0, 0.05, 0.13].into_iter()), vec![0, 1, 3]);
        assert_eq!(wanted(&times, [0.01, 0.02, 0.03].into_iter()), vec![0]); // the same picture once
        assert_eq!(wanted(&times, [9.0].into_iter()), vec![4]);
    }

    #[test]
    fn length_prefixed_units_get_start_codes() {
        let sample = [0, 0, 0, 2, 0x65, 0x88, 0, 0, 0, 1, 0x06];
        assert_eq!(annex_b(&sample, 4).unwrap(), vec![0, 0, 0, 1, 0x65, 0x88, 0, 0, 0, 1, 0x06]);
        assert!(annex_b(&[0, 0, 0, 9, 1], 4).is_err());
    }

    #[test]
    fn avc_configuration_gives_its_parameter_sets() {
        // version, profile, compatibility, level, 4-byte lengths, one SPS (2 bytes), one PPS (1 byte)
        let c = [1, 100, 0, 31, 0xff, 0xe1, 0, 2, 0x67, 0x64, 1, 0, 1, 0x68];
        let (headers, length_size) = avc_headers(&c).unwrap();
        assert_eq!(length_size, 4);
        assert_eq!(headers, vec![0, 0, 0, 1, 0x67, 0x64, 0, 0, 0, 1, 0x68]);
        assert!(avc_headers(&c[..9]).is_err());
    }

    /// Saves frames of the video in PHOTO_MAP_VIDEO (every PHOTO_MAP_EVERY seconds, 5 by default) to .cache/test/video/, and says how long
    /// it took. cargo test --release video_frames -- --ignored --nocapture
    #[test]
    #[ignore]
    fn video_frames() {
        let path = std::path::PathBuf::from(std::env::var("PHOTO_MAP_VIDEO").expect("PHOTO_MAP_VIDEO"));
        let every: f64 = std::env::var("PHOTO_MAP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
        let out = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/video");
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();
        let about = info(&path).unwrap();
        eprintln!("{:.1} s, {}x{}", about.seconds, about.width, about.height);
        let started = std::time::Instant::now();
        let mut n = 0;
        frames(&path, every, |f| {
            if n < 12 {
                image::DynamicImage::ImageRgba8(f.image.clone()).to_rgb8().save(out.join(format!("{n:03}_{:.1}s.jpg", f.seconds))).unwrap();
            }
            n += 1;
            true
        })
        .unwrap();
        eprintln!("{n} frames in {:.1} s", started.elapsed().as_secs_f32());
    }
}
