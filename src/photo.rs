//! Reading a photo file the same way everywhere (baking, streaming): EXIF orientation applied,
//! transparent parts filled with the viewer's background.

use image::{DynamicImage, ImageDecoder, ImageReader, RgbaImage};
use std::path::Path;

pub const BG: [u8; 3] = [12, 13, 16];

pub fn load(path: &Path) -> Option<RgbaImage> {
    let mut decoder = ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_decoder().ok()?;
    let orientation = decoder.orientation().ok()?;
    let mut img = DynamicImage::from_decoder(decoder).ok()?;
    img.apply_orientation(orientation);
    let mut rgba = img.to_rgba8();
    for p in rgba.pixels_mut() {
        if p[3] < 255 {
            let a = p[3] as u32;
            for c in 0..3 {
                p[c] = ((p[c] as u32 * a + BG[c] as u32 * (255 - a)) / 255) as u8;
            }
            p[3] = 255;
        }
    }
    Some(rgba)
}

/// Like `load`, for when the photo is going to be made smaller anyway: a JPEG that is bigger than needed is decoded
/// at 1/8 to 7/8 of its size (the decoder leaves out the fine detail while it decodes, which is many times faster than
/// decoding everything and shrinking after). `needed(width, height)` gets the photo's size as it is shown (turned upright) and
/// says how long the longer side must be at least; the result is never smaller than twice that (or it is the full size). Other files are read as `load` does.
pub fn load_for(path: &Path, needed: impl Fn(u32, u32) -> u32) -> Option<RgbaImage> {
    scaled_jpeg(path, &needed).or_else(|| load(path))
}

fn scaled_jpeg(path: &Path, needed: &dyn Fn(u32, u32) -> u32) -> Option<RgbaImage> {
    use image::metadata::Orientation::*;
    let bytes = std::fs::read(path).ok()?;
    let reader = ImageReader::new(std::io::Cursor::new(&bytes[..])).with_guessed_format().ok()?;
    if reader.format() != Some(image::ImageFormat::Jpeg) {
        return None;
    }
    let orientation = reader.into_decoder().ok()?.orientation().ok()?;
    let info = libjpeg_turbo_rs::probe(&bytes).ok()?;
    // libjpeg-turbo-rs 0.8 panics on CMYK (YCCK) JPEGs, reading past the end of a row: those are read by `load`
    if info.components == 4 {
        return None;
    }
    let (w, h) = (info.width as u32, info.height as u32);
    let (shown_w, shown_h) = if matches!(orientation, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH) { (h, w) } else { (w, h) };
    // twice what is needed is kept: the shrink that follows then has something to average, and the picture stays as sharp as it
    // would be from the full size (decoded at exactly the needed size it comes out a little aliased)
    let need = needed(shown_w, shown_h) * 2;
    if need == 0 {
        return None;
    }
    // the smallest size, in eighths, whose longer side is still enough
    let long = w.max(h);
    let eighths = (1..=8u32).find(|&n| (long * n).div_ceil(8) >= need).unwrap_or(8);
    // a decoder bug on some other odd file shouldn't take a build's worker down with it: that photo too is read by `load`
    let decoded = std::panic::catch_unwind(|| {
        let mut decoder = libjpeg_turbo_rs::Decoder::new(&bytes).ok()?;
        decoder.set_scale(libjpeg_turbo_rs::ScalingFactor::new(eighths, 8));
        decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgba);
        decoder.decode_image().ok()
    })
    .ok()??;
    let mut img = DynamicImage::ImageRgba8(RgbaImage::from_raw(decoded.width as u32, decoded.height as u32, decoded.data)?);
    img.apply_orientation(orientation);
    Some(img.into_rgba8())
}

/// Scaled so the longer side is `long` px (smaller photos are enlarged only if `enlarge`).
pub fn fit(img: &RgbaImage, long: u32, enlarge: bool) -> RgbaImage {
    let cur = img.width().max(img.height());
    if cur == long || (cur < long && !enlarge) {
        return img.clone();
    }
    let s = long as f32 / cur as f32;
    let (w, h) = (((img.width() as f32 * s).round() as u32).max(1), ((img.height() as f32 * s).round() as u32).max(1));
    image::imageops::resize(img, w, h, image::imageops::FilterType::Triangle)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A JPEG of 800 x 600 (a gradient) saved in the project's test folder.
    fn jpeg(name: &str) -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{name}-{}.jpg", std::process::id()));
        let img = image::RgbImage::from_fn(800, 600, |x, y| image::Rgb([(x / 4) as u8, (y / 3) as u8, 128]));
        img.save(&path).unwrap();
        path
    }

    use std::path::PathBuf;

    #[test]
    fn a_big_jpeg_is_decoded_at_the_smallest_eighth_that_is_enough() {
        let path = jpeg("eighths");
        // 200 needed, so 400 kept: 4/8 of 800 is exactly 400
        let img = load_for(&path, |_, _| 200).unwrap();
        assert_eq!((img.width(), img.height()), (400, 300));
        // more than the photo has: the full size
        let img = load_for(&path, |_, _| 2000).unwrap();
        assert_eq!((img.width(), img.height()), (800, 600));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn the_scaled_picture_looks_like_the_full_one() {
        let path = jpeg("alike");
        let small = load_for(&path, |_, _| 100).unwrap();
        let full = load(&path).unwrap();
        let (sx, sy) = (full.width() / small.width(), full.height() / small.height());
        let mut worst = 0;
        for y in (0..small.height()).step_by(7) {
            for x in (0..small.width()).step_by(7) {
                let (a, b) = (small.get_pixel(x, y), full.get_pixel(x * sx + sx / 2, y * sy + sy / 2));
                for c in 0..3 {
                    worst = worst.max((a[c] as i32 - b[c] as i32).abs());
                }
            }
        }
        assert!(worst < 12, "the scaled picture differs from the full size by up to {worst} of 255");
        let _ = std::fs::remove_file(path);
    }
}
