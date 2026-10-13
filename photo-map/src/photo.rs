//! Reading a photo file the same way everywhere (baking, streaming): EXIF orientation applied,
//! transparent parts filled with the viewer's background.

use image::{DynamicImage, ImageDecoder, ImageReader, RgbaImage};
use std::path::Path;

pub const BG: [u8; 3] = [21, 23, 28];

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
    let mut decompressor = turbojpeg::Decompressor::new().ok()?;
    let header = decompressor.read_header(&bytes).ok()?;
    let (w, h) = (header.width as u32, header.height as u32);
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
    let factor = turbojpeg::ScalingFactor::new(eighths as usize, 8);
    decompressor.set_scaling_factor(factor).ok()?;
    let size = header.scaled(factor);
    let mut pixels = vec![0u8; size.width * size.height * 4];
    let image = turbojpeg::Image { pixels: &mut pixels[..], width: size.width, pitch: size.width * 4, height: size.height, format: turbojpeg::PixelFormat::RGBA };
    decompressor.decompress(&bytes, image).ok()?;
    let mut img = DynamicImage::ImageRgba8(RgbaImage::from_raw(size.width as u32, size.height as u32, pixels)?);
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
