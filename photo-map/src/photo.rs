//! Reading a photo file the same way everywhere (baking, streaming): EXIF orientation applied,
//! transparent parts filled with the viewer's background.

use image::{DynamicImage, GrayImage, ImageDecoder, ImageReader, RgbImage, RgbaImage};
use std::io::BufReader;
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

/// Like `load`, for when the photo is going to be made smaller anyway: a JPEG that is at least twice as big as needed is decoded
/// at 1/2, 1/4 or 1/8 of its size (the decoder leaves out the fine detail while it decodes, which is many times faster than
/// decoding everything and shrinking after). `needed(width, height)` gets the photo's size as it is shown (turned upright) and
/// says how long the longer side must be at least; the result is never smaller than twice that (or it is the full size). Other files are read as `load` does.
pub fn load_for(path: &Path, needed: impl Fn(u32, u32) -> u32) -> Option<RgbaImage> {
    scaled_jpeg(path, &needed).or_else(|| load(path))
}

fn scaled_jpeg(path: &Path, needed: &dyn Fn(u32, u32) -> u32) -> Option<RgbaImage> {
    use image::metadata::Orientation::*;
    let reader = ImageReader::open(path).ok()?.with_guessed_format().ok()?;
    if reader.format() != Some(image::ImageFormat::Jpeg) {
        return None;
    }
    let mut decoder = reader.into_decoder().ok()?;
    let orientation = decoder.orientation().ok()?;
    let (w, h) = decoder.dimensions();
    let (shown_w, shown_h) = if matches!(orientation, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH) { (h, w) } else { (w, h) };
    // twice what is needed is kept: the shrink that follows then has something to average, and the picture stays as sharp as it
    // would be from the full size (decoded at exactly the needed size it comes out a little aliased)
    let need = needed(shown_w, shown_h) * 2;
    let long = w.max(h);
    if need == 0 || long < need * 2 {
        return None; // not even a halving fits
    }
    let k = need as f64 / long as f64;
    let (ask_w, ask_h) = (((w as f64 * k).ceil() as u16).max(1), ((h as f64 * k).ceil() as u16).max(1));
    let mut jpeg = jpeg_decoder::Decoder::new(BufReader::new(std::fs::File::open(path).ok()?));
    jpeg.scale(ask_w, ask_h).ok()?;
    let pixels = jpeg.decode().ok()?;
    let info = jpeg.info()?;
    let (sw, sh) = (info.width as u32, info.height as u32);
    if sw.max(sh) < need {
        return None;
    }
    let mut img = match info.pixel_format {
        jpeg_decoder::PixelFormat::RGB24 => DynamicImage::ImageRgb8(RgbImage::from_raw(sw, sh, pixels)?),
        jpeg_decoder::PixelFormat::L8 => DynamicImage::ImageLuma8(GrayImage::from_raw(sw, sh, pixels)?),
        _ => return None, // CMYK and 16-bit grey: the general reader
    };
    img.apply_orientation(orientation);
    Some(img.to_rgba8())
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
