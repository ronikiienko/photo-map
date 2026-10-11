//! Reading a photo file the same way everywhere (bake_pages, streaming, side panel): EXIF orientation applied,
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
