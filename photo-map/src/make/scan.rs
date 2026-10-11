//! Every image under the folders, with what the file says about itself (size, EXIF, PNG text), copied flat into
//! originals/: photo i is originals/<i, 6 digits>.<ext>. Where it came from is kept only as a label ("source").
//! Unreadable files are left out, so the numbers have no gaps.

use super::step;
use crate::data::Photo;
use exif::{In, Tag, Value};
use rayon::prelude::*;
use serde_json::Map;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};

const IMAGE_TYPES: [&str; 8] = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"];
/// What cameras write into the description field when the photographer wrote nothing.
const BOILERPLATE: [&str; 12] = [
    "sony dsc", "olympus digital camera", "konica minolta digital camera", "digital camera", "dcim",
    "exif_jpeg_picture", "default", "samsung", "leica", "picture", "image", "untitled",
];

pub fn scan(folders: &[PathBuf], bundle: &Path) -> io::Result<Vec<Photo>> {
    let mut sources = Vec::new(); // (file, label)
    for folder in folders {
        let name = folder.file_name().unwrap_or_default().to_string_lossy().into_owned();
        for entry in walkdir::WalkDir::new(folder).sort_by_file_name().into_iter().filter_map(Result::ok) {
            let rel = entry.path().strip_prefix(folder).unwrap_or(entry.path());
            let hidden = rel.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'));
            if entry.file_type().is_file() && !hidden && IMAGE_TYPES.contains(&extension(entry.path()).as_str()) {
                sources.push((entry.path().to_path_buf(), format!("{name}/{}", rel.to_string_lossy())));
            }
        }
    }
    step(format!("{} images found", sources.len()));

    let read: Vec<Option<Photo>> = sources.par_iter().map(|(path, label)| read_photo(path, label)).collect();
    let mut photos = Vec::new();
    let mut origin = Vec::new();
    for ((path, _), photo) in sources.iter().zip(read) {
        if let Some(mut p) = photo {
            p.file = format!("originals/{:06}.{}", photos.len(), extension(path));
            photos.push(p);
            origin.push(path);
        }
    }
    let with_meta = photos.iter().filter(|p| !p.meta.is_empty()).count();
    step(format!("{} readable, {} skipped; {with_meta} carry metadata", photos.len(), sources.len() - photos.len()));

    std::fs::create_dir_all(bundle.join("originals"))?;
    photos.par_iter().zip(&origin).try_for_each(|(p, src)| std::fs::copy(src, bundle.join(&p.file)).map(|_| ()))?;
    step("originals copied");
    Ok(photos)
}

fn extension(path: &Path) -> String {
    path.extension().unwrap_or_default().to_string_lossy().to_lowercase()
}

/// Size (as shown, after EXIF rotation) and metadata; None if the file can't be read as an image.
fn read_photo(path: &Path, label: &str) -> Option<Photo> {
    let (mut width, mut height) = image::ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_dimensions().ok()?;
    let mut meta = Map::new();
    let mut title = None;
    let mut add = |key: &str, value: Option<String>| {
        if let Some(v) = value.as_deref().and_then(clean) {
            meta.entry(key).or_insert(v.into());
        }
    };
    let exif = File::open(path).ok().and_then(|f| exif::Reader::new().read_from_container(&mut BufReader::new(f)).ok());
    if let Some(e) = &exif {
        let field = |tag| e.get_field(tag, In::PRIMARY);
        if matches!(field(Tag::Orientation).and_then(|f| f.value.get_uint(0)), Some(5..=8)) {
            (width, height) = (height, width); // rotated a quarter turn
        }
        title = text(field(Tag(exif::Context::Tiff, 0x9C9B))).as_deref().and_then(clean); // Windows XPTitle
        add("description", text(field(Tag::ImageDescription)));
        add("date taken", text(field(Tag::DateTimeOriginal)).or_else(|| text(field(Tag::DateTime))));
        let camera: Vec<String> = [Tag::Make, Tag::Model].into_iter().filter_map(|t| text(field(t)).as_deref().and_then(clean)).collect();
        add("camera", Some(camera.join(" ")));
        add("artist", text(field(Tag::Artist)));
        add("copyright", text(field(Tag::Copyright)));
        add("location", gps(e));
    }
    if extension(path) == "png" {
        for (key, value) in png_text(path) {
            match key.to_lowercase().as_str() {
                "title" if title.is_none() => title = clean(&value),
                k @ ("description" | "author" | "comment") => add(k, Some(value)),
                _ => {}
            }
        }
    }
    let title = title
        .or_else(|| meta.get("description").and_then(|v| v.as_str()).map(String::from))
        .unwrap_or_else(|| path.file_stem().unwrap_or_default().to_string_lossy().into_owned());
    Some(Photo { file: String::new(), source: label.to_string(), width, height, title, meta })
}

/// Trimmed, without camera boilerplate and empty or symbol-only values, at most 500 characters.
fn clean(s: &str) -> Option<String> {
    let s = s.replace('\0', "");
    let s = s.trim();
    let boring = s.chars().all(|c| !c.is_alphanumeric()) || BOILERPLATE.iter().any(|b| s.eq_ignore_ascii_case(b));
    (!boring).then(|| s.chars().take(500).collect())
}

fn text(field: Option<&exif::Field>) -> Option<String> {
    match &field?.value {
        Value::Ascii(parts) => Some(parts.iter().map(|p| decode(p)).collect::<Vec<_>>().join(" ")),
        Value::Byte(b) | Value::Undefined(b, _) => Some(decode(b)),
        _ => None,
    }
}

/// UTF-16 (what Windows writes) when it looks like it, else UTF-8.
fn decode(b: &[u8]) -> String {
    if b.len() % 2 == 0 && b.contains(&0) {
        String::from_utf16_lossy(&b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect::<Vec<_>>())
    } else {
        String::from_utf8_lossy(b).into_owned()
    }
}

/// EXIF GPS -> "lat, lon" in decimal degrees.
fn gps(e: &exif::Exif) -> Option<String> {
    let degrees = |value: Tag, side: Tag| {
        let Value::Rational(v) = &e.get_field(value, In::PRIMARY)?.value else { return None };
        let d = v.get(0)?.to_f64() + v.get(1)?.to_f64() / 60.0 + v.get(2)?.to_f64() / 3600.0;
        let negative = text(e.get_field(side, In::PRIMARY)).is_some_and(|s| matches!(s.trim(), "S" | "W"));
        d.is_finite().then_some(if negative { -d } else { d })
    };
    let lat = degrees(Tag::GPSLatitude, Tag::GPSLatitudeRef)?;
    let lon = degrees(Tag::GPSLongitude, Tag::GPSLongitudeRef)?;
    Some(format!("{lat:.6}, {lon:.6}"))
}

/// PNG text chunks before the image data: (keyword, text).
fn png_text(path: &Path) -> Vec<(String, String)> {
    let Ok(file) = File::open(path) else { return Vec::new() };
    let Ok(reader) = png::Decoder::new(BufReader::new(file)).read_info() else { return Vec::new() };
    let info = reader.info();
    let plain = info.uncompressed_latin1_text.iter().map(|t| (t.keyword.clone(), t.text.clone()));
    let zipped = info.compressed_latin1_text.iter().filter_map(|t| Some((t.keyword.clone(), t.get_text().ok()?)));
    let utf8 = info.utf8_text.iter().filter_map(|t| Some((t.keyword.clone(), t.get_text().ok()?)));
    plain.chain(zipped).chain(utf8).collect()
}
