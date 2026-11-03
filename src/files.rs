//! A photo's own file: shown in the file manager, opened in the default app, and what it says about itself (EXIF), read when
//! asked, not kept in the map.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The photo's file where it came from; the map's own copy (or, in maps made before, its link) if that is gone.
pub fn original(data: &crate::data::MapData, id: usize) -> PathBuf {
    let source = Path::new(&data.photos[id].source);
    if !source.as_os_str().is_empty() && source.exists() { source.to_path_buf() } else { data.path(id) }
}

/// The file manager, on the file's folder with the file selected (on Linux by the freedesktop FileManager1 call, which Dolphin,
/// Nautilus and the others answer; the folder alone if none does). Doesn't wait.
pub fn show_in_folder(path: &Path) {
    let path = path.to_path_buf();
    std::thread::spawn(move || {
        let ok = if cfg!(target_os = "macos") {
            run(Command::new("open").arg("-R").arg(&path))
        } else if cfg!(windows) {
            let mut select = std::ffi::OsString::from("/select,");
            select.push(&path);
            run(Command::new("explorer").arg(select)) || true // explorer's exit code is 1 even when it worked
        } else {
            run(Command::new("dbus-send").args([
                "--session",
                "--print-reply",
                "--dest=org.freedesktop.FileManager1",
                "--type=method_call",
                "/org/freedesktop/FileManager1",
                "org.freedesktop.FileManager1.ShowItems",
                &format!("array:string:{}", file_uri(&path)),
                "string:",
            ]))
        };
        if !ok {
            if let Some(dir) = path.parent() {
                open(dir);
            }
        }
    });
}

/// The file (or folder) in the app the system opens it with. Doesn't wait.
pub fn open(path: &Path) {
    let path = path.to_path_buf();
    std::thread::spawn(move || {
        if cfg!(target_os = "macos") {
            run(Command::new("open").arg(&path));
        } else if cfg!(windows) {
            run(Command::new("explorer").arg(&path));
        } else {
            run(Command::new("xdg-open").arg(&path));
        }
    });
}

fn run(command: &mut Command) -> bool {
    command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success())
}

/// file:// and the path, every byte but letters, digits, / and -._~ as %XX.
fn file_uri(path: &Path) -> String {
    let mut uri = String::from("file://");
    for &b in path.to_string_lossy().as_bytes() {
        if b.is_ascii_alphanumeric() || b"/-._~".contains(&b) {
            uri.push(b as char);
        } else {
            uri.push_str(&format!("%{b:02X}"));
        }
    }
    uri
}

/// What there is to say about a photo.
pub struct Details {
    /// Name and value: its file, size, and what the file says about itself (date taken, camera, lens, exposure, place), then
    /// whatever the map keeps about it (`meta`).
    pub lines: Vec<(String, String)>,
    /// Where it was taken (latitude, longitude), if the file says: the "Place" line, named by places.rs.
    pub place: Option<(f64, f64)>,
}

pub fn details(data: &crate::data::MapData, id: usize) -> Details {
    let photo = &data.photos[id];
    let path = original(data, id);
    let mut lines = vec![("File".to_string(), path.display().to_string())];
    match std::fs::metadata(&path) {
        Ok(m) => lines.push(("Size".into(), format!("{} x {} px, {}", photo.width, photo.height, bytes(m.len())))),
        Err(_) => {
            lines.push(("Size".into(), format!("{} x {} px", photo.width, photo.height)));
            lines.push(("".into(), "the file is not there any more".into()));
        }
    }
    let mut place = None;
    if let Ok(file) = std::fs::File::open(&path) {
        if let Ok(exif) = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file)) {
            lines.extend(from_exif(&exif));
            place = coordinates(&exif);
            if place.is_some() {
                lines.push(("Place".into(), String::new()));
            }
        }
    }
    for (k, v) in &photo.meta {
        let v = match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        if !v.trim().is_empty() {
            lines.push((k.clone(), v));
        }
    }
    Details { lines, place }
}

fn from_exif(exif: &exif::Exif) -> Vec<(String, String)> {
    use exif::{In, Tag};
    let text = |tag: Tag| {
        exif.get_field(tag, In::PRIMARY).map(|f| f.display_value().with_unit(exif).to_string().trim_matches(|c: char| c == '"' || c.is_whitespace()).to_string()).filter(|s| !s.is_empty())
    };
    let mut lines = Vec::new();
    let date = |tag: Tag| match &exif.get_field(tag, In::PRIMARY)?.value {
        exif::Value::Ascii(v) => v.first().and_then(|s| in_words(&String::from_utf8_lossy(s))),
        _ => None,
    };
    if let Some(date) = date(Tag::DateTimeOriginal).or_else(|| date(Tag::DateTime)) {
        lines.push(("Taken".into(), date));
    }
    if let Some(camera) = camera(exif) {
        lines.push(("Camera".into(), camera));
    }
    if let Some(lens) = text(Tag::LensModel) {
        lines.push(("Lens".into(), lens));
    }
    let exposure: Vec<String> = [Tag::ExposureTime, Tag::FNumber, Tag::FocalLength].into_iter().filter_map(text).chain(text(Tag::PhotographicSensitivity).map(|iso| format!("ISO {iso}"))).collect();
    if !exposure.is_empty() {
        lines.push(("Exposure".into(), exposure.join(", ")));
    }
    lines
}

/// The camera, by its maker and model ("Canon EOS 550D"; the maker once if the model already says it).
fn camera(exif: &exif::Exif) -> Option<String> {
    use exif::{In, Tag};
    // the raw text: some cameras pad it with empty entries and zeros ("DJI", "", "", ...), which the display form keeps
    let text = |tag: Tag| match &exif.get_field(tag, In::PRIMARY)?.value {
        exif::Value::Ascii(parts) => parts.iter().map(|p| String::from_utf8_lossy(p).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string()).find(|s| !s.is_empty()),
        _ => None,
    };
    match (text(Tag::Make), text(Tag::Model)) {
        (Some(make), Some(model)) if model.to_lowercase().starts_with(&make.to_lowercase()) => Some(model),
        (Some(make), Some(model)) => Some(format!("{make} {model}")),
        (a, b) => a.or(b),
    }
}

/// What a photo's EXIF says that search picks by (data::Facts): its camera, when it was taken, the town nearest to where.
/// None if the file has no EXIF or says none of these. Reads the file's header only.
pub fn facts(path: &Path) -> Option<Box<crate::data::Facts>> {
    use exif::{In, Tag};
    let file = std::fs::File::open(path).ok()?;
    let exif = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file)).ok()?;
    let taken = [Tag::DateTimeOriginal, Tag::DateTime].into_iter().find_map(|tag| match &exif.get_field(tag, In::PRIMARY)?.value {
        exif::Value::Ascii(v) => v.first().and_then(|s| sortable_date(&String::from_utf8_lossy(s))),
        _ => None,
    });
    let place = coordinates(&exif).and_then(|(lat, lon)| crate::places::town_at(lat, lon));
    let facts = crate::data::Facts { camera: camera(&exif).unwrap_or_default(), taken: taken.unwrap_or_default(), place: place.unwrap_or_default() };
    (facts != crate::data::Facts::default()).then(|| Box::new(facts))
}

/// EXIF's "2002:10:25 14:00:31" as "2002-10-25 14:00" (the date alone if there is no time), so that "2002-10" starts it.
fn sortable_date(raw: &str) -> Option<String> {
    let n: Vec<u32> = raw.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
    let (&year, &month, &day) = (n.first()?, n.get(1)?, n.get(2)?);
    if year == 0 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(match (n.get(3), n.get(4)) {
        (Some(h), Some(m)) if *h < 24 && *m < 60 => format!("{year:04}-{month:02}-{day:02} {h:02}:{m:02}"),
        _ => format!("{year:04}-{month:02}-{day:02}"),
    })
}

/// Latitude and longitude from the GPS fields (degrees, minutes, seconds; south and west negative).
fn coordinates(exif: &exif::Exif) -> Option<(f64, f64)> {
    use exif::{In, Tag};
    let degrees = |tag: Tag, sign: Tag, negative: &[u8]| {
        let exif::Value::Rational(v) = &exif.get_field(tag, In::PRIMARY)?.value else { return None };
        let d = v.iter().take(3).zip([1.0, 60.0, 3600.0]).map(|(r, div)| r.to_f64() / div).sum::<f64>();
        let negative = matches!(&exif.get_field(sign, In::PRIMARY)?.value, exif::Value::Ascii(s) if s.first().and_then(|s| s.first()).is_some_and(|c| negative.contains(c)));
        Some(if negative { -d } else { d })
    };
    let at = (degrees(Tag::GPSLatitude, Tag::GPSLatitudeRef, b"Ss")?, degrees(Tag::GPSLongitude, Tag::GPSLongitudeRef, b"Ww")?);
    (at != (0.0, 0.0) && at.0.is_finite() && at.1.is_finite()).then_some(at) // some cameras write zeros when they had no fix
}

/// EXIF's "2002:10:25 14:00:31" as "October 25, 2002, 14:00" (the date alone if there is no time).
fn in_words(raw: &str) -> Option<String> {
    const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    let n: Vec<u32> = raw.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).filter_map(|s| s.parse().ok()).collect();
    let (&year, &month, &day) = (n.first()?, n.get(1)?, n.get(2)?);
    if year == 0 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    let date = format!("{} {day}, {year}", MONTHS[month as usize - 1]);
    Some(match (n.get(3), n.get(4)) {
        (Some(h), Some(m)) if *h < 24 && *m < 60 => format!("{date}, {h:02}:{m:02}"),
        _ => date,
    })
}

fn bytes(n: u64) -> String {
    match n {
        n if n >= 1 << 30 => format!("{:.1} GB", n as f64 / (1u64 << 30) as f64),
        n if n >= 1 << 20 => format!("{:.1} MB", n as f64 / (1u64 << 20) as f64),
        n => format!("{} KB", n.div_ceil(1024)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_in_words() {
        assert_eq!(in_words("2002:10:25 14:00:31").as_deref(), Some("October 25, 2002, 14:00"));
        assert_eq!(in_words("2006:08:01").as_deref(), Some("August 1, 2006"));
        assert_eq!(in_words("0000:00:00 00:00:00"), None);
    }

    #[test]
    fn sortable_dates() {
        assert_eq!(sortable_date("2002:10:25 14:00:31").as_deref(), Some("2002-10-25 14:00"));
        assert_eq!(sortable_date("2006:08:01").as_deref(), Some("2006-08-01"));
        assert_eq!(sortable_date("0000:00:00 00:00:00"), None);
    }

    #[test]
    fn file_uri_escapes_what_it_must() {
        assert_eq!(file_uri(Path::new("/home/a b/ß#1.jpg")), "file:///home/a%20b/%C3%9F%231.jpg");
    }
}
