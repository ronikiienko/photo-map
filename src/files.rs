//! A photo's own file: shown in the file manager, opened in the default app, and what it says about itself (a photo's EXIF, a
//! video's tags), read once when the map is made and kept in it (data::Facts).

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The photo's (or the video's) own file.
pub fn original(data: &crate::data::MapData, id: usize) -> PathBuf {
    PathBuf::from(&data.photos[id].source)
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
    /// Name and value: its file, size, and what the map read from the file when it was made (data::Facts: date taken, camera,
    /// lens, exposure, place), then whatever else the map keeps about it (`meta`).
    pub lines: Vec<(String, String)>,
    /// Where it was taken (latitude, longitude), if the file said: shown under the "Place" line.
    pub place: Option<(f64, f64)>,
}

pub fn details(data: &crate::data::MapData, id: usize) -> Details {
    let photo = &data.photos[id];
    let path = original(data, id);
    let mut lines = vec![("File".to_string(), path.display().to_string())];
    // a video's own size (the photo's is its poster's), and how long it is
    let (width, height) = photo.video.map_or((photo.width, photo.height), |v| (v.width, v.height));
    match std::fs::metadata(&path) {
        Ok(m) => lines.push(("Size".into(), format!("{width} x {height} px, {}", bytes(m.len())))),
        Err(_) => {
            lines.push(("Size".into(), format!("{width} x {height} px")));
            lines.push(("".into(), "the file is not there any more".into()));
        }
    }
    if let Some(v) = photo.video {
        lines.push(("Length".into(), crate::clock(v.duration)));
    }
    let mut place = None;
    if let Some(f) = photo.facts.as_deref() {
        let mut line = |name: &str, value: &str| {
            if !value.is_empty() {
                lines.push((name.into(), value.into()));
            }
        };
        line("Taken", &in_words(&f.taken).unwrap_or_default());
        line("Camera", &f.camera);
        line("Lens", &f.lens);
        line("Exposure", &f.exposure);
        place = f.at.map(|[lat, lon]| (lat as f64, lon as f64));
        if place.is_some() || !f.place.is_empty() {
            lines.push(("Place".into(), f.place.clone()));
        }
    }
    Details { lines, place }
}

/// The camera, by its maker and model ("Canon EOS 550D"; the maker once if the model already says it).
fn camera_name(make: Option<String>, model: Option<String>) -> Option<String> {
    match (make, model) {
        (Some(make), Some(model)) if model.to_lowercase().starts_with(&make.to_lowercase()) => Some(model),
        (Some(make), Some(model)) => Some(format!("{make} {model}")),
        (a, b) => a.or(b),
    }
}

/// What a photo's EXIF says (data::Facts): its camera, when it was taken, where (and the town nearest to it), the lens and the
/// exposure. None if the file has no EXIF or says none of these. Reads the file's header only.
pub fn facts(path: &Path) -> Option<Box<crate::data::Facts>> {
    use exif::{In, Tag};
    let file = std::fs::File::open(path).ok()?;
    let exif = exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file)).ok()?;
    // the raw text: some cameras pad it with empty entries and zeros ("DJI", "", "", ...), which the display form keeps
    let raw = |tag: Tag| match &exif.get_field(tag, In::PRIMARY)?.value {
        exif::Value::Ascii(parts) => parts.iter().map(|p| String::from_utf8_lossy(p).trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string()).find(|s| !s.is_empty()),
        _ => None,
    };
    let text = |tag: Tag| {
        exif.get_field(tag, In::PRIMARY).map(|f| f.display_value().with_unit(&exif).to_string().trim_matches(|c: char| c == '"' || c.is_whitespace()).to_string()).filter(|s| !s.is_empty())
    };
    let taken = raw(Tag::DateTimeOriginal).or_else(|| raw(Tag::DateTime)).and_then(|d| sortable_date(&d));
    let at = coordinates(&exif);
    let exposure: Vec<String> = [Tag::ExposureTime, Tag::FNumber, Tag::FocalLength].into_iter().filter_map(text).chain(text(Tag::PhotographicSensitivity).map(|iso| format!("ISO {iso}"))).collect();
    let facts = crate::data::Facts {
        camera: camera_name(raw(Tag::Make), raw(Tag::Model)).unwrap_or_default(),
        taken: taken.unwrap_or_default(),
        place: at.and_then(|(lat, lon)| crate::places::town_at(lat, lon)).unwrap_or_default(),
        at: at.map(|(lat, lon)| [lat as f32, lon as f32]),
        lens: text(Tag::LensModel).unwrap_or_default(),
        exposure: exposure.join(", "),
    };
    (facts != crate::data::Facts::default()).then(|| Box::new(facts))
}

/// What a video's file says about itself (data::Facts), from what ffmpeg wrote reading it (`log`): the file's own tags, before
/// its streams ("    creation_time   : 2024-08-23T22:15:58.000000Z"). Phones write where (ISO 6709: "+50.4501+030.5234+123.0/"),
/// Apple's also the local time (quicktime.creationdate; creation_time is UTC). None if it says nothing.
pub fn video_facts(log: &str) -> Option<Box<crate::data::Facts>> {
    let tags: std::collections::HashMap<&str, &str> = log
        .lines()
        .take_while(|l| !l.trim_start().starts_with("Stream #"))
        .filter_map(|l| {
            let (k, v) = l.split_once(':')?;
            Some((k.trim(), v.trim()))
        })
        .filter(|(k, v)| !k.is_empty() && !v.is_empty())
        .collect();
    let tag = |keys: &[&str]| keys.iter().find_map(|k| tags.get(k).map(|v| v.to_string()));
    let taken = tag(&["com.apple.quicktime.creationdate", "creation_time"]).and_then(|d| sortable_date(&d));
    let camera = camera_name(tag(&["com.apple.quicktime.make", "make", "com.android.manufacturer"]), tag(&["com.apple.quicktime.model", "model", "com.android.model"]));
    let at = tag(&["com.apple.quicktime.location.ISO6709", "location"]).and_then(|l| iso6709(&l));
    let facts = crate::data::Facts {
        camera: camera.unwrap_or_default(),
        taken: taken.unwrap_or_default(),
        place: at.and_then(|(lat, lon)| crate::places::town_at(lat, lon)).unwrap_or_default(),
        at: at.map(|(lat, lon)| [lat as f32, lon as f32]),
        ..Default::default()
    };
    (facts != crate::data::Facts::default()).then(|| Box::new(facts))
}

/// "+50.4501+030.5234+123.0/" as latitude and longitude (None for zeros: no fix).
fn iso6709(s: &str) -> Option<(f64, f64)> {
    let starts: Vec<usize> = s.char_indices().filter(|(_, c)| *c == '+' || *c == '-').map(|(i, _)| i).collect();
    let number = |k: usize| s[*starts.get(k)?..starts.get(k + 1).copied().unwrap_or(s.len())].trim_end_matches('/').parse::<f64>().ok();
    let at = (number(0)?, number(1)?);
    (at != (0.0, 0.0) && (-90.0..=90.0).contains(&at.0) && (-180.0..=180.0).contains(&at.1)).then_some(at)
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

    /// A video's tags, as ffmpeg writes them reading the file (an iPhone's: its local time, where; a drone's zeros: no fix).
    #[test]
    fn video_facts_from_ffmpegs_log() {
        let iphone = "Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'IMG_0001.MOV':
  Metadata:
    major_brand     : qt
    creation_time   : 2023-06-01T09:30:00.000000Z
    com.apple.quicktime.location.ISO6709: +49.8397+024.0297+290.000/
    com.apple.quicktime.make: Apple
    com.apple.quicktime.model: iPhone 12
    com.apple.quicktime.creationdate: 2023-06-01T12:30:00+0300
  Duration: 00:00:05.00, start: 0.000000, bitrate: 9000 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv), 1920x1080, 8000 kb/s
    Metadata:
      creation_time   : 2023-06-01T09:30:00.000000Z
      handler_name    : Core Media Video
";
        let f = video_facts(iphone).unwrap();
        assert_eq!((f.camera.as_str(), f.taken.as_str()), ("Apple iPhone 12", "2023-06-01 12:30"));
        assert_eq!(f.at, Some([49.8397, 24.0297]));
        let drone = "  Metadata:
    creation_time   : 2024-08-23T22:15:58.000000Z
    location        : +0.0000+0.0000
  Stream #0:0[0x1](eng): Video: h264 (High), 1920x1080
";
        let f = video_facts(drone).unwrap();
        assert_eq!((f.taken.as_str(), f.at, f.camera.as_str()), ("2024-08-23 22:15", None, ""));
        assert_eq!(iso6709("-33.8688+151.2093/"), Some((-33.8688, 151.2093)));
        assert_eq!(video_facts("  Stream #0:0: Video: h264\n    creation_time   : 2024-01-01T00:00:00Z\n"), None, "a stream's own tags don't count");
    }

    #[test]
    fn file_uri_escapes_what_it_must() {
        assert_eq!(file_uri(Path::new("/home/a b/ß#1.jpg")), "file:///home/a%20b/%C3%9F%231.jpg");
    }
}
