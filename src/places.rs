//! Naming the place a photo was taken from its coordinates, offline: the nearest town of GeoNames' list of the world's towns of
//! over 1,000 people (171,731; CC BY 4.0, geonames.org), with its region and country. The list is built into the program
//! (assets/places.zip, 2.6 MB: one line per town, made by make_the_list from GeoNames' files) and read the first time a place is
//! asked for.

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Mutex, OnceLock};

/// name, latitude, longitude, "Region, Country": one line per town, in places.tsv inside.
const PACKED: &[u8] = include_bytes!("../assets/places.zip");

#[cfg(test)]
const CITIES: &str = "https://download.geonames.org/export/dump/cities1000.zip";
#[cfg(test)]
const REGIONS: &str = "https://download.geonames.org/export/dump/admin1CodesASCII.txt";
#[cfg(test)]
const COUNTRIES: &str = "https://download.geonames.org/export/dump/countryInfo.txt";
/// Further than this from the nearest town: "near" it; further than FAR_KM: how far.
const NEAR_KM: f64 = 10.0;
const FAR_KM: f64 = 100.0;

struct Town {
    name: String,
    lat: f64,
    lon: f64,
    /// "Region, Country" (either may be missing).
    within: String,
}

enum State {
    Idle,
    Reading,
    Ready(Vec<Town>),
    Failed(String),
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::Idle))
}

/// The place at lat, lon in words ("Girvan, Scotland, United Kingdom"), or what is happening instead; and whether to ask again
/// soon (the list is being read: a moment, the first time).
pub fn describe(lat: f64, lon: f64) -> (String, bool) {
    let mut now = state().lock().unwrap();
    if let State::Idle = *now {
        *now = State::Reading;
        std::thread::spawn(move || {
            let loaded = load();
            *state().lock().unwrap() = match loaded {
                Ok(towns) => State::Ready(towns),
                Err(e) => State::Failed(e),
            };
        });
    }
    match &*now {
        State::Idle | State::Reading => ("finding the place".into(), true),
        State::Failed(e) => (format!("couldn't read the list of places: {e}"), false),
        State::Ready(towns) => (nearest(towns, lat, lon).unwrap_or_default(), false),
    }
}

fn nearest(towns: &[Town], lat: f64, lon: f64) -> Option<String> {
    static ASKED: OnceLock<Mutex<HashMap<(u64, u64), Option<String>>>> = OnceLock::new();
    let asked = ASKED.get_or_init(Default::default);
    asked
        .lock()
        .unwrap()
        .entry((lat.to_bits(), lon.to_bits()))
        .or_insert_with(|| {
            let (km, town) = towns.iter().map(|t| (km_between(lat, lon, t.lat, t.lon), t)).min_by(|a, b| a.0.total_cmp(&b.0))?;
            let near = match km {
                km if km > FAR_KM => format!("{:.0} km from ", km),
                km if km > NEAR_KM => "near ".to_string(),
                _ => String::new(),
            };
            Some(if town.within.is_empty() { format!("{near}{}", town.name) } else { format!("{near}{}, {}", town.name, town.within) })
        })
        .clone()
}

/// Great-circle distance.
fn km_between(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let (dp, dl) = (p2 - p1, (lon2 - lon1).to_radians());
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    2.0 * 6371.0 * a.sqrt().min(1.0).asin()
}

/// The built-in list.
fn load() -> Result<Vec<Town>, String> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(PACKED)).map_err(|e| e.to_string())?;
    let mut text = String::new();
    archive.by_name("places.tsv").map_err(|e| e.to_string())?.read_to_string(&mut text).map_err(|e| e.to_string())?;
    Ok(text
        .lines()
        .filter_map(|line| {
            let mut f = line.split('\t');
            let (name, lat, lon, within) = (f.next()?, f.next()?.parse().ok()?, f.next()?.parse().ok()?, f.next().unwrap_or(""));
            Some(Town { name: name.to_string(), lat, lon, within: within.to_string() })
        })
        .collect())
}

/// GeoNames' three files made into one line per town: name, lat, lon, "Region, Country".
#[cfg(test)]
fn download() -> Result<String, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(std::time::Duration::from_secs(30))).timeout_recv_body(Some(std::time::Duration::from_secs(120))).build().into();
    let get = |url: &str| -> Result<Vec<u8>, String> {
        let mut body = Vec::new();
        agent.get(url).call().map_err(|e| format!("{url}: {e}"))?.into_body().into_reader().read_to_end(&mut body).map_err(|e| e.to_string())?;
        Ok(body)
    };
    let countries: HashMap<String, String> = String::from_utf8_lossy(&get(COUNTRIES)?)
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            Some((f.first()?.to_string(), f.get(4)?.to_string()))
        })
        .collect();
    let regions: HashMap<String, String> = String::from_utf8_lossy(&get(REGIONS)?)
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            Some((f.next()?.to_string(), f.next()?.to_string()))
        })
        .collect();
    let zip = get(CITIES)?;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(zip)).map_err(|e| e.to_string())?;
    let mut cities = String::new();
    archive.by_name("cities1000.txt").map_err(|e| e.to_string())?.read_to_string(&mut cities).map_err(|e| e.to_string())?;
    Ok(towns_from(&cities, &regions, &countries))
}

/// cities1000.txt's columns: 1 name, 4 latitude, 5 longitude, 8 country code, 10 region code. Parts of big cities are towns of their
/// own here ("Stare Misto, Kyiv City, Ukraine"): the region names the city.
#[cfg(test)]
fn towns_from(cities: &str, regions: &HashMap<String, String>, countries: &HashMap<String, String>) -> String {
    let mut out = String::new();
    for line in cities.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 11 {
            continue;
        }
        let (name, country) = (f[1], f[8]);
        let region = regions.get(&format!("{country}.{}", f[10])).map(String::as_str).filter(|r| *r != name);
        let within: Vec<&str> = region.into_iter().chain(countries.get(country).map(String::as_str)).collect();
        out.push_str(&format!("{name}\t{}\t{}\t{}\n", f[4], f[5], within.join(", ")));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_nearest_town() {
        let cities = "1\tGirvan\tGirvan\t\t55.24\t-4.85\tP\tPPL\tGB\t\tSCT\n2\tParis\tParis\t\t48.85\t2.35\tP\tPPLC\tFR\t\t11\n";
        let regions = HashMap::from([("GB.SCT".to_string(), "Scotland".to_string()), ("FR.11".to_string(), "Île-de-France".to_string())]);
        let countries = HashMap::from([("GB".to_string(), "United Kingdom".to_string()), ("FR".to_string(), "France".to_string())]);
        let text = towns_from(cities, &regions, &countries);
        let towns: Vec<Town> = text
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                Town { name: f[0].into(), lat: f[1].parse().unwrap(), lon: f[2].parse().unwrap(), within: f[3].into() }
            })
            .collect();
        assert_eq!(nearest(&towns, 55.25, -4.86).unwrap(), "Girvan, Scotland, United Kingdom");
        assert_eq!(nearest(&towns, 48.0, 2.35).unwrap(), "near Paris, Île-de-France, France");
        assert!((km_between(48.85, 2.35, 51.51, -0.13) - 344.0).abs() < 5.0, "Paris to London");
    }

    /// The built-in list is read and names real places.
    #[test]
    fn names_real_places() {
        let towns = load().unwrap();
        assert!(towns.len() > 150_000, "{} towns", towns.len());
        assert_eq!(nearest(&towns, 55.2375, -4.8540).unwrap(), "Girvan, Scotland, United Kingdom");
        assert_eq!(nearest(&towns, 64.1466, -21.9426).unwrap(), "Reykjavík, Capital Region, Iceland");
        assert!(nearest(&towns, 0.0, -30.0).unwrap().contains(" km from "), "mid-ocean");
    }

    /// Makes assets/places.zip again from GeoNames' files (needs the network, 10 MB):
    /// cargo test --release make_the_list -- --ignored --nocapture
    #[test]
    #[ignore]
    fn make_the_list() {
        use std::io::Write;
        let text = download().unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/places.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
        zip.start_file("places.tsv", zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated).compression_level(Some(9))).unwrap();
        zip.write_all(text.as_bytes()).unwrap();
        zip.finish().unwrap();
        eprintln!("{} towns, {} bytes", text.lines().count(), std::fs::metadata(&path).unwrap().len());
    }
}
