//! `photo-map make <bundle> <folder>...`: a map bundle from folders of images.
//! scan (find the images, read what they say about themselves, copy them in) -> embed (one vector per photo,
//! via src/embed_worker.py) -> layout (similar photos close together) -> regions (names for areas of the map)
//! -> bake (the always-loaded small levels).
//! The bundle's files are described in data.rs; work/ holds the embedding shards until the end.

mod bake;
mod embed;
mod layout;
mod regions;
mod scan;

use crate::data::WORLD;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

static START: Mutex<Option<Instant>> = Mutex::new(None);
static LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// One progress line, with the time since the start; printed, and kept for the app's "New map" window.
pub fn step(msg: impl std::fmt::Display) {
    let elapsed = START.lock().unwrap().get_or_insert_with(Instant::now).elapsed().as_secs_f32();
    let line = format!("[{elapsed:5.0}s] {msg}");
    println!("{line}");
    LOG.lock().unwrap().push(line);
}

/// The last `n` progress lines of the current (or last) run.
pub fn recent(n: usize) -> Vec<String> {
    let log = LOG.lock().unwrap();
    log[log.len().saturating_sub(n)..].to_vec()
}

/// `project` is the EmbeddingTest folder (Python environment and src/embed_worker.py).
pub fn run(project: &Path, bundle: &Path, folders: &[PathBuf]) -> io::Result<()> {
    *START.lock().unwrap() = Some(Instant::now());
    LOG.lock().unwrap().clear();
    step(format!("making {}", bundle.display()));
    std::fs::create_dir_all(bundle.join("work"))?;
    let mut python = embed::Python::new(project, &bundle.join("work"));
    let folders = folders.iter().map(|f| f.canonicalize()).collect::<io::Result<Vec<_>>>()?;

    let photos = scan::scan(&folders, bundle)?;
    if photos.len() < 2 {
        return Err(io::Error::other("need at least 2 readable images"));
    }
    std::fs::write(bundle.join("photos.json"), serde_json::to_vec(&photos)?)?;

    let vectors = embed::embed(&mut python, bundle, &photos)?;

    let xy = layout::lay_out(&vectors, photos.len(), WORLD);
    std::fs::write(bundle.join("layout.bin"), bytemuck::cast_slice(&xy))?;
    step("layout.bin written");

    regions::regions(project, &mut python, bundle, &photos, &vectors, &xy)?;
    python.finish()?;

    let name = bundle.file_stem().unwrap_or_default().to_string_lossy();
    let manifest = serde_json::json!({
        "format": 1, "name": name, "photos": photos.len(), "world": WORLD,
        "embedding": {"model": "google/embeddinggemma-2", "dim": embed::DIM, "dtype": "float16", "image_tokens": 140},
        "page": crate::pages::PAGE, "levels": crate::pages::LEVELS, "baked_levels": crate::pages::STREAMED,
    });
    std::fs::write(bundle.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?)?;

    bake::bake(bundle, &photos)?;
    std::fs::remove_dir_all(bundle.join("work"))?;
    step(format!("MAP_DONE: {}", bundle.display()));
    Ok(())
}

/// `photo-map regions <bundle>`: (re)name the regions of an existing bundle, without making it again.
pub fn add_regions(project: &Path, bundle: &Path) -> io::Result<()> {
    *START.lock().unwrap() = Some(Instant::now());
    LOG.lock().unwrap().clear();
    let photos: Vec<crate::data::Photo> = serde_json::from_slice(&std::fs::read(bundle.join("photos.json"))?)?;
    let half: Vec<u16> = bytemuck::pod_collect_to_vec(&std::fs::read(bundle.join("embeddings.bin"))?);
    let vectors: Vec<f32> = half.iter().map(|&h| half::f16::from_bits(h).to_f32()).collect();
    let xy: Vec<[f32; 2]> = bytemuck::pod_collect_to_vec(&std::fs::read(bundle.join("layout.bin"))?);
    if vectors.len() != photos.len() * embed::DIM || xy.len() != photos.len() {
        return Err(io::Error::other("bundle files don't match"));
    }
    std::fs::create_dir_all(bundle.join("work"))?;
    let mut python = embed::Python::new(project, &bundle.join("work"));
    regions::regions(project, &mut python, bundle, &photos, &vectors, &xy)?;
    python.finish()?;
    std::fs::remove_dir_all(bundle.join("work"))?;
    step(format!("REGIONS_DONE: {}", bundle.display()));
    Ok(())
}
