//! Vectors that are already made. Every finished map keeps its photos' vectors (embeddings.bin) and where each photo came from
//! (the full path, in photos.json), so a new build can look a photo up there instead of embedding it again. Nothing is stored
//! for this: the maps are the cache.
//!
//! A vector is reused when the same file is in an earlier map that holds plain image vectors from the same model ("input":
//! "image" in its manifest), and the file has not been changed since that map was started. Vectors of another model are never
//! reused, however alike the maps are: they would put the photo in the wrong place.

use crate::build::time_text;
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

struct Earlier {
    vectors: Mmap,
    /// "2026-10-13 09:25": when the map was started. A file changed after this is not the file that was embedded.
    created: String,
}

/// Where a photo's vector is: a row of an earlier map.
#[derive(Clone, Copy)]
pub struct Hit {
    map: u32,
    row: u32,
}

pub struct Found {
    /// The model whose vectors are looked for, and how many numbers each has.
    model: String,
    dim: usize,
    maps: Vec<Earlier>,
    /// A photo's full path -> where its vector is. Only for the photos asked for: the newest map holding one wins.
    index: HashMap<PathBuf, Hit>,
}

#[derive(Deserialize)]
struct Photo {
    #[serde(default)]
    source: String,
}

#[derive(Deserialize)]
struct Manifest {
    #[serde(default)]
    created: String,
    #[serde(default)]
    photos: usize,
    #[serde(default)]
    embedding: serde_json::Value,
}

impl Found {
    /// Look in the finished maps in `maps` for the photos at the paths in `wanted`: one pass over each map's photos.json (the vectors
    /// are mapped, not read). Only the wanted paths are kept, so what is held is the size of the build, not of all the maps.
    pub fn open(maps: &Path, wanted: &HashSet<PathBuf>, model: &str, dim: usize) -> Self {
        let mut found = Self { model: model.to_string(), dim, maps: Vec::new(), index: HashMap::new() };
        let mut dirs: Vec<(String, PathBuf)> = Vec::new();
        for entry in std::fs::read_dir(maps).into_iter().flatten().flatten() {
            let dir = entry.path();
            if dir.extension().is_some_and(|e| e == "photomap") {
                if let Some(manifest) = read_manifest(&dir, model, dim) {
                    dirs.push((manifest.created, dir));
                }
            }
        }
        dirs.sort(); // oldest first: a newer map's entry replaces an older one's
        for (_, dir) in dirs {
            found.add(&dir, wanted);
        }
        found
    }

    fn add(&mut self, dir: &Path, wanted: &HashSet<PathBuf>) {
        let Some(manifest) = read_manifest(dir, &self.model, self.dim) else { return };
        let Ok(photos) = std::fs::read(dir.join("photos.json")).map_err(drop).and_then(|b| serde_json::from_slice::<Vec<Photo>>(&b).map_err(drop)) else { return };
        let Ok(file) = std::fs::File::open(dir.join("embeddings.bin")) else { return };
        let Ok(vectors) = (unsafe { Mmap::map(&file) }) else { return };
        if photos.len() != manifest.photos || vectors.len() != photos.len() * self.dim * 2 {
            return;
        }
        let map = self.maps.len() as u32;
        self.maps.push(Earlier { vectors, created: manifest.created });
        for (row, photo) in photos.iter().enumerate() {
            let path = Path::new(&photo.source);
            if wanted.contains(path) {
                self.index.insert(path.to_path_buf(), Hit { map, row: row as u32 });
            }
        }
    }

    /// Where the vector of the photo at `path` is, if an earlier map has it and the file is unchanged since that map was started.
    pub fn find(&self, path: &Path) -> Option<Hit> {
        let &hit = self.index.get(path)?;
        let modified = std::fs::metadata(path).ok()?.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_secs() as i64;
        // the map's start is only known to the minute: a file changed in that minute counts as changed
        if time_text(modified).as_str() >= self.maps[hit.map as usize].created.as_str() {
            return None;
        }
        Some(hit)
    }

    /// The vector (float16, as stored), read straight from the earlier map's file.
    pub fn row(&self, hit: Hit) -> &[u8] {
        let row = hit.row as usize;
        &self.maps[hit.map as usize].vectors[row * self.dim * 2..(row + 1) * self.dim * 2]
    }
}

fn read_manifest(dir: &Path, model: &str, dim: usize) -> Option<Manifest> {
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).ok()?).ok()?;
    let e = &manifest.embedding;
    (e["model"] == model && e["dim"] == dim && e["input"] == "image").then_some(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finished map with one photo, whose manifest says it was made with `model`.
    fn map(maps: &Path, name: &str, model: &str, dim: usize, source: &Path) {
        let dir = maps.join(format!("{name}.photomap"));
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = serde_json::json!({ "name": name, "created": "2030-01-01 00:00", "photos": 1, "embedding": { "model": model, "dim": dim, "input": "image" } });
        std::fs::write(dir.join("manifest.json"), manifest.to_string()).unwrap();
        std::fs::write(dir.join("photos.json"), serde_json::json!([{ "source": source }]).to_string()).unwrap();
        std::fs::write(dir.join("embeddings.bin"), vec![1u8; dim * 2]).unwrap();
    }

    /// Vectors of one model are never found for another, even for the same file in a map of the right size.
    #[test]
    fn vectors_are_only_reused_by_the_model_that_made_them() {
        let temp = std::env::temp_dir().join(format!("photo-map-reuse-model-test-{}", std::process::id()));
        let maps = temp.join("maps");
        let photo = temp.join("a.jpg");
        std::fs::create_dir_all(&maps).unwrap();
        std::fs::write(&photo, b"x").unwrap();
        map(&maps, "by-siglip-256", "google/siglip2-base-patch16-256", 768, &photo);
        map(&maps, "by-siglip", "google/siglip2-base-patch16-512", 768, &photo); // the same size of vector, another model
        let wanted: HashSet<PathBuf> = [photo.clone()].into();
        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000); // the photo is older than the maps
        std::fs::File::options().write(true).open(&photo).unwrap().set_modified(old).unwrap();

        for (model, from) in [("google/siglip2-base-patch16-256", "by-siglip-256"), ("google/siglip2-base-patch16-512", "by-siglip")] {
            let found = Found::open(&maps, &wanted, model, 768);
            let hit = found.find(&photo).expect("the model's own map has it");
            assert_eq!(found.maps.len(), 1, "the other model's map isn't even opened ({from})");
            assert_eq!(found.row(hit).len(), 768 * 2);
        }
        assert!(Found::open(&maps, &wanted, "some/other-model", 768).find(&photo).is_none(), "no map of that model");
        assert!(Found::open(&maps, &wanted, "google/siglip2-base-patch16-256", 1024).find(&photo).is_none(), "the size of a vector must match too");
        std::fs::remove_dir_all(&temp).unwrap();
    }
}
