//! The maps the app knows: every finished bundle in the maps/ folder (main.rs: data_root) (maps/<name>.photomap, see data.rs) and every
//! one still being made (maps/<name>.building, see build.rs).

use crate::build::Job;
use std::path::{Path, PathBuf};

pub struct Entry {
    pub path: PathBuf,
    pub name: String,
    /// Everything on the map, videos included.
    pub photos: usize,
    /// How many of those are videos.
    pub videos: usize,
    /// "2026-10-11 14:00", if the map says.
    pub created: String,
    /// The id of the model the map was made with (models.rs).
    pub model: String,
}

/// A map being made that is not finished.
pub struct Unfinished {
    pub dir: PathBuf,
    pub job: Job,
}

pub struct Library {
    dir: PathBuf,
    pub maps: Vec<Entry>,
    pub builds: Vec<Unfinished>,
}

impl Library {
    pub fn new(dir: PathBuf) -> Self {
        let mut lib = Self { dir, maps: Vec::new(), builds: Vec::new() };
        lib.refresh();
        lib
    }

    /// Every finished map (one with a manifest) and every unfinished one (one with a job), by name.
    pub fn refresh(&mut self) {
        let entries: Vec<PathBuf> = std::fs::read_dir(&self.dir).into_iter().flatten().filter_map(Result::ok).map(|e| e.path()).collect();
        self.maps = entries
            .iter()
            .filter(|p| p.extension().is_some_and(|e| e == "photomap"))
            .filter_map(|p| {
                let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("manifest.json")).ok()?).ok()?;
                Some(Entry {
                    path: p.clone(),
                    name: manifest["name"].as_str()?.to_string(),
                    photos: manifest["photos"].as_u64()? as usize,
                    videos: manifest["videos"].as_u64().unwrap_or(0) as usize,
                    created: manifest["created"].as_str().unwrap_or_default().to_string(),
                    model: manifest["embedding"]["model"].as_str().unwrap_or(crate::models::DEFAULT).to_string(),
                })
            })
            .collect();
        self.maps.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
        self.builds = entries
            .iter()
            .filter(|p| p.extension().is_some_and(|e| e == "building"))
            .filter_map(|p| Some(Unfinished { dir: p.clone(), job: Job::read(p)? }))
            .collect();
        self.builds.sort_by(|a, b| a.job.created.cmp(&b.job.created));
    }

    /// Continue unfinished maps by themselves when the app starts? (A file in maps/ says.)
    pub fn resume_on_start(&self) -> bool {
        self.dir.join(".resume").exists()
    }

    pub fn set_resume_on_start(&self, on: bool) {
        let file = self.dir.join(".resume");
        if on {
            let _ = std::fs::write(file, "");
        } else {
            let _ = std::fs::remove_file(file);
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The app's folder (maps/ is in it; so are runtime/ and .cache/): main.rs, data_root.
    pub fn root(&self) -> PathBuf {
        self.dir.parent().map(PathBuf::from).unwrap_or_default()
    }
}
