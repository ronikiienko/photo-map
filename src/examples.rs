//! Example photos to make a first map from, for people without a folder of photos at hand. One is downloaded as a zip into
//! `.cache/examples/`, its photos are taken out into `examples/<folder>/` (through `<folder>.part`, so a folder that is there is
//! whole) and the zip is deleted. No map is made: the photos are offered to the New map form.

use crate::modelstore::{self, Job, Remote};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct Example {
    pub id: &'static str,
    pub label: &'static str,
    pub about: &'static str,
    pub photos: usize,
    /// The folder under `examples/` its photos go in.
    pub folder: &'static str,
    pub url: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const ALL: [Example; 1] = [Example {
    id: "coco-val2017",
    label: "Everyday scenes (COCO)",
    about: "Ordinary snapshots from Flickr: streets, kitchens, animals, sport, food, about 640 pixels wide. Made by the COCO project to teach computers to see; the photos are under Creative Commons licences.",
    photos: 5000,
    folder: "COCO 2017",
    url: "http://images.cocodataset.org/zips/val2017.zip", // no https there; the SHA-256 checks it
    size: 815_585_330,
    sha256: "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
}];

pub fn get(id: &str) -> Option<&'static Example> {
    ALL.iter().find(|e| e.id == id)
}

impl Example {
    pub fn dir(&self, root: &Path) -> PathBuf {
        root.join("examples").join(self.folder)
    }
}

/// Download the zip (most of the time: 90%), then take its photos out.
fn download(root: &Path, ex: &Example, stop: &AtomicBool, progress: &dyn Fn(f32)) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(std::time::Duration::from_secs(30))).timeout_recv_body(Some(std::time::Duration::from_secs(120))).build().into();
    let zip_path = root.join(".cache/examples").join(format!("{}.zip", ex.id));
    let remote = Remote { url: ex.url, size: ex.size, sha256: ex.sha256 };
    modelstore::fetch(&agent, &remote, &zip_path, stop, &|done| progress(0.9 * done as f32 / ex.size as f32))?;

    let dir = ex.dir(root);
    let part = dir.with_extension("part");
    let _ = std::fs::remove_dir_all(&part);
    std::fs::create_dir_all(&part).map_err(|e| format!("making {}: {e}", part.display()))?;
    let file = std::fs::File::open(&zip_path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("reading {}: {e}", zip_path.display()))?;
    let n = zip.len();
    for i in 0..n {
        if stop.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        // the photos only, by their own names, without the zip's folders (nothing can be written outside the folder)
        let Some(name) = entry.enclosed_name().and_then(|p| p.file_name().map(|n| n.to_owned())) else { continue };
        if entry.is_dir() || !crate::build::is_photo(Path::new(&name)) {
            continue;
        }
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        std::fs::write(part.join(&name), bytes).map_err(|e| format!("writing {}: {e}", part.join(&name).display()))?;
        progress(0.9 + 0.1 * (i + 1) as f32 / n as f32);
    }
    std::fs::rename(&part, &dir).map_err(|e| format!("moving {} into place: {e}", part.display()))?;
    let _ = std::fs::remove_file(&zip_path);
    Ok(())
}

pub struct Examples {
    root: PathBuf,
    pub job: Option<Arc<Job>>,
    pub failure: Option<String>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl Examples {
    pub fn new(root: PathBuf, wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self { root, job: None, failure: None, wake: Arc::new(wake) }
    }

    pub fn dir(&self, ex: &Example) -> PathBuf {
        ex.dir(&self.root)
    }

    pub fn installed(&self, ex: &Example) -> bool {
        self.dir(ex).is_dir()
    }

    /// Call every frame: takes in the end of a download.
    pub fn tick(&mut self) {
        if let Some(result) = self.job.as_ref().and_then(|j| j.state().1) {
            self.job = None;
            self.failure = match result {
                Err(e) if e != "stopped" => Some(e),
                _ => None,
            };
        }
    }

    pub fn get(&mut self, id: &str) {
        let Some(ex) = get(id) else { return };
        if self.job.is_some() || self.installed(ex) {
            return;
        }
        self.failure = None;
        let root = self.root.clone();
        self.job = Some(Job::start(ex.id, self.wake.clone(), move |stop, progress| download(&root, ex, stop, progress)));
    }

    pub fn stop(&self) {
        if let Some(job) = &self.job {
            job.stop();
        }
    }

    /// Delete its photos (and a half-made download).
    pub fn remove(&mut self, id: &str) {
        let Some(ex) = get(id) else { return };
        let dir = self.dir(ex);
        let _ = std::fs::remove_dir_all(dir.with_extension("part"));
        let _ = std::fs::remove_file(self.root.join(".cache/examples").join(format!("{}.zip", ex.id)));
        self.failure = std::fs::remove_dir_all(&dir).err().filter(|_| dir.exists()).map(|e| format!("removing {}: {e}", dir.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Takes the photos out of a small zip made here (no download: the zip is put where the download would be).
    #[test]
    fn photos_come_out_of_the_zip() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test").join(format!("pm-examples-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join(".cache/examples")).unwrap();
        let zip_path = root.join(".cache/examples/test.zip");
        {
            let mut w = zip::ZipWriter::new(std::fs::File::create(&zip_path).unwrap());
            let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
            w.add_directory("set/", opts).unwrap();
            for name in ["set/a.jpg", "set/b.png", "set/notes.txt", "../escape.jpg"] {
                w.start_file(name, opts).unwrap();
                std::io::Write::write_all(&mut w, name.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        let bytes = std::fs::read(&zip_path).unwrap();
        let hash: String = <sha2::Sha256 as sha2::Digest>::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect();
        let ex = Example { id: "test", label: "", about: "", photos: 2, folder: "Test", url: "http://localhost/none", size: bytes.len() as u64, sha256: Box::leak(hash.into_boxed_str()) };
        download(&root, &ex, &AtomicBool::new(false), &|_| {}).unwrap();
        let mut got: Vec<String> = std::fs::read_dir(ex.dir(&root)).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        got.sort();
        assert_eq!(got, ["a.jpg", "b.png"]);
        assert!(!zip_path.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Downloads the examples named in PHOTO_MAP_EXAMPLE (ids, comma-separated) into a test folder of this project, the way Get
    /// does, printing progress. PHOTO_MAP_EXAMPLE=coco-val2017 cargo test --release get_examples -- --ignored --nocapture
    #[test]
    #[ignore]
    fn get_examples() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/examples");
        for id in std::env::var("PHOTO_MAP_EXAMPLE").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
            let ex = get(id).unwrap();
            let last = std::sync::Mutex::new(-1i32);
            download(&root, ex, &AtomicBool::new(false), &|f| {
                let pct = (f * 100.0) as i32;
                if pct / 5 != *last.lock().unwrap() / 5 {
                    *last.lock().unwrap() = pct;
                    eprintln!("{id}: {pct}%");
                }
            })
            .unwrap();
            eprintln!("{id}: {} photos in {}", std::fs::read_dir(ex.dir(&root)).unwrap().count(), ex.dir(&root).display());
        }
    }
}
