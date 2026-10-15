//! Getting and removing the models (models.rs lists what exists). A model's files are downloaded from its publisher (a Hugging
//! Face repository at a fixed revision) into `.cache/onnx/<model>/`, in pieces, each file checked against its pinned size and
//! SHA-256 before it is put in place; a stopped download carries on where it was. Removing a model removes everything made for
//! it: its files, what was prepared from them (patch.rs), the networks CoreML and TensorRT made, and its vocabulary's vectors.
//! Maps already made with it stay, but can't be searched until it is got again.

use crate::models::{self, File, Spec};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

/// A model being fetched.
pub struct Job {
    pub id: &'static str,
    state: Mutex<(f32, Option<Result<(), String>>)>,
    stop: AtomicBool,
}

impl Job {
    /// How far it is (0 to 1), and how it ended (None while it runs).
    pub fn state(&self) -> (f32, Option<Result<(), String>>) {
        self.state.lock().unwrap().clone()
    }
}

/// What a model takes on disk, in bytes.
#[derive(Clone, Copy, Default)]
pub struct Taken {
    /// Its files (what a link points at counts).
    pub files: u64,
    /// What was prepared for it: networks for the Apple chip and the graphics card, the vocabulary's vectors.
    pub prepared: u64,
}

fn size_of(path: &Path) -> u64 {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.is_dir() => std::fs::read_dir(path).map_or(0, |entries| entries.flatten().map(|e| size_of(&e.path())).sum()),
        Ok(_) => std::fs::metadata(path).map_or(0, |m| m.len()), // follows a link
        Err(_) => 0,
    }
}

pub struct ModelStore {
    root: PathBuf,
    /// What each model takes (as last counted: counting walks folders, so it is done on a thread).
    pub taken: HashMap<&'static str, Taken>,
    counting: Option<Receiver<HashMap<&'static str, Taken>>>,
    pub job: Option<Arc<Job>>,
    pub failure: Option<String>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

/// What a model's files take to download, in MB.
pub fn size_mb(spec: &Spec) -> u32 {
    (spec.download_size() / 1_000_000) as u32
}

/// A piece of a file asked for at once (with its own retries).
const PIECE: u64 = 16 << 20;

/// Download one file of a model into `dir`, unless it is there already. `progress` gets the bytes of this file done so far.
pub fn fetch(agent: &ureq::Agent, spec: &Spec, file: &File, dir: &Path, stop: &AtomicBool, progress: &dyn Fn(u64)) -> Result<(), String> {
    let target = dir.join(file.path);
    if std::fs::metadata(&target).is_ok_and(|m| m.len() == file.size) {
        progress(file.size);
        return Ok(());
    }
    std::fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
    let part = target.with_file_name(format!("{}.part", target.file_name().unwrap().to_string_lossy()));
    // what an earlier, stopped download got is kept, and counted into the hash again
    let mut hasher = Sha256::new();
    let mut have = 0u64;
    if let Ok(mut f) = std::fs::File::open(&part) {
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = f.read(&mut buf).map_err(|e| e.to_string())?;
            if n == 0 || have + n as u64 > file.size {
                break;
            }
            hasher.update(&buf[..n]);
            have += n as u64;
        }
    }
    let mut out = std::fs::OpenOptions::new().create(true).write(true).truncate(false).open(&part).map_err(|e| format!("writing {}: {e}", part.display()))?;
    out.set_len(have).map_err(|e| e.to_string())?;
    std::io::Seek::seek(&mut out, std::io::SeekFrom::Start(have)).map_err(|e| e.to_string())?;
    progress(have);
    let url = format!("https://huggingface.co/{}/resolve/{}/{}", spec.repo, spec.revision, file.path);
    while have < file.size {
        if stop.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }
        let end = (have + PIECE).min(file.size) - 1;
        let mut last = String::new();
        let mut got = None;
        for attempt in 0..5u64 {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_secs(2 * attempt));
            }
            let piece = agent.get(&url).header("Range", format!("bytes={have}-{end}")).call().map_err(|e| e.to_string()).and_then(|r| {
                if r.status() != 206 {
                    return Err(format!("the server answered {} to a request for part of the file", r.status()));
                }
                let mut body = Vec::with_capacity((end - have + 1) as usize);
                r.into_body().into_reader().take(end - have + 1).read_to_end(&mut body).map_err(|e| e.to_string())?;
                if body.len() as u64 != end - have + 1 { Err(format!("got {} bytes of {}", body.len(), end - have + 1)) } else { Ok(body) }
            });
            match piece {
                Ok(body) => {
                    got = Some(body);
                    break;
                }
                Err(e) => last = e,
            }
        }
        let body = got.ok_or_else(|| format!("downloading {url}: {last}"))?;
        out.write_all(&body).map_err(|e| format!("writing {}: {e}", part.display()))?;
        hasher.update(&body);
        have += body.len() as u64;
        progress(have);
    }
    out.sync_all().map_err(|e| e.to_string())?;
    drop(out);
    let hash: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if hash != file.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(format!("{url} is not the file this version expects (SHA-256 {hash}); not using it"));
    }
    std::fs::rename(&part, &target).map_err(|e| format!("moving {} into place: {e}", part.display()))
}

/// Download all of a model's files. `progress` gets the fraction done (0 to 1).
fn download(root: &Path, spec: &Spec, stop: &AtomicBool, progress: &dyn Fn(f32)) -> Result<(), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(std::time::Duration::from_secs(30))).timeout_recv_body(Some(std::time::Duration::from_secs(120))).build().into();
    let dir = spec.dir(root);
    let total = spec.download_size().max(1) as f64;
    let mut before = 0u64;
    for file in spec.files {
        fetch(&agent, spec, file, &dir, stop, &|done| progress(((before + done) as f64 / total) as f32))?;
        before += file.size;
    }
    Ok(())
}

/// Everything made for a model besides its files, and its files: what `remove` takes away. (Links are removed, never what they
/// point at.)
fn leftovers(root: &Path, spec: &Spec) -> Vec<PathBuf> {
    let mut found = vec![root.join(".cache/onnx").join(spec.folder), root.join(".cache/coreml").join(spec.folder), root.join(".cache/trt").join(spec.folder)];
    // the vocabulary's vectors for this model
    let slug = crate::naming::slug(spec.id);
    if let Ok(entries) = std::fs::read_dir(root.join(".cache/vocab")) {
        found.extend(entries.flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().contains(&format!(".{slug}.")))));
    }
    if spec.id == models::GEMMA {
        // TensorRT's engines for Gemma are files in .cache/trt itself (the SigLIP ones are in folders of their own)
        if let Ok(entries) = std::fs::read_dir(root.join(".cache/trt")) {
            found.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_file()));
        }
    }
    found.sort();
    found.dedup();
    found
}

impl ModelStore {
    pub fn new(root: PathBuf, wake: impl Fn() + Send + Sync + 'static) -> Self {
        let mut store = Self { root, taken: HashMap::new(), counting: None, job: None, failure: None, wake: Arc::new(wake) };
        store.count();
        store
    }

    /// Count again what each model takes, on a thread.
    pub fn count(&mut self) {
        let (tx, rx) = channel();
        let (root, wake) = (self.root.clone(), self.wake.clone());
        std::thread::spawn(move || {
            let counted = models::ALL
                .iter()
                .map(|spec| {
                    // the downloaded files; everything else in its folder (patch.rs) and elsewhere was made from them
                    let files: u64 = spec.files.iter().filter_map(|f| std::fs::metadata(spec.dir(&root).join(f.path)).ok()).map(|m| m.len()).sum();
                    let prepared = leftovers(&root, spec).iter().map(|p| size_of(p)).sum::<u64>().saturating_sub(files);
                    (spec.id, Taken { files, prepared })
                })
                .collect();
            let _ = tx.send(counted);
            wake();
        });
        self.counting = Some(rx);
    }

    pub fn installed(&self, spec: &Spec) -> bool {
        spec.installed(&self.root)
    }

    pub fn fetching(&self) -> bool {
        self.job.is_some()
    }

    /// Call every frame: takes in the end of a fetch.
    pub fn tick(&mut self) {
        if let Some(counted) = self.counting.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.taken = counted;
            self.counting = None;
        }
        let finished = self.job.as_ref().and_then(|j| j.state().1);
        if let Some(result) = finished {
            self.job = None;
            self.count();
            self.failure = match result {
                Err(e) if e != "stopped" => Some(e),
                _ => None,
            };
        }
    }

    pub fn get(&mut self, id: &str) {
        let Some(spec) = models::spec(id) else { return };
        if self.job.is_some() || self.installed(spec) {
            return;
        }
        let job = Arc::new(Job { id: spec.id, state: Mutex::new((0.0, None)), stop: AtomicBool::new(false) });
        self.job = Some(job.clone());
        self.failure = None;
        let (root, wake) = (self.root.clone(), self.wake.clone());
        std::thread::spawn(move || {
            let result = download(&root, spec, &job.stop, &|fraction| {
                job.state.lock().unwrap().0 = fraction;
                wake();
            });
            job.state.lock().unwrap().1 = Some(result);
            wake();
        });
    }

    pub fn stop(&self) {
        if let Some(job) = &self.job {
            job.stop.store(true, Ordering::Relaxed);
        }
    }

    pub fn remove(&mut self, id: &str) {
        let Some(spec) = models::spec(id) else { return };
        for path in leftovers(&self.root, spec) {
            let result = match std::fs::symlink_metadata(&path) {
                Ok(m) if m.is_dir() => std::fs::remove_dir_all(&path),
                Ok(_) => std::fs::remove_file(&path),
                Err(_) => Ok(()),
            };
            if let Err(e) = result {
                self.failure = Some(format!("removing {}: {e}", path.display()));
                return;
            }
        }
        self.failure = None;
        self.count();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_model_takes_what_was_made_for_it_and_leaves_the_others() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test").join(format!("pm-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let spec = models::spec(models::SIGLIP_BASE_256).unwrap();
        let other = models::spec(models::SIGLIP_BASE_512).unwrap();
        for s in [spec, other] {
            for f in s.files {
                let path = s.dir(&root).join(f.path);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::File::create(&path).unwrap().set_len(f.size).unwrap(); // sparse: takes no room
            }
        }
        assert!(spec.installed(&root) && other.installed(&root));
        let slug = crate::naming::slug(spec.id);
        std::fs::create_dir_all(root.join(".cache/coreml").join(spec.folder)).unwrap();
        std::fs::write(root.join(".cache/coreml").join(spec.folder).join("net"), b"x").unwrap();
        std::fs::create_dir_all(root.join(".cache/vocab")).unwrap();
        std::fs::write(root.join(".cache/vocab").join(format!("curated2.{slug}.f16")), b"x").unwrap();
        std::fs::write(root.join(".cache/vocab/curated2.other.f16"), b"x").unwrap();

        let mut store = ModelStore::new(root.clone(), || {});
        store.remove(spec.id);
        assert!(store.failure.is_none());
        assert!(!spec.installed(&root) && other.installed(&root));
        assert!(!root.join(".cache/coreml").join(spec.folder).exists());
        assert!(!root.join(".cache/vocab").join(format!("curated2.{slug}.f16")).exists());
        assert!(root.join(".cache/vocab/curated2.other.f16").exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Downloads SigLIP 256's tokenizer (34 MB), stops half way, and carries on. cargo test --release modelstore -- --ignored
    #[test]
    #[ignore]
    fn a_stopped_download_carries_on_and_is_checked() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test").join(format!("pm-fetch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let spec = models::spec(models::SIGLIP_BASE_256).unwrap();
        let file = &spec.files[0];
        let agent = ureq::Agent::new_with_defaults();
        let stop = AtomicBool::new(false);
        let err = fetch(&agent, spec, file, &spec.dir(&root), &stop, &|done| if done > file.size / 2 { stop.store(true, Ordering::Relaxed) }).unwrap_err();
        assert_eq!(err, "stopped");
        stop.store(false, Ordering::Relaxed);
        let first = std::sync::Mutex::new(None);
        fetch(&agent, spec, file, &spec.dir(&root), &stop, &|done| { first.lock().unwrap().get_or_insert(done); }).unwrap();
        assert!(first.lock().unwrap().unwrap() > file.size / 2, "it started again from the beginning");
        assert_eq!(std::fs::metadata(spec.dir(&root).join(file.path)).unwrap().len(), file.size);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Downloads the models named in PHOTO_MAP_GET (ids, comma-separated) into this project, as Get does, printing progress.
    /// PHOTO_MAP_GET=google/siglip2-base-patch16-256 cargo test --release get_models -- --ignored --nocapture
    #[test]
    #[ignore]
    fn get_models() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        for id in std::env::var("PHOTO_MAP_GET").unwrap_or_default().split(',').filter(|s| !s.is_empty()) {
            let spec = models::spec(id).unwrap();
            let last = Mutex::new(-1i32);
            download(root, spec, &AtomicBool::new(false), &|f| {
                let pct = (f * 100.0) as i32;
                if pct / 5 != *last.lock().unwrap() / 5 {
                    *last.lock().unwrap() = pct;
                    eprintln!("{id}: {pct}%");
                }
            })
            .unwrap();
            eprintln!("{id}: done");
        }
    }
}
