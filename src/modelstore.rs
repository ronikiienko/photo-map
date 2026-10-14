//! Getting and removing the models (models.rs lists what exists). Getting is simulated for now: it takes a few seconds, then
//! links the files from a sample folder (`.cache/upstream/`) into `.cache/onnx/<model>/`, where the app looks for them.
//! Removing a model removes everything made for it: its files, the networks CoreML and TensorRT prepared for it, and its
//! vocabulary's vectors. Maps already made with it stay, but can't be searched until it is got again.

use crate::models::{self, Spec};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
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

pub struct ModelStore {
    root: PathBuf,
    pub job: Option<Arc<Job>>,
    pub failure: Option<String>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

/// About how much a model's files take, in MB.
pub fn size_mb(spec: &Spec) -> u32 {
    match spec.id {
        models::GEMMA => 840,
        _ => 720,
    }
}

/// Where the sample files of a model are, and what each is called in the model's folder: (name in the folder, path under `.cache/upstream/`).
fn sample(spec: &Spec) -> &'static [(&'static str, &'static str)] {
    match spec.id {
        models::GEMMA => &[
            ("tokenizer.json", "gemma2/tokenizer.json"),
            ("onnx/model_fp16.onnx", "gemma2/onnx/model_fp16.onnx"),
            ("onnx/model_fp16.onnx_data", "gemma2/onnx/model_fp16.onnx_data"),
            ("onnx/vision_encoder_fp16.onnx", "gemma2/onnx/vision_encoder_fp16.onnx"),
            ("onnx/vision_encoder_fp16.onnx_data", "gemma2/onnx/vision_encoder_fp16.onnx_data"),
        ],
        models::SIGLIP_BASE_512 => &[
            ("tokenizer.json", "siglip2-512/tokenizer.json"),
            ("text_fp16.onnx", "siglip2-512/onnx/text_model.onnx"),
            ("vision_fp16.onnx", "siglip2-512/onnx/vision_model_padded.onnx"),
        ],
        _ => &[
            ("tokenizer.json", "siglip2-256/tokenizer.json"),
            ("text_fp16.onnx", "siglip2-256/onnx/text_model_fp16.onnx"),
            ("vision_fp16.onnx", "siglip2-256/onnx/vision_model_fp16_padded.onnx"),
        ],
    }
}

fn install(root: &Path, spec: &Spec) -> Result<(), String> {
    let dir = root.join(".cache/onnx").join(spec.folder);
    for (name, from) in sample(spec) {
        let source = root.join(".cache/upstream").join(from);
        if !source.exists() {
            return Err(format!("simulated fetch: the sample file {} is not there", source.display()));
        }
        let target = dir.join(name);
        std::fs::create_dir_all(target.parent().unwrap()).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&target);
        #[cfg(unix)]
        std::os::unix::fs::symlink(&source, &target).map_err(|e| e.to_string())?;
        #[cfg(not(unix))]
        std::fs::copy(&source, &target).map_err(|e| e.to_string())?;
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
        Self { root, job: None, failure: None, wake: Arc::new(wake) }
    }

    pub fn installed(&self, spec: &Spec) -> bool {
        spec.installed(&self.root)
    }

    pub fn fetching(&self) -> bool {
        self.job.is_some()
    }

    /// Call every frame: takes in the end of a fetch.
    pub fn tick(&mut self) {
        let finished = self.job.as_ref().and_then(|j| j.state().1);
        if let Some(result) = finished {
            self.job = None;
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
            // simulated: the time a download of about this size takes on a good line
            let steps = (size_mb(spec) / 20).max(1);
            for i in 0..steps {
                if job.stop.load(Ordering::Relaxed) {
                    job.state.lock().unwrap().1 = Some(Err("stopped".into()));
                    wake();
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
                job.state.lock().unwrap().0 = (i + 1) as f32 / steps as f32;
                wake();
            }
            let result = install(&root, spec);
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_model_takes_what_was_made_for_it_and_leaves_the_sample_files() {
        let root = std::env::temp_dir().join(format!("pm-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let spec = models::spec(models::SIGLIP_BASE_256).unwrap();
        for (_, from) in sample(spec) {
            let path = root.join(".cache/upstream").join(from);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"x").unwrap();
        }
        assert!(!spec.installed(&root));
        install(&root, spec).unwrap();
        assert!(spec.installed(&root));
        let slug = crate::naming::slug(spec.id);
        std::fs::create_dir_all(root.join(".cache/coreml").join(spec.folder)).unwrap();
        std::fs::write(root.join(".cache/coreml").join(spec.folder).join("net"), b"x").unwrap();
        std::fs::create_dir_all(root.join(".cache/vocab")).unwrap();
        std::fs::write(root.join(".cache/vocab").join(format!("curated2.{slug}.f16")), b"x").unwrap();
        std::fs::write(root.join(".cache/vocab/curated2.other.f16"), b"x").unwrap();

        let mut store = ModelStore::new(root.clone(), || {});
        store.remove(spec.id);
        assert!(store.failure.is_none());
        assert!(!spec.installed(&root));
        assert!(!root.join(".cache/coreml").join(spec.folder).exists());
        assert!(!root.join(".cache/vocab").join(format!("curated2.{slug}.f16")).exists());
        assert!(root.join(".cache/vocab/curated2.other.f16").exists());
        assert!(root.join(".cache/upstream/siglip2-256/tokenizer.json").exists());
        let _ = std::fs::remove_dir_all(&root);
    }
}
