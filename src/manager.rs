//! What the interface knows about the runtime libraries: what is installed (runtime.rs), what could be installed (packs.rs), and
//! the install under way. Nothing is installed without the user pressing the button, and only what this program installed can be
//! removed from here.

use crate::packs::{self, Pack};
use crate::runtime::{Nvidia, Place, Setup, Source};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

pub struct Offer {
    /// The packs still missing for it.
    pub packs: Vec<&'static Pack>,
    pub title: &'static str,
    pub why: String,
}

impl Offer {
    pub fn mb(&self) -> u32 {
        self.packs.iter().map(|p| p.size_mb).sum()
    }
}

pub enum Advice {
    Nothing,
    Offer(Offer),
    /// The card's driver is too old for the CUDA libraries: nothing to download, the user has to update it.
    Driver(String),
}

pub struct Job {
    pub packs: Vec<&'static Pack>,
    state: Mutex<JobState>,
    stop: AtomicBool,
}

#[derive(Clone)]
pub struct JobState {
    /// The pack being fetched.
    pub current: &'static str,
    pub fraction: f32,
    pub finished: Option<Result<(), String>>,
}

impl Job {
    pub fn state(&self) -> JobState {
        self.state.lock().unwrap().clone()
    }
}

pub struct Manager {
    root: PathBuf,
    pub setup: Setup,
    /// The NVIDIA card, once the driver has answered (it can take a moment): None if there is none.
    pub nvidia: Option<Nvidia>,
    probe: Option<Receiver<Option<Nvidia>>>,
    /// Other versions of the libraries found (what a "not installed" is not for lack of looking).
    pub wrong: Vec<String>,
    pub job: Option<Arc<Job>>,
    /// An install finished: this copy of the program still runs on what it started with.
    pub restart: bool,
    pub failure: Option<String>,
    dismissed: bool,
    wake: Arc<dyn Fn() + Send + Sync>,
}

fn dismissed_file(root: &Path) -> PathBuf {
    root.join(".cache/runtime-offer-dismissed")
}

impl Manager {
    pub fn new(root: PathBuf, wake: impl Fn() + Send + Sync + 'static) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(Nvidia::probe());
        });
        let setup = Setup::detect(&root);
        let wrong = Setup::wrong_versions(&root);
        let dismissed = dismissed_file(&root).exists();
        Self { root, setup, nvidia: None, probe: Some(rx), wrong, job: None, restart: false, failure: None, dismissed, wake: Arc::new(wake) }
    }

    /// Look again at what is installed.
    pub fn refresh(&mut self) {
        self.setup = Setup::detect(&self.root);
        self.wrong = Setup::wrong_versions(&self.root);
    }

    /// Call every frame: takes in the card's answer and the end of an install.
    pub fn tick(&mut self) {
        if let Some(rx) = &self.probe {
            if let Ok(card) = rx.try_recv() {
                self.nvidia = card;
                self.probe = None;
            }
        }
        let finished = self.job.as_ref().and_then(|j| j.state().finished);
        if let Some(result) = finished {
            self.job = None;
            self.refresh();
            match result {
                Ok(()) => {
                    self.restart = true;
                    self.failure = None;
                }
                Err(e) if e == "stopped" => self.failure = None,
                Err(e) => self.failure = Some(e),
            }
        }
    }

    pub fn card_known(&self) -> bool {
        self.probe.is_none()
    }

    pub fn installing(&self) -> bool {
        self.job.is_some()
    }

    /// What to suggest, given what is installed and what the machine has. Never anything for a machine with no usable card, except
    /// the one thing nothing runs without.
    pub fn advice(&self) -> Advice {
        if self.restart || !self.card_known() {
            return Advice::Nothing;
        }
        let usable_card = self.nvidia.as_ref().filter(|c| c.driver_ok());
        if self.setup.ort.is_none() {
            return match usable_card {
                Some(card) => Advice::Offer(Offer {
                    packs: vec![&packs::ONNXRUNTIME_GPU, &packs::CUDA, &packs::CUDNN],
                    title: "Download GPU acceleration",
                    why: format!("The models need ONNX Runtime, which is not installed. {} was found; this installs the version that uses it, with the NVIDIA libraries it needs.", card.name),
                }),
                None => Advice::Offer(Offer {
                    packs: vec![packs::onnxruntime_cpu()],
                    title: "Download ONNX Runtime",
                    why: if cfg!(target_os = "macos") {
                        "The models need ONNX Runtime, which is not installed. This installs the version for this Mac: it runs on the processor and on the Apple chip.".into()
                    } else {
                        "The models need ONNX Runtime, which is not installed. This installs the version that runs on the processor.".into()
                    },
                }),
            };
        }
        if let Some(card) = &self.nvidia {
            if !card.driver_ok() {
                return Advice::Driver(format!(
                    "{} was found, but its driver only supports CUDA {} and CUDA 13 is needed to use it. Update the NVIDIA driver; until then the models run on the processor.",
                    card.name,
                    cuda_version(card.driver_cuda)
                ));
            }
            if self.setup.gpu.is_none() || !self.setup.ort_cuda {
                let mut missing = Vec::new();
                if !self.setup.ort_cuda {
                    missing.push(&packs::ONNXRUNTIME_GPU);
                }
                if self.setup.gpu.is_none() {
                    missing.push(&packs::CUDA);
                    missing.push(&packs::CUDNN);
                }
                return Advice::Offer(Offer {
                    packs: missing,
                    title: "Download GPU acceleration",
                    why: format!("{} was found, but the libraries to run the models on it are not installed, so they run on the processor.", card.name),
                });
            }
            if self.setup.tensorrt.is_none() && self.setup.ort_tensorrt {
                return Advice::Offer(Offer {
                    packs: vec![&packs::TENSORRT],
                    title: "Download TensorRT",
                    why: "TensorRT builds the photo encoder for your card (the first time a model is used, once). The models work without it.".into(),
                });
            }
        }
        Advice::Nothing
    }

    /// Whether the offer should be shown where it can be dismissed (the New map form).
    pub fn nagging(&self) -> bool {
        !self.dismissed
    }

    pub fn dismiss(&mut self) {
        self.dismissed = true;
        let _ = std::fs::create_dir_all(self.root.join(".cache"));
        let _ = std::fs::write(dismissed_file(&self.root), b"");
    }

    /// Fetch and install the packs of an offer, one after the other, on a thread.
    pub fn start(&mut self, offer: &Offer) {
        if self.job.is_some() {
            return;
        }
        let job = Arc::new(Job {
            packs: offer.packs.clone(),
            state: Mutex::new(JobState { current: offer.packs.first().map_or("", |p| p.label), fraction: 0.0, finished: None }),
            stop: AtomicBool::new(false),
        });
        self.job = Some(job.clone());
        self.failure = None;
        let (root, card, wake) = (self.root.clone(), self.nvidia.clone(), self.wake.clone());
        std::thread::spawn(move || {
            let total: f32 = job.packs.iter().map(|p| p.size_mb as f32).sum::<f32>().max(1.0);
            let mut before = 0.0f32;
            let mut outcome = Ok(());
            for pack in &job.packs {
                let weight = pack.size_mb as f32;
                job.state.lock().unwrap().current = pack.label;
                let result = packs::install(&root, pack, card.as_ref(), &job.stop, &|done, all| {
                    let part = if all == 0 { 0.0 } else { done as f32 / all as f32 };
                    job.state.lock().unwrap().fraction = (before + part * weight) / total;
                    wake();
                });
                if let Err(e) = result {
                    outcome = Err(e);
                    break;
                }
                before += weight;
            }
            job.state.lock().unwrap().finished = Some(outcome);
            wake();
        });
    }

    pub fn stop(&mut self) {
        if let Some(job) = &self.job {
            job.stop.store(true, Ordering::Relaxed);
        }
    }

    /// Remove a pack this program installed.
    pub fn remove(&mut self, id: &str) {
        match packs::remove(&self.root, id) {
            Ok(()) => {
                self.failure = None;
                self.restart = true; // what is loaded stays loaded until the program is restarted
            }
            Err(e) => self.failure = Some(e),
        }
        self.refresh();
    }

    pub fn installed(&self) -> Vec<packs::Installed> {
        packs::installed(&self.root)
    }
}

/// Where a set of libraries is from, in words.
pub fn describe(place: &Place) -> String {
    match &place.source {
        Source::Managed(_) => format!("installed by photo-map ({})", place.dir.display()),
        Source::Project => format!("found in {} (not installed by photo-map)", place.dir.display()),
        Source::Override => format!("from PHOTO_MAP_RUNTIME ({})", place.dir.display()),
        Source::System => format!("found on this computer ({})", place.dir.display()),
    }
}

/// The CUDA version a driver reports (13000 is "13.0").
pub fn cuda_version(n: u32) -> String {
    format!("{}.{}", n / 1000, n % 1000 / 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Gpu;

    fn manager(setup: Setup, nvidia: Option<Nvidia>) -> Manager {
        Manager { root: PathBuf::from("/nonexistent"), setup, nvidia, probe: None, wrong: Vec::new(), job: None, restart: false, failure: None, dismissed: false, wake: Arc::new(|| {}) }
    }

    fn place(name: &str) -> Place {
        Place { dir: PathBuf::from(name), source: Source::Project }
    }

    fn card(driver_cuda: u32) -> Nvidia {
        Nvidia { name: "Test card".into(), compute: 86, driver_cuda }
    }

    fn ids(advice: Advice) -> Vec<&'static str> {
        match advice {
            Advice::Offer(o) => o.packs.iter().map(|p| p.id).collect(),
            Advice::Driver(_) => vec!["driver"],
            Advice::Nothing => vec![],
        }
    }

    fn gpu_setup(tensorrt: bool) -> Setup {
        Setup { ort: Some(place("ort")), ort_cuda: true, ort_tensorrt: true, gpu: Some(Gpu { cuda: place("cuda"), cudnn: place("cudnn") }), tensorrt: tensorrt.then(|| place("trt")) }
    }

    #[test]
    fn nothing_runs_without_onnx_runtime_so_that_is_offered_first() {
        assert_eq!(ids(manager(Setup::default(), None).advice()), [packs::onnxruntime_cpu().id]);
        assert_eq!(ids(manager(Setup::default(), Some(card(13000))).advice()), ["onnxruntime-gpu", "cuda-13", "cudnn-9"]);
    }

    #[test]
    fn a_card_that_is_not_used_gets_the_pieces_that_are_missing() {
        let cpu_only = Setup { ort: Some(place("ort")), ..Setup::default() };
        assert_eq!(ids(manager(cpu_only, Some(card(13000))).advice()), ["onnxruntime-gpu", "cuda-13", "cudnn-9"]);
        let no_cuda = Setup { gpu: None, ..gpu_setup(false) };
        assert_eq!(ids(manager(no_cuda, Some(card(13000))).advice()), ["cuda-13", "cudnn-9"]);
    }

    #[test]
    fn an_old_driver_is_told_to_update_not_offered_a_download() {
        let cpu_only = Setup { ort: Some(place("ort")), ..Setup::default() };
        assert_eq!(ids(manager(cpu_only, Some(card(12040))).advice()), ["driver"]);
    }

    #[test]
    fn tensorrt_is_offered_only_when_the_rest_works_and_it_is_missing() {
        assert_eq!(ids(manager(gpu_setup(false), Some(card(13000))).advice()), ["tensorrt-10"]);
        assert!(ids(manager(gpu_setup(true), Some(card(13000))).advice()).is_empty());
    }

    #[test]
    fn a_computer_without_a_card_is_not_offered_anything_once_it_runs() {
        let cpu_only = Setup { ort: Some(place("ort")), ..Setup::default() };
        assert!(ids(manager(cpu_only, None).advice()).is_empty());
    }

    #[test]
    fn nothing_is_suggested_before_the_driver_has_answered_or_after_an_install() {
        let mut waiting = manager(Setup::default(), None);
        let (_tx, rx) = channel();
        waiting.probe = Some(rx);
        assert!(ids(waiting.advice()).is_empty());
        let mut done = manager(Setup::default(), None);
        done.restart = true;
        assert!(ids(done.advice()).is_empty());
    }
}
