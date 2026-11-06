//! What the models may run on: the choice made in the Runtime window, kept in `.cache/backends` (one name a line).
//!
//! One of them is chosen; the processor always works. Nothing chosen yet means the best one available (see `PREFERRED`).

use crate::runtime::Setup;
use ort::ep::coreml::ComputeUnits;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Cpu,
    /// CoreML, free to use the processor, the graphics chip and the neural engine.
    AppleAll,
    AppleGpu,
    AppleNeural,
    Cuda,
    TensorRt,
}

impl Backend {
    pub const ALL: [Backend; 6] = [Backend::Cpu, Backend::AppleNeural, Backend::AppleGpu, Backend::AppleAll, Backend::Cuda, Backend::TensorRt];

    fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::AppleAll => "apple-all",
            Backend::AppleGpu => "apple-gpu",
            Backend::AppleNeural => "apple-neural",
            Backend::Cuda => "cuda",
            Backend::TensorRt => "tensorrt",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cpu => "Processor",
            Backend::AppleAll => "Apple chip, all parts",
            Backend::AppleGpu => "Apple chip, graphics",
            Backend::AppleNeural => "Apple chip, neural engine",
            Backend::Cuda => "NVIDIA graphics card (CUDA)",
            Backend::TensorRt => "NVIDIA graphics card (TensorRT)",
        }
    }

    pub fn note(self) -> &'static str {
        match self {
            Backend::Cpu => "Works everywhere, slowest.",
            Backend::AppleAll => "CoreML picks the part for each piece of the model.",
            Backend::AppleGpu => "CoreML on the graphics chip: as fast as the neural engine, but busy while the map is shown.",
            Backend::AppleNeural => "CoreML on the neural engine: fast, and the graphics chip stays free. Applies to the photo encoders.",
            Backend::Cuda => "Needs CUDA and cuDNN.",
            Backend::TensorRt => "Faster than CUDA; slow to start the first time a model is used.",
        }
    }

    /// Does this machine have such a part at all (what is not there is not shown)?
    pub fn exists_here(self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::AppleAll | Backend::AppleGpu | Backend::AppleNeural => cfg!(target_os = "macos"),
            Backend::Cuda | Backend::TensorRt => !cfg!(target_os = "macos"),
        }
    }

    /// Is what it needs installed?
    pub fn available(self, setup: &Setup) -> bool {
        match self {
            Backend::Cpu | Backend::AppleAll | Backend::AppleGpu | Backend::AppleNeural => setup.ort.is_some(),
            Backend::Cuda => setup.gpu.is_some() && setup.ort_cuda,
            Backend::TensorRt => setup.gpu.is_some() && setup.ort_cuda && setup.ort_tensorrt && setup.tensorrt.is_some(),
        }
    }
}

/// The one backend chosen. `None`: never chosen, so the best one available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice(Option<Backend>);

fn file(root: &Path) -> PathBuf {
    root.join(".cache/backends")
}

/// The default, best first (on a Mac the neural engine, which was the fastest and leaves the graphics chip to the viewer).
const PREFERRED: [Backend; 4] = [Backend::TensorRt, Backend::Cuda, Backend::AppleNeural, Backend::Cpu];

impl Choice {
    pub fn load(root: &Path) -> Self {
        // for benchmarks: PHOTO_MAP_COREML = off, cpu, gpu, ane or all
        if cfg!(target_os = "macos") {
            match std::env::var("PHOTO_MAP_COREML").as_deref() {
                Ok("off") | Ok("cpu") => return Choice(Some(Backend::Cpu)),
                Ok("gpu") => return Choice(Some(Backend::AppleGpu)),
                Ok("ane") => return Choice(Some(Backend::AppleNeural)),
                Ok("all") => return Choice(Some(Backend::AppleAll)),
                _ => {}
            }
        }
        // for tests and benchmarks: PHOTO_MAP_BACKEND = cpu, cuda, tensorrt, apple-neural, apple-gpu or apple-all
        if let Some(b) = std::env::var("PHOTO_MAP_BACKEND").ok().and_then(|v| Backend::ALL.into_iter().find(|b| b.name() == v)) {
            return Choice(Some(b));
        }
        // the file once held a list ("cuda" and "tensorrt" together): the last of Backend::ALL in it is the one
        match std::fs::read_to_string(file(root)) {
            Ok(text) => Choice(Backend::ALL.into_iter().rev().find(|b| text.lines().any(|l| l.trim() == b.name()))),
            Err(_) => Choice(None),
        }
    }

    pub fn save(&self, root: &Path) {
        if let Some(b) = self.0 {
            let _ = std::fs::create_dir_all(root.join(".cache"));
            let _ = std::fs::write(file(root), format!("{}\n", b.name()));
        }
    }

    /// The backend in use: the one chosen if it is available here, else the best one that is.
    pub fn selected(&self, setup: &Setup) -> Backend {
        let usable = |b: &Backend| b.exists_here() && b.available(setup);
        self.0.filter(usable).or_else(|| PREFERRED.into_iter().find(usable)).unwrap_or(Backend::Cpu)
    }

    /// Is this the one in use (never one that is not available)?
    pub fn is_on(&self, b: Backend, setup: &Setup) -> bool {
        b.exists_here() && b.available(setup) && self.selected(setup) == b
    }

    pub fn set(&mut self, b: Backend) {
        self.0 = Some(b);
    }

    /// TensorRT runs on top of CUDA: whatever it can't take runs there.
    pub fn cuda(&self, setup: &Setup) -> bool {
        self.is_on(Backend::Cuda, setup) || self.is_on(Backend::TensorRt, setup)
    }

    pub fn tensorrt(&self, setup: &Setup) -> bool {
        self.is_on(Backend::TensorRt, setup)
    }

    pub fn apple(&self, setup: &Setup) -> Option<ComputeUnits> {
        match self.selected(setup) {
            b if !self.is_on(b, setup) => None,
            Backend::AppleNeural => Some(ComputeUnits::CPUAndNeuralEngine),
            Backend::AppleGpu => Some(ComputeUnits::CPUAndGPU),
            Backend::AppleAll => Some(ComputeUnits::All),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_os = "macos"))]
    use crate::runtime::Gpu;
    use crate::runtime::{Place, Source};

    fn place() -> Place {
        Place { dir: PathBuf::from("/x"), source: Source::Project }
    }

    fn with_ort() -> Setup {
        Setup { ort: Some(place()), ..Setup::default() }
    }

    #[test]
    fn nothing_is_in_use_without_the_runtime() {
        let none = Choice(None);
        assert!(Backend::ALL.iter().all(|b| !none.is_on(*b, &Setup::default())));
    }

    #[test]
    fn one_at_a_time_and_the_best_available_when_the_choice_is_not_there() {
        let setup = with_ort();
        let mut c = Choice(None);
        c.set(Backend::Cuda);
        // no CUDA here: the processor, or on a Mac the neural engine
        let fallback = if cfg!(target_os = "macos") { Backend::AppleNeural } else { Backend::Cpu };
        assert!(c.is_on(fallback, &setup) && !c.cuda(&setup));
        c.set(Backend::Cpu);
        assert_eq!(Backend::ALL.iter().filter(|b| c.is_on(**b, &setup)).count(), 1);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn tensorrt_is_one_choice_and_uses_cuda_underneath() {
        let setup = Setup { ort: Some(place()), ort_cuda: true, ort_tensorrt: true, gpu: Some(Gpu { cuda: place(), cudnn: place() }), tensorrt: Some(place()) };
        let mut c = Choice(None);
        assert!(c.is_on(Backend::TensorRt, &setup) && !c.is_on(Backend::Cuda, &setup) && c.cuda(&setup));
        c.set(Backend::Cuda);
        assert!(c.is_on(Backend::Cuda, &setup) && !c.tensorrt(&setup));
        let no_trt = Setup { tensorrt: None, ..setup };
        assert!(Choice(Some(Backend::TensorRt)).is_on(Backend::Cuda, &no_trt));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn one_part_of_the_apple_chip_at_a_time() {
        let setup = with_ort();
        let mut c = Choice(None);
        assert!(c.is_on(Backend::AppleNeural, &setup) && !c.is_on(Backend::Cpu, &setup));
        c.set(Backend::AppleGpu);
        assert!(c.is_on(Backend::AppleGpu, &setup) && !c.is_on(Backend::AppleNeural, &setup));
        c.set(Backend::Cpu);
        assert!(c.is_on(Backend::Cpu, &setup) && c.apple(&setup).is_none());
    }
}
