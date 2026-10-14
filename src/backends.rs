//! What the models may run on: the choice made in the Runtime window, kept in `.cache/backends` (one name a line).
//!
//! The processor always works; the rest is a choice among what the machine has and what is installed. Nothing chosen yet means
//! everything that is available (on a Mac the neural engine, which was the fastest and leaves the graphics chip to the viewer).

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
            Backend::AppleNeural => "CoreML on the neural engine: fast, and the graphics chip stays free. Photo encoders only.",
            Backend::Cuda => "Needs CUDA and cuDNN.",
            Backend::TensorRt => "On top of CUDA; faster, and slow to start the first time.",
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

    fn apple(self) -> bool {
        matches!(self, Backend::AppleAll | Backend::AppleGpu | Backend::AppleNeural)
    }
}

/// The backends ticked. `None`: never chosen, so the default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice(Option<Vec<Backend>>);

fn file(root: &Path) -> PathBuf {
    root.join(".cache/backends")
}

impl Choice {
    pub fn load(root: &Path) -> Self {
        // for benchmarks: PHOTO_MAP_COREML = off, cpu, gpu, ane or all
        if cfg!(target_os = "macos") {
            let one = |b| Choice(Some(vec![b]));
            match std::env::var("PHOTO_MAP_COREML").as_deref() {
                Ok("off") | Ok("cpu") => return one(Backend::Cpu),
                Ok("gpu") => return one(Backend::AppleGpu),
                Ok("ane") => return one(Backend::AppleNeural),
                Ok("all") => return one(Backend::AppleAll),
                _ => {}
            }
        }
        match std::fs::read_to_string(file(root)) {
            Ok(text) => Choice(Some(text.lines().filter_map(|l| Backend::ALL.into_iter().find(|b| b.name() == l.trim())).collect())),
            Err(_) => Choice(None),
        }
    }

    pub fn save(&self, root: &Path) {
        if let Some(list) = &self.0 {
            let _ = std::fs::create_dir_all(root.join(".cache"));
            let _ = std::fs::write(file(root), list.iter().map(|b| format!("{}\n", b.name())).collect::<String>());
        }
    }

    /// What is ticked, given what is available (an unavailable one is never in use, whatever was chosen).
    pub fn is_on(&self, b: Backend, setup: &Setup) -> bool {
        if !b.exists_here() || !b.available(setup) {
            return false;
        }
        match &self.0 {
            Some(list) => list.contains(&b) || (b == Backend::Cpu && !list.iter().any(|x| x.available(setup) && *x != Backend::Cpu)),
            None => b == Backend::AppleNeural || b == Backend::Cuda || b == Backend::TensorRt || b == Backend::Cpu && !Backend::ALL.iter().any(|x| *x != Backend::Cpu && x.exists_here() && x.available(setup)),
        }
    }

    fn current(&self, setup: &Setup) -> Vec<Backend> {
        Backend::ALL.into_iter().filter(|b| self.is_on(*b, setup)).collect()
    }

    /// Tick or untick one. Only one part of the Apple chip is used at a time, and TensorRT works on top of CUDA.
    pub fn set(&mut self, b: Backend, on: bool, setup: &Setup) {
        let mut list = self.current(setup);
        list.retain(|x| *x != b);
        if on {
            if b.apple() {
                list.retain(|x| !x.apple());
            }
            if b == Backend::TensorRt {
                list.push(Backend::Cuda);
            }
            list.push(b);
        } else if b == Backend::Cuda {
            list.retain(|x| *x != Backend::TensorRt);
        }
        list.sort_by_key(|x| Backend::ALL.iter().position(|y| y == x));
        list.dedup();
        self.0 = Some(list);
    }

    pub fn cuda(&self, setup: &Setup) -> bool {
        self.is_on(Backend::Cuda, setup)
    }

    pub fn tensorrt(&self, setup: &Setup) -> bool {
        self.is_on(Backend::TensorRt, setup)
    }

    pub fn apple(&self, setup: &Setup) -> Option<ComputeUnits> {
        if self.is_on(Backend::AppleNeural, setup) {
            Some(ComputeUnits::CPUAndNeuralEngine)
        } else if self.is_on(Backend::AppleGpu, setup) {
            Some(ComputeUnits::CPUAndGPU)
        } else if self.is_on(Backend::AppleAll, setup) {
            Some(ComputeUnits::All)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Place, Source};

    fn with_ort() -> Setup {
        Setup { ort: Some(Place { dir: PathBuf::from("/x"), source: Source::Project }), ..Setup::default() }
    }

    #[test]
    fn nothing_is_in_use_without_the_runtime() {
        let none = Choice(None);
        assert!(Backend::ALL.iter().all(|b| !none.is_on(*b, &Setup::default())));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn one_part_of_the_apple_chip_at_a_time() {
        let setup = with_ort();
        let mut c = Choice(None);
        assert!(c.is_on(Backend::AppleNeural, &setup) && !c.is_on(Backend::Cpu, &setup));
        c.set(Backend::AppleGpu, true, &setup);
        assert!(c.is_on(Backend::AppleGpu, &setup) && !c.is_on(Backend::AppleNeural, &setup));
        c.set(Backend::AppleGpu, false, &setup);
        assert!(c.is_on(Backend::Cpu, &setup) && c.apple(&setup).is_none());
    }
}
