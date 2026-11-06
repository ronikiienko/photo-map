//! What every model shares: ONNX Runtime, started once in this process (no Python).
//!
//! Everything it needs lives in the app's data folder: ONNX Runtime and the CUDA libraries in runtime/, the models in .cache/onnx/
//! (models.rs). The CUDA libraries are loaded by full path before the first session is made, so nothing has to be installed or put
//! on a search path.

use crate::runtime::{self, Setup};
use ort::ep;
use ort::environment::Environment;
use ort::session::Session;
use std::path::Path;

pub type Result<T> = std::result::Result<T, String>;

fn err(what: &str) -> impl Fn(ort::Error) -> String + '_ {
    move |e| format!("{what}: {e}")
}

/// What every model shares: the ONNX Runtime environment (one per process), and whether the graphics card and TensorRT can be used.
pub struct Runtime {
    pub env: Environment,
    pub on_gpu: bool,
    pub tensorrt: bool,
    /// On a Mac: what the photo encoders run on besides the processor (CoreML); None: the processor alone.
    pub apple: Option<ep::coreml::ComputeUnits>,
}

/// A session builder for a photo encoder: with CoreML when the machine has it and `apple` says so.
/// `sizes`: the dimensions the model leaves open, fixed to these sizes (CoreML needs shapes that do not change).
pub fn vision_builder(env: &Environment, apple: Option<ep::coreml::ComputeUnits>, cache: &Path, sizes: &[(&str, i64)]) -> ort::Result<ort::session::builder::SessionBuilder> {
    let mut builder = Session::builder(env)?;
    let Some(units) = apple else { return Ok(builder) };
    for (name, size) in sizes {
        builder = builder.with_dimension_override(name, *size)?;
    }
    let _ = std::fs::create_dir_all(cache);
    let coreml = ep::CoreML::default().with_compute_units(units).with_model_format(if std::env::var("PHOTO_MAP_COREML_FORMAT").as_deref() == Ok("nn") { ep::coreml::ModelFormat::NeuralNetwork } else { ep::coreml::ModelFormat::MLProgram }).with_model_cache_dir(cache.display()).build();
    builder.with_execution_providers([coreml]).map_err(|e| e.into())
}

/// The CUDA provider, for the sessions a build runs on the card (a search's never start CUDA, which holds video memory of its own).
pub fn cuda() -> ep::ExecutionProviderDispatch {
    ep::CUDA::default().with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested).build() // allocate what is needed, not the next power of two (the card is shared with the map)
}

impl Runtime {
    /// The process's runtime, started the first time it is asked for (ONNX Runtime allows one environment in a process: the models'
    /// hub and the tests share it). A failure to start is kept too: it would fail the same way again.
    pub fn shared(root: &Path) -> Result<&'static Runtime> {
        static RUNTIME: std::sync::OnceLock<Result<Runtime>> = std::sync::OnceLock::new();
        RUNTIME.get_or_init(|| Runtime::start(root)).as_ref().map_err(|e| e.clone())
    }

    /// Start ONNX Runtime from the libraries `runtime::Setup` finds, with the CUDA libraries loaded by full path (a missing one
    /// means no GPU).
    fn start(root: &Path) -> Result<Self> {
        let setup = Setup::detect(root);
        let Some(ort_place) = &setup.ort else {
            return Err(format!("ONNX Runtime has not been downloaded (looked in {})", root.join("runtime").display()));
        };
        let onnx = ort_place.dir.join(runtime::ORT_LIB);
        let chosen = crate::backends::Choice::load(root);
        let mut on_gpu = false;
        if let (Some(gpu), true, true) = (&setup.gpu, setup.ort_cuda, chosen.cuda(&setup)) {
            on_gpu = runtime::CUDA_LIBS.iter().all(|l| ort::util::preload_dylib(gpu.cuda.dir.join(l)).is_ok())
                && runtime::CUDNN_LIBS.iter().all(|l| ort::util::preload_dylib(gpu.cudnn.dir.join(l)).is_ok());
        }
        // no provider for every session: searches run on the processor, only a build's sessions ask for the card (`cuda`)
        let env = ort::init_from(&onnx).map_err(|e| format!("loading ONNX Runtime: {e}"))?.with_name("photo-map").build().map_err(err("starting ONNX Runtime"))?;
        let tensorrt = on_gpu
            && chosen.tensorrt(&setup)
            && setup.ort_tensorrt
            && setup.tensorrt.as_ref().is_some_and(|t| runtime::TENSORRT_LIBS.iter().all(|l| ort::util::preload_dylib(t.dir.join(l)).is_ok()));
        let apple = chosen.apple(&setup);
        Ok(Self { env, on_gpu, tensorrt, apple })
    }
}
