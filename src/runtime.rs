//! Where the libraries the models run on come from: what is installed, by whom, and what the machine could use.
//!
//! The libraries (ONNX Runtime, CUDA, cuDNN, TensorRT) are looked for in this order, and the first place that has a complete set wins:
//! 1. `runtime/<folder>/` in the project: a pack this program installed (the folder has a `pack.json`, see packs.rs) or a folder
//!    somebody put there by hand;
//! 2. the folders named in the environment variable `PHOTO_MAP_RUNTIME`;
//! 3. the system's own: `LD_LIBRARY_PATH` and the usual install folders.
//! A set is never put together from two of these (cuDNN of one place with the CUDA of another), and a library only counts when its
//! file name has the version this build of ONNX Runtime needs (`libcudart.so.13`: CUDA 13). The program never writes to, or offers
//! to remove, anything it did not install itself.

use std::path::{Path, PathBuf};

/// ONNX Runtime's core library.
pub const ORT_LIB: &str = "libonnxruntime.so";
/// Its CUDA provider, next to the core library: only a build for graphics cards has it.
pub const ORT_CUDA_PROVIDER: &str = "libonnxruntime_providers_cuda.so";
/// Its TensorRT provider.
pub const ORT_TENSORRT_PROVIDER: &str = "libonnxruntime_providers_tensorrt.so";
/// CUDA libraries to load, dependencies first (ONNX Runtime's CUDA provider needs these four).
pub const CUDA_LIBS: [&str; 4] = ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13", "libcurand.so.10"];
/// cuDNN (its softmax and attention helpers), also dependencies first.
pub const CUDNN_LIBS: [&str; 8] = [
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
];
/// TensorRT's libraries, dependencies first.
pub const TENSORRT_LIBS: [&str; 3] = ["libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"];
/// The oldest CUDA the graphics card's driver must support for the CUDA libraries above (13.0), in the driver's own numbering.
pub const CUDA_DRIVER_NEEDED: u32 = 13000;

/// Where a set of libraries was found.
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// Installed by this program (the pack's id).
    Managed(String),
    /// In `runtime/` but not installed by this program.
    Project,
    /// In a folder named by `PHOTO_MAP_RUNTIME`.
    Override,
    /// The system's own.
    System,
}

impl Source {
    /// Sets are not mixed across ranks.
    fn rank(&self) -> u8 {
        match self {
            Source::Managed(_) | Source::Project => 0,
            Source::Override => 1,
            Source::System => 2,
        }
    }

    /// Is it ours to remove?
    pub fn ours(&self) -> bool {
        matches!(self, Source::Managed(_))
    }
}

#[derive(Clone, Debug)]
pub struct Place {
    pub dir: PathBuf,
    pub source: Source,
}

/// What is installed and usable.
#[derive(Clone, Debug, Default)]
pub struct Setup {
    /// ONNX Runtime (the folder with `libonnxruntime.so`).
    pub ort: Option<Place>,
    /// Whether that ONNX Runtime has its CUDA provider.
    pub ort_cuda: bool,
    /// Whether it has its TensorRT provider.
    pub ort_tensorrt: bool,
    /// CUDA and cuDNN, from one source.
    pub gpu: Option<Gpu>,
    pub tensorrt: Option<Place>,
}

#[derive(Clone, Debug)]
pub struct Gpu {
    pub cuda: Place,
    pub cudnn: Place,
}

/// The folders libraries could be in, best first: the project's `runtime/` subfolders, the override, the system's.
fn candidates(root: &Path, extra: Option<&std::ffi::OsStr>) -> Vec<Place> {
    let mut found = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root.join("runtime")) {
        let mut dirs: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for dir in dirs {
            let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if name.starts_with('.') {
                continue; // a download in progress
            }
            let source = match crate::packs::installed_id(&dir) {
                Some(id) => Source::Managed(id),
                None => Source::Project,
            };
            found.push(Place { dir, source });
        }
    }
    if let Some(paths) = extra {
        found.extend(std::env::split_paths(paths).map(|dir| Place { dir, source: Source::Override }));
    }
    let mut system: Vec<PathBuf> = std::env::var_os("LD_LIBRARY_PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
    for fixed in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib", "/opt/cuda/lib64", "/opt/cuda/lib"] {
        system.push(PathBuf::from(fixed));
    }
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut cudas: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("cuda"))).collect();
        cudas.sort();
        for c in cudas {
            system.push(c.join("lib64"));
        }
    }
    found.extend(system.into_iter().filter(|d| !d.as_os_str().is_empty() && d.is_dir()).map(|dir| Place { dir, source: Source::System }));
    found
}

fn complete(dir: &Path, libs: &[&str]) -> bool {
    libs.iter().all(|l| dir.join(l).exists())
}

fn first<'a>(places: impl Iterator<Item = &'a Place>, libs: &[&str]) -> Option<Place> {
    places.filter(|p| complete(&p.dir, libs)).next().cloned()
}

impl Setup {
    pub fn detect(root: &Path) -> Setup {
        Self::detect_in(root, std::env::var_os("PHOTO_MAP_RUNTIME").as_deref())
    }

    /// `extra`: the folders of `PHOTO_MAP_RUNTIME`.
    fn detect_in(root: &Path, extra: Option<&std::ffi::OsStr>) -> Setup {
        let places = candidates(root, extra);
        // ONNX Runtime itself is ours or the user's own choice, never the system's (its version has to be the one this build is for)
        let own = || places.iter().filter(|p| p.source != Source::System);
        let ort = first(own().filter(|p| p.dir.join(ORT_CUDA_PROVIDER).exists()), &[ORT_LIB]).or_else(|| first(own(), &[ORT_LIB]));
        let (ort_cuda, ort_tensorrt) = ort.as_ref().map_or((false, false), |o| (o.dir.join(ORT_CUDA_PROVIDER).exists(), o.dir.join(ORT_TENSORRT_PROVIDER).exists()));
        let mut gpu = None;
        let mut tensorrt = None;
        for rank in 0..=2 {
            let here = || places.iter().filter(move |p| p.source.rank() == rank);
            if gpu.is_none() {
                if let (Some(cuda), Some(cudnn)) = (first(here(), &CUDA_LIBS), first(here(), &CUDNN_LIBS)) {
                    gpu = Some(Gpu { cuda, cudnn });
                }
            }
            if tensorrt.is_none() {
                tensorrt = first(here(), &TENSORRT_LIBS);
            }
        }
        Setup { ort, ort_cuda, ort_tensorrt, gpu, tensorrt }
    }

    /// Libraries of another version than the one needed, if that is why a set was not found (`libcudart.so.12` where `.13` is needed).
    pub fn wrong_versions(root: &Path) -> Vec<String> {
        let mut notes = Vec::new();
        for (what, stem, wanted) in [("CUDA", "libcudart.so.", CUDA_LIBS[0]), ("cuDNN", "libcudnn.so.", CUDNN_LIBS[0]), ("TensorRT", "libnvinfer.so.", TENSORRT_LIBS[0])] {
            let mut others = Vec::new();
            for place in candidates(root, std::env::var_os("PHOTO_MAP_RUNTIME").as_deref()) {
                let Ok(entries) = std::fs::read_dir(&place.dir) else { continue };
                for e in entries.flatten() {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if let Some(rest) = name.strip_prefix(stem) {
                        let major = rest.split('.').next().unwrap_or_default();
                        if !major.is_empty() && format!("{stem}{major}") != wanted && !others.contains(&major.to_string()) {
                            others.push(major.to_string());
                        }
                    }
                }
            }
            let major_wanted = wanted.rsplit('.').next().unwrap_or_default();
            if !others.is_empty() {
                notes.push(format!("{what} {} is installed on this computer, but this program needs {what} {major_wanted}", others.join(" and ")));
            }
        }
        notes
    }
}

/// The NVIDIA graphics card the CUDA libraries would run on, as its driver reports it.
#[derive(Clone, Debug)]
pub struct Nvidia {
    pub name: String,
    /// Compute capability: 86 for 8.6.
    pub compute: u32,
    /// The newest CUDA the driver supports, as 13000 for 13.0.
    pub driver_cuda: u32,
}

impl Nvidia {
    /// Is the driver new enough for the CUDA libraries this program uses?
    pub fn driver_ok(&self) -> bool {
        self.driver_cuda >= CUDA_DRIVER_NEEDED
    }

    /// Ask the driver (`libcuda.so.1`, which the graphics driver installs); None if there is no NVIDIA card or driver.
    pub fn probe() -> Option<Nvidia> {
        use std::os::raw::{c_char, c_int};
        unsafe {
            let lib = libloading::Library::new("libcuda.so.1").ok()?;
            let init: libloading::Symbol<unsafe extern "C" fn(u32) -> c_int> = lib.get(b"cuInit").ok()?;
            let count: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> = lib.get(b"cuDeviceGetCount").ok()?;
            let get: libloading::Symbol<unsafe extern "C" fn(*mut c_int, c_int) -> c_int> = lib.get(b"cuDeviceGet").ok()?;
            let name: libloading::Symbol<unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int> = lib.get(b"cuDeviceGetName").ok()?;
            let attribute: libloading::Symbol<unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int> = lib.get(b"cuDeviceGetAttribute").ok()?;
            let version: libloading::Symbol<unsafe extern "C" fn(*mut c_int) -> c_int> = lib.get(b"cuDriverGetVersion").ok()?;
            if init(0) != 0 {
                return None;
            }
            let (mut n, mut device, mut driver) = (0, 0, 0);
            if count(&mut n) != 0 || n < 1 || get(&mut device, 0) != 0 || version(&mut driver) != 0 {
                return None;
            }
            let mut buf = [0 as c_char; 128];
            let label = if name(buf.as_mut_ptr(), buf.len() as c_int, device) == 0 { std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() } else { "NVIDIA graphics card".to_string() };
            let (mut major, mut minor) = (0, 0);
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76
            if attribute(&mut major, 75, device) != 0 || attribute(&mut minor, 76, device) != 0 {
                return None;
            }
            Some(Nvidia { name: label, compute: (major * 10 + minor) as u32, driver_cuda: driver as u32 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(dir: &Path, names: &[&str]) {
        std::fs::create_dir_all(dir).unwrap();
        for n in names {
            std::fs::write(dir.join(n), b"x").unwrap();
        }
    }

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("photo-map-runtime-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_folder_somebody_put_there_is_found_but_not_ours() {
        let root = temp("project");
        touch(&root.join("runtime/ort"), &[ORT_LIB, ORT_CUDA_PROVIDER]);
        touch(&root.join("runtime/cuda"), &CUDA_LIBS);
        touch(&root.join("runtime/cudnn"), &CUDNN_LIBS);
        let setup = Setup::detect(&root);
        assert!(setup.ort_cuda && !setup.ort_tensorrt);
        let gpu = setup.gpu.expect("a complete set");
        assert_eq!(gpu.cuda.source, Source::Project);
        assert!(!gpu.cudnn.source.ours());
        assert!(setup.tensorrt.is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_set_with_a_library_missing_is_not_used() {
        let root = temp("partial");
        touch(&root.join("runtime/ort"), &[ORT_LIB]);
        touch(&root.join("runtime/cuda"), &CUDA_LIBS[..3]);
        touch(&root.join("runtime/cudnn"), &CUDNN_LIBS);
        assert!(Setup::detect(&root).gpu.is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cuda_and_cudnn_are_not_taken_from_different_places() {
        let root = temp("mixed");
        let other = temp("mixed-override");
        touch(&root.join("runtime/cuda"), &CUDA_LIBS); // the project has CUDA only
        touch(&other, &CUDNN_LIBS); // the override has cuDNN only
        let setup = Setup::detect_in(&root, Some(other.as_os_str()));
        assert!(setup.gpu.is_none(), "a set from two places");
        std::fs::remove_dir_all(&root).unwrap();
        std::fs::remove_dir_all(&other).unwrap();
    }

    #[test]
    fn libraries_of_another_version_are_reported_not_used() {
        let root = temp("version");
        touch(&root.join("runtime/cuda12"), &["libcudart.so.12", "libcublas.so.12"]);
        assert!(Setup::detect(&root).gpu.is_none());
        let notes = Setup::wrong_versions(&root);
        assert!(notes.iter().any(|n| n.contains("CUDA 12") && n.contains("needs CUDA 13")), "{notes:?}");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
