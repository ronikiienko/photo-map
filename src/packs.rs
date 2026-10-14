//! Packs: the runtime libraries this program can install itself (see runtime.rs for where libraries are looked for).
//!
//! A pack is some files out of a published Python wheel (a zip file), kept in `runtime/<pack id>/` with a `pack.json` that lists
//! what was put there. Only the wanted files are fetched, with HTTP range requests, so the TensorRT wheel (3.7 GB, for every card
//! made) costs the few hundred MB the one card needs. What is fetched is checked against a pinned fingerprint of the wanted entries
//! (names, sizes and CRC-32s as the wheel's own directory lists them), and every file is checked against its CRC-32 as it is
//! unpacked, so a wheel that changed under its address is refused. Nothing is installed without the user asking, and only a folder
//! with a `pack.json` is ever removed.

use crate::runtime::Nvidia;
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

pub const MANIFEST: &str = "pack.json";

/// One file to take out of a wheel, and what to call it.
pub struct Want {
    /// The file's name inside the wheel (the folder it is in does not matter).
    pub member: &'static str,
    pub save_as: &'static str,
}

const fn same(name: &'static str) -> Want {
    Want { member: name, save_as: name }
}

pub struct Wheel {
    pub url: &'static str,
    /// Fingerprint of the wanted entries; see `fingerprint`.
    pub pin: &'static str,
    pub wants: &'static [Want],
}

pub struct Pack {
    pub id: &'static str,
    pub label: &'static str,
    pub version: &'static str,
    /// About how much it downloads, in MB (the real figure is shown while it downloads).
    pub size_mb: u32,
    pub wheels: &'static [Wheel],
}

const ORT_VERSION: &str = "1.31.0";

pub const ONNXRUNTIME: Pack = Pack {
    id: "onnxruntime",
    label: "ONNX Runtime (processor only)",
    version: ORT_VERSION,
    size_mb: 11,
    wheels: &[Wheel {
        url: "https://files.pythonhosted.org/packages/6c/44/1e9e762b95b7da0a8424913a1ed7c38cdaf88624a3c41ddba24ebac88bc9/onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        pin: "c65e5b3e56e6181244335676c560d23d31198ab55c59fc63813ef9c95bf22aa8",
        wants: &[Want { member: "libonnxruntime.so.1.31.0", save_as: "libonnxruntime.so" }],
    }],
};

/// The Mac build (Apple Silicon, macOS 14 or newer): the processor and the CoreML provider are in the one library.
pub const ONNXRUNTIME_MACOS: Pack = Pack {
    id: "onnxruntime-macos",
    label: "ONNX Runtime (processor and Apple chip)",
    version: ORT_VERSION,
    size_mb: 21,
    wheels: &[Wheel {
        url: "https://files.pythonhosted.org/packages/b3/bd/2ac094311163b803e3626c3937461d6900934bd56cca7601f6150ff860c3/onnxruntime-1.31.0-cp312-cp312-macosx_14_0_arm64.whl",
        pin: "",
        wants: &[Want { member: "libonnxruntime.1.31.0.dylib", save_as: "libonnxruntime.dylib" }],
    }],
};

pub const ONNXRUNTIME_GPU: Pack = Pack {
    id: "onnxruntime-gpu",
    label: "ONNX Runtime (with graphics card support)",
    version: ORT_VERSION,
    size_mb: 232,
    wheels: &[Wheel {
        url: "https://files.pythonhosted.org/packages/8f/48/6fbfdbce25e5756c7b9ff8f40645d98b166b8022cf00ede58a2bb4571b8e/onnxruntime_gpu-1.31.0-cp312-cp312-manylinux_2_28_x86_64.whl",
        pin: "9e3f1b84ba7baaf3bc5d6e7827a46cbeb8e35785283a31ac2614cd534062ae4e",
        wants: &[
            Want { member: "libonnxruntime.so.1.31.0", save_as: "libonnxruntime.so" },
            same("libonnxruntime_providers_shared.so"),
            same("libonnxruntime_providers_cuda.so"),
            same("libonnxruntime_providers_tensorrt.so"),
        ],
    }],
};

pub const CUDA: Pack = Pack {
    id: "cuda-13",
    label: "NVIDIA CUDA 13",
    version: "13.0",
    size_mb: 460,
    wheels: &[
        Wheel {
            url: "https://files.pythonhosted.org/packages/2e/24/d1558f3b68b1d26e706813b1d10aa1d785e4698c425af8db8edc3dced472/nvidia_cuda_runtime-13.0.96-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
            pin: "0587711defcf8dc46750181c0e0a75eb7944569b4522ee59bd35d2676b445335",
            wants: &[same("libcudart.so.13")],
        },
        Wheel {
            url: "https://files.pythonhosted.org/packages/3b/cd/154ca20c38269e05eff77c1464e6c1da89f50a6390b565e9d82e06bc11e1/nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl",
            pin: "2ced140860a41c4d164096b7d4b6a872cf780b062c159a2eeaab695f3176463a",
            wants: &[same("libcublas.so.13"), same("libcublasLt.so.13")],
        },
        Wheel {
            url: "https://files.pythonhosted.org/packages/a5/9f/be0a41ca4a4917abf5cb9ae0daff1a6060cc5de950aec0396de9f3b52bc5/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl",
            pin: "c0906a373df926f861e7c8ab9f78bd563d181447bd8e913fcfeea413850d690a",
            wants: &[same("libcurand.so.10")],
        },
    ],
};

pub const CUDNN: Pack = Pack {
    id: "cudnn-9",
    label: "NVIDIA cuDNN 9",
    version: "9.24",
    size_mb: 527,
    wheels: &[Wheel {
        url: "https://files.pythonhosted.org/packages/5c/ba/791cffd048fe5b044e620df55267e3e95c0e6e07d50b41e377c03dfc910f/nvidia_cudnn_cu13-9.24.0.43-py3-none-manylinux_2_27_x86_64.whl",
        pin: "5406885654fe5530a169be2773d38ff50a4998484b3fbab2fd86ba321df03d31",
        wants: &[
            same("libcudnn.so.9"),
            same("libcudnn_graph.so.9"),
            same("libcudnn_ops.so.9"),
            same("libcudnn_heuristic.so.9"),
            same("libcudnn_adv.so.9"),
            same("libcudnn_cnn.so.9"),
            same("libcudnn_engines_precompiled.so.9"),
            same("libcudnn_engines_runtime_compiled.so.9"),
            same("libcudnn_engines_tensor_ir.so.9"),
            same("libcudnn_ext.so.9"),
        ],
    }],
};

/// TensorRT. Besides the three libraries it needs the "builder resource" for the card's architecture (one file per architecture
/// in the wheel, `sm86` for compute capability 8.6), so what is wanted depends on the card; see `wants_for`.
pub const TENSORRT: Pack = Pack {
    id: "tensorrt-10",
    label: "NVIDIA TensorRT 10",
    version: "10.16",
    size_mb: 456,
    wheels: &[Wheel {
        url: "https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-10.16.1.11-py3-none-manylinux_2_28_x86_64.whl",
        pin: "4fa55167a28ef01fdd7002e3240f3a2e9eda594310a41a1f27b6ad3c8ba78e72",
        wants: &[same("libnvinfer.so.10"), same("libnvinfer_plugin.so.10"), same("libnvonnxparser.so.10")],
    }],
};

pub const ALL: [&Pack; 6] = [&ONNXRUNTIME, &ONNXRUNTIME_MACOS, &ONNXRUNTIME_GPU, &CUDA, &CUDNN, &TENSORRT];

/// The ONNX Runtime for a machine with no graphics card of the kind the other packs are for: this operating system's own build.
pub fn onnxruntime_cpu() -> &'static Pack {
    if cfg!(target_os = "macos") { &ONNXRUNTIME_MACOS } else { &ONNXRUNTIME }
}

pub fn pack(id: &str) -> Option<&'static Pack> {
    ALL.iter().copied().find(|p| p.id == id)
}

/// The files of a pack's wheel to take, for this card. Only TensorRT's list depends on it.
fn wants_for(pack: &Pack, wheel: &Wheel, card: Option<&Nvidia>) -> Vec<(String, String)> {
    let mut wanted: Vec<(String, String)> = wheel.wants.iter().map(|w| (w.member.to_string(), w.save_as.to_string())).collect();
    if pack.id == TENSORRT.id {
        let name = match card {
            Some(c) => format!("libnvinfer_builder_resource_sm{}.so.10.16.1", c.compute),
            None => "libnvinfer_builder_resource_ptx.so.10.16.1".to_string(),
        };
        wanted.push((name.clone(), name));
    }
    wanted
}

/// The id in a folder's `pack.json`, if the folder is a pack this program installed.
pub fn installed_id(dir: &Path) -> Option<String> {
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).ok()?).ok()?;
    Some(manifest.id)
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub id: String,
    pub version: String,
    /// The files put in the folder (the manifest itself is not in the list).
    pub files: Vec<File>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct File {
    pub name: String,
    pub size: u64,
}

/// A pack installed by this program: what it is, how big, where.
pub struct Installed {
    pub id: String,
    pub version: String,
    pub bytes: u64,
}

/// The packs this program installed (not other folders in `runtime/`).
pub fn installed(root: &Path) -> Vec<Installed> {
    let Ok(entries) = std::fs::read_dir(root.join("runtime")) else { return Vec::new() };
    let mut found: Vec<Installed> = entries
        .flatten()
        .map(|e| e.path())
        .filter_map(|dir| {
            let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).ok()?).ok()?;
            Some(Installed { bytes: manifest.files.iter().map(|f| f.size).sum(), id: manifest.id, version: manifest.version })
        })
        .collect();
    found.sort_by(|a, b| a.id.cmp(&b.id));
    found
}

/// Remove a pack this program installed: exactly the files its `pack.json` lists. A folder without one is left alone.
pub fn remove(root: &Path, id: &str) -> Result<(), String> {
    let dir = root.join("runtime").join(id);
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(dir.join(MANIFEST)).map_err(|_| format!("{id} was not installed by this program; it is left alone"))?)
        .map_err(|e| format!("reading {}: {e}", dir.join(MANIFEST).display()))?;
    if manifest.id != id {
        return Err(format!("{} belongs to {}, not {id}", dir.display(), manifest.id));
    }
    for f in &manifest.files {
        // a name with a path in it is not one this program wrote
        if f.name.contains('/') || f.name.contains('\\') || f.name == ".." {
            return Err(format!("unexpected file name {:?} in {}", f.name, dir.join(MANIFEST).display()));
        }
        match std::fs::remove_file(dir.join(&f.name)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(format!("removing {}: {e}", dir.join(&f.name).display())),
        }
    }
    std::fs::remove_file(dir.join(MANIFEST)).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir(&dir); // only if nothing else is in it
    Ok(())
}

/// A file on a web server, read with range requests (so a zip file's directory and then the wanted files can be taken out of it
/// without the rest). Reads ahead in blocks.
struct Remote {
    agent: ureq::Agent,
    url: String,
    len: u64,
    pos: u64,
    block: Vec<u8>,
    block_at: u64,
}

const BLOCK: u64 = 4 << 20;

impl Remote {
    fn open(url: &str) -> Result<Remote, String> {
        let agent: ureq::Agent = ureq::Agent::config_builder().timeout_connect(Some(std::time::Duration::from_secs(30))).timeout_recv_body(Some(std::time::Duration::from_secs(120))).build().into();
        let head = agent.head(url).call().map_err(|e| format!("reaching {url}: {e}"))?;
        let len = head.headers().get("content-length").and_then(|v| v.to_str().ok()).and_then(|v| v.parse().ok()).ok_or_else(|| format!("{url} did not say how big it is"))?;
        if head.headers().get("accept-ranges").and_then(|v| v.to_str().ok()) != Some("bytes") {
            return Err(format!("{url} cannot be read in parts"));
        }
        Ok(Remote { agent, url: url.to_string(), len, pos: 0, block: Vec::new(), block_at: 0 })
    }

    fn fetch(&mut self, at: u64) -> std::io::Result<()> {
        let end = (at + BLOCK).min(self.len) - 1;
        let mut last = None;
        for attempt in 0..4 {
            if attempt > 0 {
                std::thread::sleep(std::time::Duration::from_secs(2 * attempt));
            }
            let got = self.agent.get(&self.url).header("Range", format!("bytes={at}-{end}")).call().map_err(|e| e.to_string()).and_then(|r| {
                let mut body = Vec::with_capacity((end - at + 1) as usize);
                r.into_body().into_reader().take(end - at + 1).read_to_end(&mut body).map_err(|e| e.to_string())?;
                if body.len() as u64 != end - at + 1 { Err(format!("got {} bytes of {}", body.len(), end - at + 1)) } else { Ok(body) }
            });
            match got {
                Ok(body) => {
                    self.block = body;
                    self.block_at = at;
                    return Ok(());
                }
                Err(e) => last = Some(e),
            }
        }
        Err(std::io::Error::other(format!("reading {}: {}", self.url, last.unwrap_or_default())))
    }
}

impl Read for Remote {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || out.is_empty() {
            return Ok(0);
        }
        if self.pos < self.block_at || self.pos >= self.block_at + self.block.len() as u64 {
            self.fetch(self.pos)?;
        }
        let from = (self.pos - self.block_at) as usize;
        let n = out.len().min(self.block.len() - from);
        out[..n].copy_from_slice(&self.block[from..from + n]);
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Remote {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let new = match to {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::End(n) => self.len as i64 + n,
            SeekFrom::Current(n) => self.pos as i64 + n,
        };
        if new < 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, "before the start"));
        }
        self.pos = new as u64;
        Ok(self.pos)
    }
}

fn base_name<R: Read>(entry: &zip::read::ZipFile<'_, R>) -> String {
    entry.name().map(|n| n.rsplit('/').next().unwrap_or_default().to_string()).unwrap_or_default()
}

/// Which entries of the archive are wanted: (index, name to save as).
fn pick<R: Read + Seek>(zip: &mut zip::ZipArchive<R>, wanted: &[(String, String)]) -> Result<Vec<(usize, String)>, String> {
    let mut picked = Vec::new();
    for (member, save_as) in wanted {
        let mut hit = None;
        for i in 0..zip.len() {
            if base_name(&zip.by_index_raw(i).map_err(|e| e.to_string())?) == *member {
                hit = Some(i);
                break;
            }
        }
        picked.push((hit.ok_or_else(|| format!("{member} is not in the download"))?, save_as.clone()));
    }
    Ok(picked)
}

/// The fingerprint of the wanted entries: SHA-256 of "name size crc32" lines, in the order wanted.
pub fn fingerprint<R: Read + Seek>(zip: &mut zip::ZipArchive<R>, picked: &[(usize, String)]) -> Result<String, String> {
    let mut hash = Sha256::new();
    for (i, _) in picked {
        let entry = zip.by_index_raw(*i).map_err(|e| e.to_string())?;
        hash.update(format!("{} {} {:08x}\n", base_name(&entry), entry.size(), entry.crc32()));
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// What installing a pack will fetch, in bytes (from the wheels' directories; reads only those).
#[cfg(test)]
fn download_size(pack: &Pack, card: Option<&Nvidia>) -> Result<u64, String> {
    let mut total = 0;
    for wheel in pack.wheels {
        let mut zip = zip::ZipArchive::new(Remote::open(wheel.url)?).map_err(|e| e.to_string())?;
        for (i, _) in pick(&mut zip, &wants_for(pack, wheel, card))? {
            total += zip.by_index_raw(i).map_err(|e| e.to_string())?.compressed_size();
        }
    }
    Ok(total)
}

/// Install a pack into `runtime/<id>/`. `progress(done, total)`: bytes fetched so far of all there is. Files already fetched by a
/// stopped attempt are kept (in `runtime/.<id>.partial/`) and not fetched again. `stop`: set to give up (what was fetched is kept).
pub fn install(root: &Path, pack: &Pack, card: Option<&Nvidia>, stop: &AtomicBool, progress: &dyn Fn(u64, u64)) -> Result<(), String> {
    let runtime = root.join("runtime");
    let dest = runtime.join(pack.id);
    if dest.exists() && installed_id(&dest).as_deref() != Some(pack.id) {
        return Err(format!("{} is there already and was not installed by this program; it is left alone", dest.display()));
    }
    let partial = runtime.join(format!(".{}.partial", pack.id));
    std::fs::create_dir_all(&partial).map_err(|e| format!("making {}: {e}", partial.display()))?;

    // what there is to fetch, so that the progress is of the whole
    let mut plan = Vec::new();
    for wheel in pack.wheels {
        let mut zip = zip::ZipArchive::new(Remote::open(wheel.url)?).map_err(|e| format!("reading the list of files in {}: {e}", wheel.url))?;
        let picked = pick(&mut zip, &wants_for(pack, wheel, card))?;
        if !wheel.pin.is_empty() {
            // the fingerprint covers what the card-dependent TensorRT file adds only when it is the same card; those are checked by CRC
            let fixed = &picked[..wheel.wants.len()];
            let got = fingerprint(&mut zip, fixed)?;
            if got != wheel.pin {
                return Err(format!("{} is not the file this version expects (fingerprint {got}); not installing it", wheel.url));
            }
        }
        let total: u64 = picked.iter().map(|(i, _)| zip.by_index_raw(*i).map(|e| e.compressed_size()).unwrap_or(0)).sum();
        plan.push((zip, picked, total));
    }
    let all: u64 = plan.iter().map(|(_, _, t)| t).sum();
    let mut done = 0u64;
    progress(0, all);
    let mut files = Vec::new();
    for (mut zip, picked, _) in plan {
        for (i, save_as) in picked {
            let (size, compressed) = {
                let e = zip.by_index_raw(i).map_err(|e| e.to_string())?;
                (e.size(), e.compressed_size())
            };
            let target = partial.join(&save_as);
            let have = std::fs::metadata(&target).map(|m| m.len()).ok();
            if have != Some(size) {
                let part = partial.join(format!("{save_as}.part"));
                let mut out = std::fs::File::create(&part).map_err(|e| format!("writing {}: {e}", part.display()))?;
                let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
                let mut buf = vec![0u8; 1 << 20];
                let mut written = 0u64;
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return Err("stopped".into());
                    }
                    let n = entry.read(&mut buf).map_err(|e| format!("unpacking {save_as}: {e}"))?; // the CRC-32 is checked at the end
                    if n == 0 {
                        break;
                    }
                    out.write_all(&buf[..n]).map_err(|e| format!("writing {}: {e}", part.display()))?;
                    written += n as u64;
                    progress(done + (written * compressed / size.max(1)).min(compressed), all);
                }
                out.flush().map_err(|e| e.to_string())?;
                drop(out);
                std::fs::rename(&part, &target).map_err(|e| e.to_string())?;
            }
            done += compressed;
            progress(done, all);
            files.push(File { name: save_as, size });
        }
    }
    let manifest = Manifest { id: pack.id.to_string(), version: pack.version.to_string(), files };
    std::fs::write(partial.join(MANIFEST), serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if dest.exists() {
        remove(root, pack.id)?; // an older install of this pack
        let _ = std::fs::remove_dir_all(&dest);
    }
    std::fs::rename(&partial, &dest).map_err(|e| format!("moving {} into place: {e}", partial.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp(name: &str) -> PathBuf {
        // inside the project, like everything this program downloads
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test").join(format!("packs-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("runtime")).unwrap();
        dir
    }

    #[test]
    fn only_a_pack_this_program_installed_can_be_removed() {
        let root = temp("remove");
        let own = root.join("runtime/cuda-13");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(own.join("libcudart.so.13"), b"x").unwrap();
        std::fs::write(own.join("somebody-elses.txt"), b"keep").unwrap();
        let manifest = Manifest { id: "cuda-13".into(), version: "13.0".into(), files: vec![File { name: "libcudart.so.13".into(), size: 1 }] };
        std::fs::write(own.join(MANIFEST), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let theirs = root.join("runtime/cudnn");
        std::fs::create_dir_all(&theirs).unwrap();
        std::fs::write(theirs.join("libcudnn.so.9"), b"x").unwrap();

        assert!(remove(&root, "cudnn").is_err(), "a folder with no pack.json");
        assert!(theirs.join("libcudnn.so.9").exists());
        assert_eq!(installed(&root).len(), 1);

        remove(&root, "cuda-13").unwrap();
        assert!(!own.join("libcudart.so.13").exists() && !own.join(MANIFEST).exists());
        assert!(own.join("somebody-elses.txt").exists(), "what is not in the manifest stays");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_manifest_cannot_point_outside_its_folder() {
        let root = temp("escape");
        let own = root.join("runtime/cuda-13");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(root.join("precious"), b"keep").unwrap();
        let manifest = Manifest { id: "cuda-13".into(), version: "13.0".into(), files: vec![File { name: "../../precious".into(), size: 1 }] };
        std::fs::write(own.join(MANIFEST), serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(remove(&root, "cuda-13").is_err());
        assert!(root.join("precious").exists());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_folder_that_is_not_ours_is_not_installed_over() {
        let root = temp("over");
        std::fs::create_dir_all(root.join("runtime/cuda-13")).unwrap();
        let stop = AtomicBool::new(false);
        let err = install(&root, &CUDA, None, &stop, &|_, _| {}).unwrap_err();
        assert!(err.contains("left alone"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Installs the small processor-only pack for real, then removes it (needs the network): cargo test --release installs_a_real_pack -- --ignored --nocapture
    #[test]
    #[ignore]
    fn installs_a_real_pack() {
        let root = temp("real");
        let stop = AtomicBool::new(false);
        let last = std::sync::Mutex::new((0, 0));
        install(&root, onnxruntime_cpu(), None, &stop, &|done, all| *last.lock().unwrap() = (done, all)).unwrap();
        let (done, all) = *last.lock().unwrap();
        assert_eq!(done, all, "progress ends complete");
        let dir = root.join("runtime").join(onnxruntime_cpu().id);
        assert!(std::fs::metadata(dir.join(crate::runtime::ORT_LIB)).unwrap().len() > 10_000_000);
        assert_eq!(installed(&root).len(), 1);
        assert!(!root.join("runtime").join(format!(".{}.partial", onnxruntime_cpu().id)).exists());
        let setup = crate::runtime::Setup::detect(&root);
        let ort = setup.ort.expect("found after installing");
        assert!(ort.source.ours());
        remove(&root, onnxruntime_cpu().id).unwrap();
        assert!(!dir.join(crate::runtime::ORT_LIB).exists());
        assert!(crate::runtime::Setup::detect(&root).ort.is_none());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Installs cuDNN for real and checks every library cuDNN needs is there (needs the network, 0.5 GB): cargo test --release installs_all_of_cudnn -- --ignored --nocapture
    #[test]
    #[ignore]
    fn installs_all_of_cudnn() {
        let root = temp("cudnn");
        let stop = AtomicBool::new(false);
        install(&root, &CUDNN, None, &stop, &|_, _| {}).unwrap();
        let dir = root.join("runtime/cudnn-9");
        for lib in crate::runtime::CUDNN_LIBS {
            assert!(std::fs::metadata(dir.join(lib)).unwrap().len() > 100_000, "{lib}");
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Prints the fingerprints to pin and what each pack downloads (needs the network): cargo test --release print_pins -- --ignored --nocapture
    #[test]
    #[ignore]
    fn print_pins() {
        let card = Nvidia { name: "any".into(), compute: 86, driver_cuda: 13000 };
        for p in ALL {
            for w in p.wheels {
                let mut zip = zip::ZipArchive::new(Remote::open(w.url).unwrap()).unwrap();
                let picked = pick(&mut zip, &wants_for(p, w, Some(&card))[..w.wants.len()]).unwrap();
                eprintln!("{} {} pin: \"{}\"", p.id, w.url.rsplit('/').next().unwrap(), fingerprint(&mut zip, &picked).unwrap());
            }
            eprintln!("{} downloads {:.0} MB (for a card of compute capability 8.6)", p.id, download_size(p, Some(&card)).unwrap() as f64 / 1048576.0);
        }
    }
}
