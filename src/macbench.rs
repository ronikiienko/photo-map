//! Speed and accuracy of the models on whatever the machine has, for the photos in `.cache/photos/` (any folder of real photos
//! will do: BENCH_PHOTOS). One process measures one setting, since ONNX Runtime allows one environment in a process:
//!
//!   BENCH_MODEL=s256|s512 BENCH_TAG=name [BENCH_N=96] [BENCH_BATCH=4] [BENCH_REF=tag] PHOTO_MAP_COREML=off|cpu|gpu|ane|all \
//!   cargo test --release mac_embed_speed -- --ignored --nocapture
//!
//! The vectors go to `.cache/bench/<tag>.f32`; with BENCH_REF the cosine to the vectors of that tag is printed (lowest and mean).

use crate::models;
use rayon::prelude::*;
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (dot, na, nb) = a.iter().zip(b).fold((0f32, 0f32, 0f32), |(d, x, y), (p, q)| (d + p * q, x + p * p, y + q * q));
    dot / (na.sqrt() * nb.sqrt())
}

/// Peak memory of this process so far, in MB.
fn peak_mb() -> f64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    let raw = usage.ru_maxrss as f64;
    if cfg!(target_os = "macos") { raw / 1048576.0 } else { raw / 1024.0 }
}

#[test]
#[ignore]
fn mac_embed_speed() {
    let root = crate::data_root();
    let id = match std::env::var("BENCH_MODEL").as_deref() {
        Ok("s512") => models::SIGLIP_BASE_512,
        _ => models::SIGLIP_BASE_256,
    };
    let env_num = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (n, batch) = (env_num("BENCH_N", 96), env_num("BENCH_BATCH", 4));
    let tag = std::env::var("BENCH_TAG").unwrap_or_else(|_| "run".into());
    let photos = std::env::var("BENCH_PHOTOS").map(PathBuf::from).unwrap_or_else(|_| root.join(".cache/photos"));
    let spec = models::spec(id).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(&photos).unwrap().flatten().map(|e| e.path()).collect();
    files.sort();
    files.truncate(n);

    let started = std::time::Instant::now();
    let prepared: Vec<_> = files.par_iter().filter_map(|f| crate::photo::load_for(f, |w, h| spec.longest_side_needed(w, h)).map(|i| spec.prepare(&i).unwrap())).collect();
    let prep = started.elapsed().as_secs_f32();
    let count = prepared.len();

    let started = std::time::Instant::now();
    let hub = models::Hub::start(root.clone(), || {});
    assert_eq!(hub.wait_for(id, &|| false), Ok(true), "{:?}", hub.state().failed);
    let load = started.elapsed().as_secs_f32();

    let mut vectors: Vec<f32> = Vec::new();
    let (first, rest_time, rest) = hub
        .with(id, |m| {
            let mut prepared = prepared;
            let head: Vec<_> = prepared.drain(..batch.min(prepared.len())).collect();
            let started = std::time::Instant::now();
            for r in m.embed_prepared(head) {
                vectors.extend(r.unwrap());
            }
            let first = started.elapsed().as_secs_f32(); // the first photos pay for loading and compiling the network
            let rest = prepared.len();
            let started = std::time::Instant::now();
            while !prepared.is_empty() {
                let group: Vec<_> = prepared.drain(..batch.min(prepared.len())).collect();
                for r in m.embed_prepared(group) {
                    vectors.extend(r.unwrap());
                }
            }
            (first, started.elapsed().as_secs_f32(), rest)
        })
        .unwrap();
    let dir = root.join(".cache/bench");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{tag}.f32")), vectors.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let mut accuracy = String::new();
    if let Some(reference) = std::env::var("BENCH_REF").ok().filter(|r| r != "none") {
        let bytes = std::fs::read(dir.join(format!("{reference}.f32"))).unwrap();
        let theirs: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let dim = spec.dim;
        let cosines: Vec<f32> = vectors.chunks_exact(dim).zip(theirs.chunks_exact(dim)).map(|(a, b)| cosine(a, b)).collect();
        let low = cosines.iter().cloned().fold(f32::MAX, f32::min);
        accuracy = format!("; cosine to {reference}: lowest {low:.5}, mean {:.5} over {}", cosines.iter().sum::<f32>() / cosines.len() as f32, cosines.len());
    }
    eprintln!(
        "RESULT {tag}: {} {count} photos, batch {batch}: {:.1} photos/s (after the first {batch}: {rest} photos in {rest_time:.1} s); first group {first:.1} s, load {load:.1} s, prepare {:.0} photos/s, peak memory {:.0} MB{accuracy}",
        spec.label,
        rest as f32 / rest_time,
        count as f32 / prep,
        peak_mb()
    );
}


/// What the inputs and outputs of the model files in BENCH_FILES (a list, separated by commas) are called, and their types.
#[test]
#[ignore]
fn show_model_edges() {
    let root = crate::data_root();
    let rt = crate::embed::Runtime::shared(&root).unwrap();
    for f in std::env::var("BENCH_FILES").unwrap().split(',') {
        let session = ort::session::Session::builder(&rt.env).unwrap().commit_from_file(f).unwrap();
        eprintln!("EDGES {f}");
        for i in session.inputs() {
            eprintln!("  in  {} {:?}", i.name(), i.dtype());
        }
        for o in session.outputs() {
            eprintln!("  out {} {:?}", o.name(), o.dtype());
        }
    }
}
