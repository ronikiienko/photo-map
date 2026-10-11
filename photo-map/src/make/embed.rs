//! One EmbeddingGemma 2 vector per photo. The model runs in PyTorch, the fastest option on this GPU (see
//! src/bench_onnx.py), inside a small Python worker, src/embed_worker.py, which this drives over stdin/stdout:
//! one JSON line per job ({"paths": [...]} for images, {"texts": [...], "prompt": ...} for words), each with an
//! "out" file the worker writes the vectors to (float32) before answering "done".
//! Shards are kept in work/, so a stopped run continues where it left off (same folders needed).

use super::step;
use crate::data::Photo;
use std::io::{self, BufRead, BufReader, Lines, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

pub const DIM: usize = 768;
const SHARD: usize = 1000;

/// The Python worker, started the first time it's needed (the model takes a while to load).
pub struct Python {
    project: PathBuf,
    log: PathBuf,
    worker: Option<(Child, ChildStdin, Lines<BufReader<ChildStdout>>)>,
}

impl Python {
    pub fn new(project: &Path, work: &Path) -> Self {
        Self { project: project.to_path_buf(), log: work.join("embed_worker.log"), worker: None }
    }

    /// Run one job; the worker writes its vectors (n x DIM float32) to `out`.
    pub fn ask(&mut self, mut job: serde_json::Value, out: &Path) -> io::Result<Vec<f32>> {
        let failed = |log: &Path| io::Error::other(format!("embed worker stopped; see {}", log.display()));
        if self.worker.is_none() {
            let python = self.project.join(".venv/bin/python");
            let mut child = Command::new(&python)
                .arg(self.project.join("src/embed_worker.py"))
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(std::fs::File::create(&self.log)?)
                .spawn()
                .map_err(|e| io::Error::other(format!("can't start {}: {e}", python.display())))?;
            let (to, from) = (child.stdin.take().unwrap(), BufReader::new(child.stdout.take().unwrap()).lines());
            self.worker = Some((child, to, from));
            self.answer().map_err(|_| failed(&self.log))?;
            step("model loaded");
        }
        job["out"] = out.to_string_lossy().into();
        let (_, to, _) = self.worker.as_mut().unwrap();
        writeln!(to, "{job}")?;
        to.flush()?;
        self.answer().map_err(|_| failed(&self.log))?;
        read_f32(out)
    }

    fn answer(&mut self) -> Result<(), ()> {
        match self.worker.as_mut().unwrap().2.next() {
            Some(Ok(line)) if line == "done" || line == "ready" => Ok(()),
            _ => Err(()),
        }
    }

    /// End of input: the worker exits.
    pub fn finish(&mut self) -> io::Result<()> {
        if let Some((mut child, to, _)) = self.worker.take() {
            drop(to);
            child.wait()?;
        }
        Ok(())
    }
}

pub fn read_f32(path: &Path) -> io::Result<Vec<f32>> {
    Ok(std::fs::read(path)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

pub fn embed(python: &mut Python, bundle: &Path, photos: &[Photo]) -> io::Result<Vec<f32>> {
    let work = bundle.join("work");
    let n = photos.len();
    let shards: Vec<(usize, usize)> = (0..n).step_by(SHARD).map(|s| (s, (s + SHARD).min(n))).collect();
    let path = |k: usize| work.join(format!("emb_{k:04}.f32"));
    let todo: Vec<usize> = (0..shards.len()).filter(|&k| !path(k).exists()).collect();
    step(format!("embedding {n} photos in {} shards ({} already done)", shards.len(), shards.len() - todo.len()));

    let started = Instant::now();
    let mut done = 0;
    for (j, &k) in todo.iter().enumerate() {
        let (a, b) = shards[k];
        let paths: Vec<_> = photos[a..b].iter().map(|p| bundle.join(&p.file)).collect();
        let tmp = work.join("shard.tmp");
        python.ask(serde_json::json!({ "paths": paths }), &tmp)?;
        std::fs::rename(&tmp, path(k))?;
        done += b - a;
        let rate = done as f32 / started.elapsed().as_secs_f32();
        let left: usize = todo[j + 1..].iter().map(|&k| shards[k].1 - shards[k].0).sum();
        step(format!("shard {}/{}: {rate:.1} img/s, ~{:.0} min left", j + 1, todo.len(), left as f32 / rate / 60.0));
    }

    let mut vectors = Vec::with_capacity(n * DIM);
    for (k, &(a, b)) in shards.iter().enumerate() {
        let shard = read_f32(&path(k))?;
        if shard.len() != (b - a) * DIM {
            return Err(io::Error::other(format!("{} has the wrong size", path(k).display())));
        }
        vectors.extend(shard);
    }
    let half: Vec<u16> = vectors.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
    std::fs::write(bundle.join("embeddings.bin"), bytemuck::cast_slice(&half))?;
    step("embeddings.bin written");
    Ok(vectors)
}
