//! Making a map from a folder of photos, in this process (no Python): scan, copy, embed with Gemma, lay out, find regions, name them.
//!
//! A map being made is a folder maps/<name>.building/ that keeps everything done so far, so closing the app (or pausing) loses at
//! most a few seconds of work:
//!     job.json      name, source folder, the stage to run next, photo count
//!     files.json    the photos found (scan), in order; photo i is originals/<i, 6 digits>.<ext> (copy)
//!     originals/    hard links to the photos (copies if they are on another disk)
//!     vectors/      the photos' vectors in shards of SHARD photos (embed): a finished shard is never redone
//! When the last stage is done the folder is renamed to maps/<name>.photomap (see data.rs for what is in one).
//! Each stage can be stopped at any time (pause, close) and run again; it picks up what it can.

use crate::embed::{self, Gemma};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Photos per shard of vectors: a pause or close loses at most one shard's work (about 25 s).
pub const SHARD: usize = 256;
/// The photo files a folder is searched for.
pub const TYPES: [&str; 8] = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"];
const DIM: usize = 768;
/// Photos the encoder runs on at once.
const BATCH: usize = 4;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Scan,
    Copy,
    Embed,
    Layout,
    Regions,
    Names,
    Bake,
    Finish,
}

impl Stage {
    pub const ALL: [Stage; 8] = [Stage::Scan, Stage::Copy, Stage::Embed, Stage::Layout, Stage::Regions, Stage::Names, Stage::Bake, Stage::Finish];

    /// What the stage is doing, for the list of steps.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Scan => "Finding photos",
            Stage::Copy => "Copying photos into the map",
            Stage::Embed => "Understanding photos",
            Stage::Layout => "Laying out the map",
            Stage::Regions => "Finding regions",
            Stage::Names => "Naming regions",
            Stage::Bake => "Preparing pictures for viewing",
            Stage::Finish => "Finishing",
        }
    }

    fn next(self) -> Option<Stage> {
        Stage::ALL.get(Stage::ALL.iter().position(|&s| s == self)? + 1).copied()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Job {
    pub name: String,
    /// What the photos come from: folders (their sub-folders too) and single photo files. A job file from before this
    /// had a single folder here as a string.
    #[serde(rename = "source", deserialize_with = "one_or_many")]
    pub sources: Vec<PathBuf>,
    pub created: String,
    /// The stage to run next (everything before it is done).
    pub stage: Stage,
    /// Set by the scan.
    #[serde(default)]
    pub photos: usize,
    /// Every photo's file name is its caption (shown above it, and searched along with the picture).
    #[serde(default)]
    pub names_as_captions: bool,
    /// Use every Nth photo (1: all): a smaller map made quicker, picked evenly through the sorted list.
    #[serde(default = "one")]
    pub every: usize,
}

fn one() -> usize {
    1
}

fn one_or_many<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Vec<PathBuf>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Sources {
        One(PathBuf),
        Many(Vec<PathBuf>),
    }
    Ok(match Sources::deserialize(d)? {
        Sources::One(p) => vec![p],
        Sources::Many(v) => v,
    })
}

/// A few words for where the photos come from: the folder, or how many folders and photos.
pub fn describe(sources: &[PathBuf]) -> String {
    match sources {
        [one] => one.display().to_string(),
        _ => {
            let folders = sources.iter().filter(|p| p.is_dir()).count();
            let photos = sources.len() - folders;
            let part = |n: usize, what: &str| format!("{n} {what}{}", if n == 1 { "" } else { "s" });
            match (folders, photos) {
                (0, p) => part(p, "photo"),
                (f, 0) => part(f, "folder"),
                (f, p) => format!("{} and {}", part(f, "folder"), part(p, "photo")),
            }
        }
    }
}

impl Job {
    pub fn read(dir: &Path) -> Option<Job> {
        serde_json::from_slice(&fs::read(dir.join("job.json")).ok()?).ok()
    }

    pub fn write(&self, dir: &Path) -> io::Result<()> {
        write_atomic(&dir.join("job.json"), &serde_json::to_vec_pretty(self)?)
    }
}

/// Write a file whole or not at all.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)
}

pub fn building_dir(maps: &Path, name: &str) -> PathBuf {
    maps.join(format!("{name}.building"))
}

pub fn finished_dir(maps: &Path, name: &str) -> PathBuf {
    maps.join(format!("{name}.photomap"))
}

/// A name usable as a folder name: letters, digits, spaces, dashes, underscores and dots; at most 60 characters.
pub fn clean_name(name: &str) -> String {
    let kept: String = name.trim().chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect();
    kept.trim_matches('.').chars().take(60).collect::<String>().trim().to_string()
}

/// Start a new build: its folder with job.json. Err with what is wrong (name taken, folder missing).
pub fn create(maps: &Path, name: &str, sources: &[PathBuf], names_as_captions: bool, every: usize) -> Result<PathBuf, String> {
    let name = clean_name(name);
    if name.is_empty() {
        return Err("give the map a name".into());
    }
    if sources.is_empty() || sources.iter().any(|s| !s.exists()) {
        return Err("that folder or photo doesn't exist".into());
    }
    // full paths, links resolved: they are how a photo is recognised later (see reuse.rs)
    let sources: Vec<PathBuf> = sources.iter().map(|s| fs::canonicalize(s).unwrap_or_else(|_| s.clone())).collect();
    if building_dir(maps, &name).exists() || finished_dir(maps, &name).exists() {
        return Err(format!("there is already a map called {name}"));
    }
    let dir = building_dir(maps, &name);
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let created = chrono_like_now();
    Job { name, sources, created, stage: Stage::Scan, photos: 0, names_as_captions, every: every.max(1) }.write(&dir).map_err(|e| e.to_string())?;
    Ok(dir)
}

/// "2026-10-13 21:40" (UTC), without a date library.
fn chrono_like_now() -> String {
    time_text(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64)
}

/// A time (seconds since 1970) as "2026-10-13 21:40" (UTC): the form maps record when they were started, which compares as text.
pub fn time_text(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (d, m) = (doy - (153 * mp + 2) / 5 + 1, if mp < 10 { mp + 3 } else { mp - 9 });
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}", rem / 3600, rem % 3600 / 60)
}

// ---------------------------------------------------------------------------------------------------------------------
// Running a build

#[derive(Clone)]
pub struct State {
    pub stage: Stage,
    pub done: usize,
    pub total: usize,
    pub message: String,
    pub stage_started: Instant,
    /// Where the current piece of work began (time, count done then): speed and time left are measured from here, so a resumed stage is not fooled by what was already done.
    pub base: (Instant, usize),
    /// (time, count done) a few times a second over the last `WINDOW`, for the current speed.
    pub samples: std::collections::VecDeque<(Instant, usize)>,
    pub error: Option<String>,
    pub finished: bool,
}

/// How far back the speed looks.
const WINDOW: std::time::Duration = std::time::Duration::from_secs(30);

impl State {
    /// Items per second over the last 30 seconds (or since the work began, if that is shorter); None until there is 3 s to go by.
    pub fn rate(&self) -> Option<f32> {
        let &(first_at, first_done) = self.samples.front().unwrap_or(&self.base);
        let secs = first_at.elapsed().as_secs_f32();
        (secs >= 3.0 && self.done > first_done).then(|| (self.done - first_done) as f32 / secs)
    }
}

struct Shared {
    pause: AtomicBool,
    state: Mutex<State>,
}

pub struct Run {
    pub dir: PathBuf,
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
}

enum Flow {
    Done,
    Paused,
}

struct Ctx {
    dir: PathBuf,
    root: PathBuf,
    job: Job,
    shared: Arc<Shared>,
    model: Arc<Mutex<Option<Gemma>>>,
}

impl Ctx {
    fn paused(&self) -> bool {
        self.shared.pause.load(Ordering::Relaxed)
    }

    fn progress(&self, done: usize, total: usize, message: impl Into<String>) {
        let mut s = self.shared.state.lock().unwrap();
        s.done = done;
        s.total = total;
        s.base = (Instant::now(), done);
        s.samples.clear();
        s.message = message.into();
    }

    fn set_done(&self, done: usize) {
        let mut s = self.shared.state.lock().unwrap();
        s.done = done;
        let now = Instant::now();
        if s.samples.back().is_none_or(|&(t, _)| now.duration_since(t).as_millis() >= 500) {
            s.samples.push_back((now, done));
        }
        while s.samples.len() > 2 && now.duration_since(s.samples[1].0) >= WINDOW {
            s.samples.pop_front();
        }
    }
}

/// The lock a build holds on its folder while it runs, so that a second window (or a second copy of the program) can't run the
/// same build and write the same files. The system drops it when the build ends, or when the program dies.
fn lock(dir: &Path) -> Option<fs::File> {
    let file = fs::OpenOptions::new().create(true).write(true).truncate(false).open(dir.join(".lock")).ok()?;
    file.try_lock().ok()?;
    Some(file)
}

/// Is something holding the lock of this build's folder? (Its own run counts too: the caller knows its own runs.)
pub fn locked(dir: &Path) -> bool {
    match fs::OpenOptions::new().write(true).open(dir.join(".lock")) {
        Ok(file) => file.try_lock().is_err(),
        Err(_) => false,
    }
}

impl Run {
    /// Run (or go on with) the build in `dir` on a thread. `wake`: called when something changed, to redraw.
    /// None if there is no job in `dir`, or another run of it (in another window) holds it.
    pub fn start(dir: PathBuf, root: PathBuf, model: Arc<Mutex<Option<Gemma>>>, wake: impl Fn() + Send + 'static) -> Option<Run> {
        let job = Job::read(&dir)?;
        let lock = lock(&dir)?;
        let state = State { stage: job.stage, done: 0, total: 0, message: String::new(), stage_started: Instant::now(), base: (Instant::now(), 0), samples: Default::default(), error: None, finished: false };
        let shared = Arc::new(Shared { pause: AtomicBool::new(false), state: Mutex::new(state) });
        let ctx = Ctx { dir: dir.clone(), root, job, shared: shared.clone(), model };
        let handle = std::thread::spawn(move || {
            let _lock = lock; // held until the run is over
            let result = run(ctx);
            let mut s = shared_state(&result.0);
            if let Err(e) = result.1 {
                s.error = Some(e);
            }
            drop(s);
            wake();
        });
        Some(Run { dir, shared: shared.clone(), handle: Some(handle) })
    }

    /// Ask it to stop at the next safe point (a few seconds); `running` turns false when it has.
    pub fn pause(&self) {
        self.shared.pause.store(true, Ordering::Relaxed);
    }

    pub fn running(&self) -> bool {
        self.handle.as_ref().is_some_and(|h| !h.is_finished())
    }

    pub fn state(&self) -> State {
        self.shared.state.lock().unwrap().clone()
    }
}

fn shared_state(shared: &Arc<Shared>) -> std::sync::MutexGuard<'_, State> {
    shared.state.lock().unwrap()
}

/// All the stages from the one the job is at.
fn run(mut ctx: Ctx) -> (Arc<Shared>, Result<(), String>) {
    let shared = ctx.shared.clone();
    let result = (|| {
        loop {
            let stage = ctx.job.stage;
            {
                let mut s = ctx.shared.state.lock().unwrap();
                s.stage = stage;
                s.done = 0;
                s.total = 0;
                s.message.clear();
                s.stage_started = Instant::now();
            }
            let flow = match stage {
                Stage::Scan => scan(&mut ctx)?,
                Stage::Copy => copy(&mut ctx)?,
                Stage::Embed => embed_photos(&mut ctx)?,
                Stage::Layout => lay_out(&mut ctx)?,
                Stage::Regions => find_regions(&mut ctx)?,
                Stage::Names => name_regions(&mut ctx)?,
                Stage::Bake => bake_pages(&mut ctx)?,
                Stage::Finish => finish(&mut ctx)?,
            };
            if matches!(flow, Flow::Paused) {
                return Ok(());
            }
            match stage.next() {
                Some(next) => {
                    ctx.job.stage = next;
                    ctx.job.write(&ctx.dir).map_err(|e| e.to_string())?;
                }
                None => return Ok(()),
            }
        }
    })();
    (shared, result)
}

// ---------------------------------------------------------------------------------------------------------------------
// Stages

#[derive(Serialize, Deserialize)]
struct Found {
    src: PathBuf,
    width: u32,
    height: u32,
}

pub fn is_photo(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| TYPES.contains(&e.to_lowercase().as_str()))
}

/// Every photo among the sources: the photo files themselves, and those in the folders and the folders inside them (not hidden
/// ones). Sorted, each once; with `every` above 1, every Nth of them.
pub fn collect_photos(sources: &[PathBuf], every: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for source in sources {
        if source.is_dir() {
            walk(source, 0, &mut out);
        } else if is_photo(source) {
            out.push(source.clone());
        }
    }
    out.sort();
    out.dedup();
    out.into_iter().step_by(every.max(1)).collect()
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let hidden = entry.file_name().to_string_lossy().starts_with('.');
        if hidden {
            continue;
        }
        if path.is_dir() {
            if depth < 16 {
                walk(&path, depth + 1, out);
            }
        } else if is_photo(&path) {
            out.push(path);
        }
    }
}

/// The size a photo shows at (its EXIF turn applied), from the file's header only.
fn shown_size(path: &Path) -> Option<(u32, u32)> {
    use image::ImageDecoder;
    let mut decoder = image::ImageReader::open(path).ok()?.with_guessed_format().ok()?.into_decoder().ok()?;
    let (w, h) = decoder.dimensions();
    let turned = matches!(
        decoder.orientation().ok()?,
        image::metadata::Orientation::Rotate90 | image::metadata::Orientation::Rotate270 | image::metadata::Orientation::Rotate90FlipH | image::metadata::Orientation::Rotate270FlipH
    );
    Some(if turned { (h, w) } else { (w, h) })
}

fn scan(ctx: &mut Ctx) -> Result<Flow, String> {
    ctx.progress(0, 0, "looking for photos");
    let files = collect_photos(&ctx.job.sources, ctx.job.every);
    let total = files.len();
    ctx.progress(0, total, format!("{total} photo files, checking them"));
    let done = AtomicUsize::new(0);
    let found: Vec<Option<Found>> = files
        .into_par_iter()
        .map(|src| {
            if ctx.paused() {
                return None;
            }
            let size = shown_size(&src);
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 64 == 0 {
                ctx.set_done(d);
            }
            size.map(|(width, height)| Found { src, width, height })
        })
        .collect();
    if ctx.paused() {
        return Ok(Flow::Paused);
    }
    let found: Vec<Found> = found.into_iter().flatten().collect();
    if found.is_empty() {
        return Err("no photos in that folder".into());
    }
    write_atomic(&ctx.dir.join("files.json"), &serde_json::to_vec(&found).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    ctx.job.photos = found.len();
    Ok(Flow::Done)
}

fn read_found(dir: &Path) -> Result<Vec<Found>, String> {
    serde_json::from_slice(&fs::read(dir.join("files.json")).map_err(|e| format!("files.json: {e}"))?).map_err(|e| e.to_string())
}

fn original_name(i: usize, src: &Path) -> String {
    format!("originals/{i:06}.{}", src.extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_lowercase())
}

fn copy(ctx: &mut Ctx) -> Result<Flow, String> {
    let found = read_found(&ctx.dir)?;
    let total = found.len();
    fs::create_dir_all(ctx.dir.join("originals")).map_err(|e| e.to_string())?;
    ctx.progress(0, total, "linking the photos into the map");
    let done = AtomicUsize::new(0);
    let failed = found.par_iter().enumerate().try_for_each(|(i, f)| {
        if ctx.paused() {
            return Err(None);
        }
        let dst = ctx.dir.join(original_name(i, &f.src));
        if !dst.exists() {
            // the same disk: a link costs nothing; another disk: a copy (to a temporary name first: a half copy never counts)
            if fs::hard_link(&f.src, &dst).is_err() {
                let tmp = dst.with_extension("part");
                fs::copy(&f.src, &tmp).and_then(|_| fs::rename(&tmp, &dst)).map_err(|e| Some(format!("{}: {e}", f.src.display())))?;
            }
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 64 == 0 {
            ctx.set_done(d);
        }
        Ok(())
    });
    match failed {
        Err(None) => return Ok(Flow::Paused),
        Err(Some(e)) => return Err(e),
        Ok(()) => {}
    }
    let names = ctx.job.names_as_captions;
    let photos: Vec<serde_json::Value> = found
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let caption = if names { name_as_caption(&f.src) } else { String::new() };
            serde_json::json!({
                "file": original_name(i, &f.src),
                "source": f.src.to_string_lossy(),
                "width": f.width,
                "height": f.height,
                "title": f.src.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
                "caption": caption,
                "meta": {},
            })
        })
        .collect();
    write_atomic(&ctx.dir.join("photos.json"), &serde_json::to_vec(&photos).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

/// A file name read as words: no folders or extension, separators as spaces ("red_fox-02.jpg" -> "red fox 02").
fn name_as_caption(path: &Path) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    stem.split(|c: char| c == '_' || c == '-' || c == '.' || c.is_whitespace()).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" ")
}

fn shard_path(dir: &Path, k: usize) -> PathBuf {
    dir.join("vectors").join(format!("{k:05}.f16"))
}

fn shard_len(photos: usize, k: usize) -> usize {
    (photos - k * SHARD).min(SHARD)
}

fn embed_photos(ctx: &mut Ctx) -> Result<Flow, String> {
    let n = ctx.job.photos;
    let shards = n.div_ceil(SHARD);
    fs::create_dir_all(ctx.dir.join("vectors")).map_err(|e| e.to_string())?;
    let finished = |k: usize| fs::metadata(shard_path(&ctx.dir, k)).is_ok_and(|m| m.len() as usize == shard_len(n, k) * DIM * 2);
    let todo: Vec<usize> = (0..shards).filter(|&k| !finished(k)).collect();
    let already = n - todo.iter().map(|&k| shard_len(n, k)).sum::<usize>();
    ctx.progress(already, n, "waiting for the model");

    // the model is loaded by the search engine at start-up; wait for it
    let waited = Instant::now();
    while ctx.model.lock().unwrap().is_none() {
        if ctx.paused() {
            return Ok(Flow::Paused);
        }
        if waited.elapsed().as_secs() > 180 {
            return Err("the model didn't load".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    // photos an earlier map already has vectors for (the same file, unchanged): copied, not embedded
    ctx.progress(already, n, "looking for photos embedded before");
    let found = read_found(&ctx.dir)?;
    let mut in_todo = vec![false; n];
    for &k in &todo {
        in_todo[k * SHARD..k * SHARD + shard_len(n, k)].fill(true);
    }
    let wanted: std::collections::HashSet<PathBuf> = (0..n).filter(|&i| in_todo[i]).map(|i| found[i].src.clone()).collect();
    let earlier = crate::reuse::Found::open(ctx.dir.parent().ok_or("no maps folder")?, &wanted);
    drop(wanted);
    let hits: Vec<Option<crate::reuse::Hit>> = (0..n).into_par_iter().map(|i| if in_todo[i] { earlier.find(&found[i].src) } else { None }).collect();
    let reused: Arc<Vec<bool>> = Arc::new(hits.iter().map(Option::is_some).collect());
    let reused_count = reused.iter().filter(|&&r| r).count();
    // the reused photos cost nothing: they count as done from the start, so the speed and the time left are about the photos the model does
    ctx.progress(already + reused_count, n, if reused_count > 0 { format!("{reused_count} photos were embedded in earlier maps; embedding the rest") } else { "reading the photos and embedding them".to_string() });

    // CPU side ahead of the GPU: the next shard's photos are decoded and cut into patches while this one is embedded
    let photos: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(ctx.dir.join("photos.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let files: Vec<PathBuf> = photos.iter().map(|p| ctx.dir.join(p["file"].as_str().unwrap_or_default())).collect();
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Option<embed::Patches>>)>(1);
    let stop = Arc::new(AtomicBool::new(false));
    let producer = {
        let (todo, files, stop, reused) = (todo.clone(), files.clone(), stop.clone(), reused.clone());
        std::thread::spawn(move || {
            for k in todo {
                let range = k * SHARD..k * SHARD + shard_len(n, k);
                let cuts: Vec<Option<embed::Patches>> = range
                    .into_par_iter()
                    .map(|i| if stop.load(Ordering::Relaxed) || reused[i] { None } else { crate::photo::load(&files[i]).and_then(|img| embed::patches(&img).ok()) })
                    .collect();
                if stop.load(Ordering::Relaxed) || tx.send((k, cuts)).is_err() {
                    return;
                }
            }
        })
    };

    let mut processed = 0usize; // photos that were not reused, seen to so far
    let mut outcome = Ok(Flow::Done);
    'shards: for (k, mut cuts) in rx {
        let mut vectors = vec![0f32; cuts.len() * DIM];
        for i in 0..cuts.len() {
            if let Some(hit) = hits[k * SHARD + i] {
                for (slot, b) in vectors[i * DIM..(i + 1) * DIM].iter_mut().zip(earlier.row(hit).chunks_exact(2)) {
                    *slot = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
        }
        let mut at = 0;
        while at < cuts.len() {
            if ctx.paused() {
                outcome = Ok(Flow::Paused);
                break 'shards;
            }
            let end = (at + BATCH).min(cuts.len());
            let group: Vec<(usize, embed::Patches)> = (at..end).filter_map(|i| cuts[i].take().map(|p| (i, p))).collect();
            if !group.is_empty() {
                let (idx, patches): (Vec<usize>, Vec<embed::Patches>) = group.into_iter().unzip();
                let results = match ctx.model.lock().unwrap().as_mut() {
                    Some(m) => m.embed_patches(patches),
                    None => idx.iter().map(|_| Err("the model is gone".to_string())).collect(),
                };
                for (i, r) in idx.into_iter().zip(results) {
                    match r {
                        Ok(v) => vectors[i * DIM..(i + 1) * DIM].copy_from_slice(&v),
                        Err(e) if e.contains("running") || e.contains("loading") => {
                            outcome = Err(e);
                            break 'shards;
                        }
                        Err(_) => {} // a photo it couldn't embed keeps a zero vector
                    }
                }
            }
            processed += (at..end).filter(|&i| hits[k * SHARD + i].is_none()).count();
            at = end;
            ctx.set_done(already + reused_count + processed);
        }
        let bytes: Vec<u8> = vectors.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect();
        write_atomic(&shard_path(&ctx.dir, k), &bytes).map_err(|e| e.to_string())?;
    }
    stop.store(true, Ordering::Relaxed);
    let _ = producer.join();
    let flow = outcome?;
    if matches!(flow, Flow::Paused) {
        return Ok(flow);
    }

    // all the vectors in one file; a photo it couldn't read has the average vector (it lands in the middle of the map)
    let mut all = Vec::with_capacity(n * DIM * 2);
    for k in 0..shards {
        all.extend(fs::read(shard_path(&ctx.dir, k)).map_err(|e| e.to_string())?);
    }
    write_atomic(&ctx.dir.join("embeddings.bin"), &all).map_err(|e| e.to_string())?;
    if ctx.job.names_as_captions {
        return embed_captions(ctx);
    }
    Ok(Flow::Done)
}

/// captions.bin: the vector of every photo's caption (zeros for a photo with none). Written in pieces as it goes, so a pause loses little.
fn embed_captions(ctx: &mut Ctx) -> Result<Flow, String> {
    let n = ctx.job.photos;
    let target = ctx.dir.join("captions.bin");
    if fs::metadata(&target).is_ok_and(|m| m.len() as usize == n * DIM * 2) {
        return Ok(Flow::Done);
    }
    let photos: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(ctx.dir.join("photos.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let part = ctx.dir.join("captions.part");
    let row = DIM * 2;
    // go on from the rows already written
    let have = (fs::metadata(&part).map_or(0, |m| m.len() as usize) / row).min(n);
    let mut file = fs::OpenOptions::new().create(true).append(true).open(&part).map_err(|e| e.to_string())?;
    file.set_len((have * row) as u64).map_err(|e| e.to_string())?;
    ctx.progress(have, n, "reading the captions");
    for chunk_start in (have..n).step_by(64) {
        if ctx.paused() {
            return Ok(Flow::Paused);
        }
        let mut bytes = Vec::with_capacity(64 * row);
        for photo in &photos[chunk_start..(chunk_start + 64).min(n)] {
            let text = photo["caption"].as_str().unwrap_or("");
            let vector = if text.is_empty() { None } else { ctx.model.lock().unwrap().as_mut().ok_or("the model is gone")?.embed_document(text).ok() };
            match vector {
                Some(v) => bytes.extend(v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes())),
                None => bytes.extend(std::iter::repeat_n(0u8, row)),
            }
        }
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        ctx.set_done((chunk_start + 64).min(n));
    }
    drop(file);
    fs::rename(&part, &target).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

fn read_vectors(dir: &Path, n: usize) -> Result<Vec<f32>, String> {
    let raw = fs::read(dir.join("embeddings.bin")).map_err(|e| format!("embeddings.bin: {e}"))?;
    if raw.len() != n * DIM * 2 {
        return Err("embeddings.bin is not the size of the photos".into());
    }
    Ok(raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect())
}

fn read_layout(dir: &Path, n: usize) -> Result<Vec<f32>, String> {
    let raw = fs::read(dir.join("layout.bin")).map_err(|e| format!("layout.bin: {e}"))?;
    if raw.len() != n * 8 {
        return Err("layout.bin is not the size of the photos".into());
    }
    Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn lay_out(ctx: &mut Ctx) -> Result<Flow, String> {
    let n = ctx.job.photos;
    ctx.progress(0, 0, "preparing the vectors");
    let vectors = ndarray::Array2::from_shape_vec((n, DIM), read_vectors(&ctx.dir, n)?).map_err(|e| e.to_string())?;
    let xy = crate::layout::layout(vectors, &|what, done, total| ctx.progress(done, total, what), &|| ctx.paused())?;
    let Some(xy) = xy else { return Ok(Flow::Paused) };
    let bytes: Vec<u8> = xy.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_atomic(&ctx.dir.join("layout.bin"), &bytes).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

fn find_regions(ctx: &mut Ctx) -> Result<Flow, String> {
    let xy = read_layout(&ctx.dir, ctx.job.photos)?;
    let levels = crate::regions::cluster(&xy, crate::data::WORLD, &|what, done, total| ctx.progress(done, total, what), &|| ctx.paused());
    let Some(levels) = levels else { return Ok(Flow::Paused) };
    let mut bytes: Vec<u8> = Vec::new();
    for v in [levels.len() as u32, ctx.job.photos as u32] {
        bytes.extend(v.to_le_bytes());
    }
    for level in &levels {
        bytes.extend(level.iter().flat_map(|v| v.to_le_bytes()));
    }
    write_atomic(&ctx.dir.join("levels.bin"), &bytes).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

fn name_regions(ctx: &mut Ctx) -> Result<Flow, String> {
    let n = ctx.job.photos;
    let raw = fs::read(ctx.dir.join("levels.bin")).map_err(|e| format!("levels.bin: {e}"))?;
    let words: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let (count, photos) = (words[0] as usize, words[1] as usize);
    if photos != n || words.len() != 2 + count * n {
        return Err("levels.bin is not the size of the photos".into());
    }
    let levels: Vec<Vec<u32>> = (0..count).map(|l| words[2 + l * n..2 + (l + 1) * n].to_vec()).collect();
    let phrases = crate::naming::phrases(&ctx.root)?;

    // the vocabulary's vectors: cached after the first map; made with the shared model otherwise
    ctx.progress(0, phrases.len(), "reading the vocabulary");
    let waited = Instant::now();
    while ctx.model.lock().unwrap().is_none() {
        if ctx.paused() {
            return Ok(Flow::Paused);
        }
        if waited.elapsed().as_secs() > 180 {
            return Err("the model didn't load".into());
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    let model = ctx.model.clone();
    let mut embed = |text: &str| match model.lock().unwrap().as_mut() {
        Some(m) => m.embed_query(text),
        None => Err("the model is gone".to_string()),
    };
    let vocabulary = crate::naming::vectors(&ctx.root, &phrases, &mut embed, &|done, total| ctx.progress(done, total, "learning the vocabulary (first map only)"), &|| ctx.paused())?;
    let Some(vocabulary) = vocabulary else { return Ok(Flow::Paused) };

    ctx.progress(0, 0, "naming the regions");
    let vectors = read_vectors(&ctx.dir, n)?;
    let xy = read_layout(&ctx.dir, n)?;
    let named = crate::naming::name_levels(&vectors, &xy, &levels, &phrases, &vocabulary);
    let json = serde_json::to_vec(&serde_json::json!({ "levels": named })).map_err(|e| e.to_string())?;
    write_atomic(&ctx.dir.join("region_names.json"), &json).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

/// The small pictures the viewer draws from (bake.rs), made now so that the finished map opens at once.
fn bake_pages(ctx: &mut Ctx) -> Result<Flow, String> {
    let photos: Vec<crate::data::Photo> = serde_json::from_slice(&fs::read(ctx.dir.join("photos.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let progress = crate::data::Progress::default();
    let finished = AtomicBool::new(false);
    let result = std::thread::scope(|scope| {
        scope.spawn(|| {
            while !finished.load(Ordering::Relaxed) {
                let (text, fraction) = progress.now();
                ctx.progress((fraction.unwrap_or(0.0) * 1000.0) as usize, 1000, text);
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        });
        let result = crate::bake::bake(&ctx.dir, &photos, &progress);
        finished.store(true, Ordering::Relaxed);
        result
    });
    result.map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

fn finish(ctx: &mut Ctx) -> Result<Flow, String> {
    let manifest = serde_json::json!({
        "format": 1,
        "name": ctx.job.name,
        "created": ctx.job.created,
        "photos": ctx.job.photos,
        "world": crate::data::WORLD as u32,
        "embedding": { "model": "google/embeddinggemma-2", "dim": DIM, "dtype": "float16", "input": "image" },
    });
    write_atomic(&ctx.dir.join("manifest.json"), &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let maps = ctx.dir.parent().ok_or("no maps folder")?;
    let target = finished_dir(maps, &ctx.job.name);
    fs::remove_dir_all(ctx.dir.join("vectors")).map_err(|e| e.to_string())?;
    let _ = fs::remove_file(ctx.dir.join("files.json"));
    let _ = fs::remove_file(ctx.dir.join("levels.bin"));
    let _ = fs::remove_file(ctx.dir.join("job.json"));
    let _ = fs::remove_file(ctx.dir.join(".lock"));
    fs::rename(&ctx.dir, &target).map_err(|e| e.to_string())?;
    ctx.shared.state.lock().unwrap().finished = true;
    Ok(Flow::Done)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A build's folder is held while it runs: a second run can't start on it, and it is free again afterwards.
    #[test]
    fn a_build_folder_has_one_runner() {
        let dir = std::env::temp_dir().join(format!("photo-map-lock-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        assert!(!locked(&dir), "nobody holds a new folder");
        let held = lock(&dir).expect("the first one gets it");
        assert!(locked(&dir));
        assert!(lock(&dir).is_none(), "a second one does not");
        drop(held);
        assert!(!locked(&dir));
        assert!(lock(&dir).is_some(), "free again once the first is over");
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Photos come from folders and single files, each once; a job file from before the list reads as one folder.
    #[test]
    fn sources_can_be_folders_and_files() {
        let temp = std::env::temp_dir().join(format!("photo-map-sources-test-{}", std::process::id()));
        fs::create_dir_all(temp.join("a/inner")).unwrap();
        fs::create_dir_all(temp.join("b")).unwrap();
        for f in ["a/1.jpg", "a/inner/2.png", "a/notes.txt", "b/3.jpg", "b/4.jpg"] {
            fs::write(temp.join(f), b"x").unwrap();
        }
        let found = collect_photos(&[temp.join("a"), temp.join("a/1.jpg"), temp.join("b/3.jpg"), temp.join("b/notes.txt")], 1);
        let names: Vec<String> = found.iter().map(|p| p.strip_prefix(&temp).unwrap().display().to_string()).collect();
        assert_eq!(names, ["a/1.jpg", "a/inner/2.png", "b/3.jpg"]);
        let all = [temp.join("a"), temp.join("b")];
        assert_eq!(collect_photos(&all, 1).len(), 4);
        assert_eq!(collect_photos(&all, 2).len(), 2, "every second one");
        assert_eq!(collect_photos(&all, 3).len(), 2, "the first, then the fourth");
        assert_eq!(describe(&[temp.join("a"), temp.join("b/3.jpg"), temp.join("b/4.jpg")]), "1 folder and 2 photos");
        let old: Job = serde_json::from_str(r#"{"name":"x","source":"/photos","created":"","stage":"scan"}"#).unwrap();
        assert_eq!(old.sources, [PathBuf::from("/photos")]);
        let new: Job = serde_json::from_str(r#"{"name":"x","source":["/a","/b/c.jpg"],"created":"","stage":"scan"}"#).unwrap();
        assert_eq!(new.sources.len(), 2);
        fs::remove_dir_all(&temp).unwrap();
    }

    /// A photo an earlier map already holds a vector for (same file, not changed since) is not embedded again: its row is copied.
    /// Proved by changing a row in the first map to a vector the model would never make, and finding it in the second.
    #[test]
    fn reuses_vectors_of_earlier_maps() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let temp = std::env::temp_dir().join(format!("photo-map-reuse-test-{}", std::process::id()));
        let (source, maps) = (temp.join("photos"), temp.join("maps"));
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&maps).unwrap();
        let hour_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        for i in 0..24 {
            let img = image::RgbImage::from_fn(96 + (i % 4) as u32 * 8, 80, |x, y| image::Rgb([(i * 9) as u8, (x * 2) as u8, (y * 3 + i as u32) as u8]));
            let path = source.join(format!("p{i:02}.png"));
            img.save(&path).unwrap();
            fs::File::options().write(true).open(&path).unwrap().set_modified(hour_ago).unwrap(); // older than the map started
        }
        let model = embed::shared_model();
        let build = |name: &str| {
            let dir = create(&maps, name, &[source.clone()], false, 1).unwrap();
            let run = Run::start(dir, root.clone(), model.clone(), || {}).unwrap();
            while run.running() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            assert!(run.state().error.is_none(), "{:?}", run.state().error);
            finished_dir(&maps, name)
        };
        let first = build("first");
        let photos: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(first.join("photos.json")).unwrap()).unwrap();
        assert!(Path::new(photos[0]["source"].as_str().unwrap()).is_absolute() && photos[0]["source"].as_str().unwrap().ends_with(".png"), "the full path of the file");

        // the first map is the only one the second can reuse from: its row 0 becomes a vector the model would not give
        let mut raw = fs::read(first.join("embeddings.bin")).unwrap();
        let marker: Vec<u8> = (0..DIM).flat_map(|d| half::f16::from_f32(if d == 0 { 1.0 } else { 0.0 }).to_le_bytes()).collect();
        raw[..DIM * 2].copy_from_slice(&marker);
        fs::write(first.join("embeddings.bin"), &raw).unwrap();

        let second = build("second");
        let again = fs::read(second.join("embeddings.bin")).unwrap();
        assert_eq!(&again[..DIM * 2], &marker[..], "row 0 was copied from the first map");
        assert_eq!(&again[DIM * 2..], &raw[DIM * 2..], "and the others");

        // a file changed since the first map was started is embedded again
        let changed = first_source_path(&photos);
        fs::File::options().write(true).open(&changed).unwrap().set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(3600)).unwrap();
        let third = build("third");
        let last = fs::read(third.join("embeddings.bin")).unwrap();
        assert_ne!(&last[..DIM * 2], &marker[..], "a changed file is embedded again");
        fs::remove_dir_all(&temp).unwrap();

        fn first_source_path(photos: &[serde_json::Value]) -> PathBuf {
            PathBuf::from(photos[0]["source"].as_str().unwrap())
        }
    }

    /// A folder of photos becomes a map's vectors, also when the build is paused and continued several times on the way.
    #[test]
    fn builds_from_a_folder_and_resumes() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
        let temp = std::env::temp_dir().join(format!("photo-map-build-test-{}", std::process::id()));
        let (source, maps) = (temp.join("photos"), temp.join("maps"));
        fs::create_dir_all(source.join("sub")).unwrap();
        fs::create_dir_all(&maps).unwrap();
        let count = SHARD + 44; // two shards
        for i in 0..count {
            let dir = if i % 3 == 0 { source.join("sub") } else { source.clone() };
            let img = image::RgbImage::from_fn(96 + (i % 5) as u32 * 8, 80, |x, y| image::Rgb([(i * 7) as u8, (x * 2) as u8, (y * 3 + i as u32) as u8]));
            img.save(dir.join(format!("p{i:04}.png"))).unwrap();
        }
        fs::write(source.join("notes.txt"), "not a photo").unwrap();

        let model = embed::shared_model();
        let dir = create(&maps, "test map", &[source.clone()], true, 1).unwrap();
        assert!(create(&maps, "test map", &[source.clone()], true, 1).is_err(), "a second map with the same name");
        // stop and go on a few times at arbitrary moments
        for pause_after_ms in [0u64, 150, 900, 2500] {
            let run = Run::start(dir.clone(), root.clone(), model.clone(), || {}).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(pause_after_ms));
            run.pause();
            while run.running() {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            assert!(run.state().error.is_none(), "{:?}", run.state().error);
            if !dir.exists() {
                break; // already finished
            }
        }
        if dir.exists() {
            let run = Run::start(dir.clone(), root.clone(), model.clone(), || {}).unwrap();
            while run.running() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            assert!(run.state().error.is_none(), "{:?}", run.state().error);
        }
        let done = finished_dir(&maps, "test map");
        assert!(done.join("manifest.json").exists(), "the map folder: {:?}", fs::read_dir(&maps).unwrap().flatten().map(|e| e.file_name()).collect::<Vec<_>>());
        assert_eq!(fs::metadata(done.join("embeddings.bin")).unwrap().len() as usize, count * DIM * 2);
        let photos: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(done.join("photos.json")).unwrap()).unwrap();
        assert_eq!(photos.len(), count);
        assert!(photos.iter().all(|p| p["caption"] == p["title"]), "every photo's caption is its file name");
        assert_eq!(fs::metadata(done.join("captions.bin")).unwrap().len() as usize, count * DIM * 2);
        assert!(done.join(photos[count - 1]["file"].as_str().unwrap()).exists());
        // the vectors are unit length and not all alike
        let raw = fs::read(done.join("embeddings.bin")).unwrap();
        let vector = |i: usize| -> Vec<f32> { raw[i * DIM * 2..(i + 1) * DIM * 2].chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect() };
        let (a, b) = (vector(0), vector(count - 1));
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        eprintln!("norm {norm:.3}, cosine between the first and last photo {cos:.3}");
        assert!((norm - 1.0).abs() < 0.01 && cos < 0.999);
        // laid out, with named regions, and the viewer's loader takes it
        assert_eq!(fs::metadata(done.join("layout.bin")).unwrap().len() as usize, count * 8);
        let names: serde_json::Value = serde_json::from_slice(&fs::read(done.join("region_names.json")).unwrap()).unwrap();
        let levels = names["levels"].as_array().unwrap();
        let named: usize = levels.iter().map(|l| l.as_array().unwrap().len()).sum();
        eprintln!("{} levels, {named} named regions, e.g. {:?}", levels.len(), levels.iter().flat_map(|l| l.as_array().unwrap().iter().take(2).map(|r| r["name"].as_str().unwrap().to_string())).collect::<Vec<_>>());
        assert!(named > 0);
        assert!(done.join("pages_bc7.bin").exists(), "baked before the map is finished");
        let data = crate::data::MapData::load(&done, &crate::data::Progress::default()).unwrap();
        assert_eq!(data.n(), count);
        fs::remove_dir_all(&temp).unwrap();
    }
}
