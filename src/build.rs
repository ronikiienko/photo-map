//! Making a map from a folder of photos, in this process (no Python): read the videos, find the photos, embed, lay out,
//! find regions, name them, bake.
//!
//! A map being made is a folder maps/<name>.building/ that keeps everything done so far, so closing the app (or pausing) loses at
//! most a few seconds of work. Besides the map's own files (data.rs), as each stage makes them, it holds:
//!     job.json      name, sources, the stage to run next, photo count
//!     files.json    the photos and videos found (scan), in order: they are the map's items
//!     videos/<k>/   the kth video's frames (0.jpg, 1.jpg...), times.txt, video.json, facts.json, poster.jpg: a video with
//!                   them is never redone. They become frames/<i>/.
//!     pictures/     the vectors of the pictures (photos and videos' frames) in shards of SHARD (embed): a finished shard is never redone;
//!                   beside each, the same pictures at 32 px (<k>.tiles, bake::tile_of), for the bake, which then deletes them
//!     videos/<k>/small/  its frames at FRAME_SIDE, made by embed from the frames it decodes; put in their place by the bake
//!     levels.bin    the regions of every level (regions), until they are named
//! When the last stage is done the folder is renamed to maps/<name>.photomap.
//! Each stage can be stopped at any time (pause, close) and run again; it picks up what it can.

use crate::models::{self, Hub, Prepared, Spec};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

/// Pictures per shard of vectors: a pause or close loses at most one shard's work (a few seconds).
pub const SHARD: usize = 256;
/// The photo files a folder is searched for.
pub const TYPES: [&str; 8] = ["jpg", "jpeg", "png", "webp", "gif", "bmp", "tif", "tiff"];
/// The video files a folder is searched for: each is one photo on the map (its frames are read with ffmpeg, see `video_frames`).
pub const VIDEO_TYPES: [&str; 7] = ["mp4", "mov", "m4v", "mkv", "webm", "avi", "3gp"];
/// A video's frames are its keyframes at least this far apart (seconds): about one every 2 s, as many as its length asks.
const FRAME_GAP: f32 = 2.0;
/// Pictures handed to the model at once.
const BATCH: usize = 16;

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Videos,
    Scan,
    Embed,
    Layout,
    Tree,
    Regions,
    Names,
    Bake,
    Finish,
}

impl Stage {
    pub const ALL: [Stage; 9] = [Stage::Videos, Stage::Scan, Stage::Embed, Stage::Layout, Stage::Tree, Stage::Regions, Stage::Names, Stage::Bake, Stage::Finish];

    /// What the stage is doing, for the list of steps.
    pub fn label(self) -> &'static str {
        match self {
            Stage::Videos => "Reading videos",
            Stage::Scan => "Finding photos",
            Stage::Embed => "Understanding photos",
            Stage::Layout => "Laying out the map",
            Stage::Tree => "Linking similar photos",
            Stage::Regions => "Finding regions",
            Stage::Names => "Naming regions",
            Stage::Bake => "Preparing pictures for viewing",
            Stage::Finish => "Finishing",
        }
    }

    /// What the stage does and why, in plain words, for the card of a map being made.
    pub fn explain(self) -> &'static str {
        match self {
            Stage::Videos => "Takes still pictures out of each video (MP4, MOV, MKV and so on) with ffmpeg: about one every 2 seconds. Nothing to do if there are no videos.",
            Stage::Scan => "Looks through the folders you gave for photo files (JPEG, PNG, WebP and so on), counts them and notes where each one is: the map reads them there, and takes no space for them. Nothing is changed in your folders.",
            Stage::Embed => "The model looks at every photo and turns it into a vector: hundreds of numbers that describe what is in the picture. Photos of similar things get similar numbers, and that is all the map uses to decide what belongs together. A video's frames each go through the model: the video is shown by its most typical frame (the one closest to their average), placed on the map by that frame's vector, and a search finds it by its best-matching frame. This is the slow part, because every photo goes through the model on the graphics card.",
            Stage::Layout => "Hundreds of numbers can't be drawn on a screen, so they are squeezed down to 2 (x and y) in a way that keeps photos with similar vectors close together. It first finds each photo's 10 nearest neighbours, then arranges the photos in the plane so that neighbours stay neighbours (the method is called UMAP). Last, photos that would cover each other are moved apart a little, so that every photo can be seen.",
            Stage::Tree => "Joins all the photos into one tree by the shortest links between them (the most similar pairs). A path between two photos (right-click one on the map) follows this tree: of all the ways from one to the other, it is the one whose biggest change from photo to photo is the smallest. Each photo is compared with the photos in the parts of the map nearest to it, not with all of them.",
            Stage::Regions => "Groups the photos on the map into areas, from a few big ones down to small ones of about 10 photos. The groups are made from neighbours on the map, so each area is one connected piece. Zooming in shows the finer ones.",
            Stage::Names => "Gives each area a name. About 3,400 common phrases (\"beach\", \"cathedral\", \"vegetable garden\"...) are run through the same model; the phrase whose vector is closest to the area's average vector is its name.",
            Stage::Bake => "Makes small copies of every photo at several sizes (from 32 px down to 1 px) and packs them into big pictures the graphics card can draw directly. This is what lets the viewer show thousands of photos at once, and it is done now so the map opens at once.",
            Stage::Finish => "Writes the map's description, tidies away the working files and moves the map to the list.",
        }
    }

    fn next(self) -> Option<Stage> {
        Stage::ALL.get(Stage::ALL.iter().position(|&s| s == self)? + 1).copied()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Job {
    pub name: String,
    /// What the photos come from: folders (their sub-folders too) and single photo files.
    pub sources: Vec<PathBuf>,
    pub created: String,
    /// The stage to run next (everything before it is done).
    pub stage: Stage,
    /// Set by the scan.
    pub photos: usize,
    /// Use every Nth photo (1: all): a smaller map made quicker, picked evenly through the sorted list.
    pub every: usize,
    /// The model that makes the photos' vectors (models.rs). Fixed for the life of the job: its vectors, and the map's search, are
    /// only good with the same one.
    pub model: String,
    /// A photo's longer side on the map (world units), set by the layout: the photos are spread to it (data::spread).
    pub photo_side: Option<f32>,
    /// The videos ffmpeg couldn't read (their file names), left out of the map: set by the Videos step.
    pub unread_videos: Vec<String>,
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
pub fn create(maps: &Path, name: &str, sources: &[PathBuf], every: usize, model: &str) -> Result<PathBuf, String> {
    if models::spec(model).is_none() {
        return Err(format!("unknown model {model}"));
    }
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
    Job { name, sources, created, stage: Stage::Videos, photos: 0, every: every.max(1), model: model.to_string(), photo_side: None, unread_videos: Vec::new() }.write(&dir).map_err(|e| e.to_string())?;
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
    /// What `done` and `total` count: "photos", or "pictures" when a video's frames count one each.
    pub unit: &'static str,
    pub message: String,
    pub stage_started: Instant,
    /// Where the current piece of work began (time, count done then): speed and time left are measured from here, so a resumed stage is not fooled by what was already done.
    pub base: (Instant, usize),
    /// (time, count done) a few times a second over the last `WINDOW`, for the current speed.
    pub samples: std::collections::VecDeque<(Instant, usize)>,
    pub error: Option<String>,
    pub finished: bool,
    /// The photos the model finished last, newest at the back, as small squares (id, pixels), for the card to show.
    pub recent: std::collections::VecDeque<(u64, Arc<image::RgbaImage>)>,
    /// The first few of the 768 numbers the model gave the last photo.
    pub vector_head: Vec<f32>,
    /// The videos that couldn't be read (Job::unread_videos).
    pub unread_videos: Vec<String>,
}

/// How many numbers of a vector the card shows.
pub const HEAD: usize = 8;

/// How many of the latest photos the card shows.
pub const RECENT: usize = 9;
/// A thumbnail's side.
const THUMB: u32 = 64;

/// The middle square of a photo, small.
pub(crate) fn thumbnail(img: &image::RgbaImage) -> image::RgbaImage {
    use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::{Image, ImageRef}};
    let side = img.width().min(img.height());
    let (x, y) = ((img.width() - side) / 2, (img.height() - side) / 2);
    let from = ImageRef::new(img.width(), img.height(), img.as_raw(), PixelType::U8x4).expect("the buffer is the photo's");
    let mut to = Image::new(THUMB, THUMB, PixelType::U8x4);
    // the middle square, shrunk with the box filter (every source pixel counts equally, like `imageops::thumbnail`)
    let options = ResizeOptions::new().crop(x as f64, y as f64, side as f64, side as f64).resize_alg(ResizeAlg::Convolution(FilterType::Box));
    Resizer::new().resize(&from, &mut to, &options).expect("same pixel type");
    image::RgbaImage::from_raw(THUMB, THUMB, to.into_vec()).expect("THUMB squared pixels")
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
    hub: Arc<Hub>,
    spec: &'static Spec,
}

impl Ctx {
    /// How many numbers each of this build's vectors has.
    fn dim(&self) -> usize {
        self.spec.dim
    }

    /// Wait for the build's model to be loaded (the search engine's thread loads it). Ok(false) if paused while waiting.
    fn wait_for_model(&self) -> Result<bool, String> {
        self.hub.wait_for(&self.job.model, &|| self.paused())
    }

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

    /// Photos the model has just finished: the card shows them going by.
    fn show_recent(&self, thumbs: impl IntoIterator<Item = image::RgbaImage>, head: Option<Vec<f32>>) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let mut s = self.shared.state.lock().unwrap();
        for t in thumbs {
            s.recent.push_back((NEXT.fetch_add(1, Ordering::Relaxed), Arc::new(t)));
        }
        while s.recent.len() > RECENT {
            s.recent.pop_front();
        }
        if let Some(head) = head {
            s.vector_head = head;
        }
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
    pub fn start(dir: PathBuf, root: PathBuf, hub: Arc<Hub>, wake: impl Fn() + Send + 'static) -> Option<Run> {
        let job = Job::read(&dir)?;
        let spec = models::spec(&job.model)?;
        let lock = lock(&dir)?;
        let state = State { stage: job.stage, done: 0, total: 0, unit: "photos", message: String::new(), stage_started: Instant::now(), base: (Instant::now(), 0), samples: Default::default(), error: None, finished: false, recent: Default::default(), vector_head: Vec::new(), unread_videos: job.unread_videos.clone() };
        let shared = Arc::new(Shared { pause: AtomicBool::new(false), state: Mutex::new(state) });
        let ctx = Ctx { dir: dir.clone(), root, job, shared: shared.clone(), hub, spec };
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
                s.unit = "photos";
                s.message.clear();
                s.stage_started = Instant::now();
            }
            let flow = match stage {
                Stage::Videos => read_videos(&mut ctx)?,
                Stage::Scan => scan(&mut ctx)?,
                Stage::Embed => embed_photos(&mut ctx)?,
                Stage::Layout => lay_out(&mut ctx)?,
                Stage::Tree => make_tree(&mut ctx)?,
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
    /// A video: the folder its frames were taken out to (videos/<k>), where its poster.jpg goes too.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frames: Option<String>,
    /// A video: how long it is and its own size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    video: Option<crate::data::Video>,
}

/// A photo or a video file.
pub fn is_photo(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        let e = e.to_lowercase();
        TYPES.contains(&e.as_str()) || VIDEO_TYPES.contains(&e.as_str())
    })
}

pub fn is_video(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| VIDEO_TYPES.contains(&e.to_lowercase().as_str()))
}

/// Takes a video's keyframes out to JPEGs in `dir` (0.jpg, 1.jpg...; the longer side at most data::POSTER_SIDE, data::poster_size). A keyframe is a whole
/// picture (the frames between only say what changed), so only they are read (-discard: the others are dropped as the file is
/// read, never decoded): encoders put one every 0.5-2 s or so, and at most cuts. Of them, one at least FRAME_GAP after the last
/// one kept is decoded (the others are dropped before the decoder: decoding a big keyframe is most of the work); where they are
/// further apart, each. Their times (seconds from the start) go to times.txt, a line each, in
/// order; how long the video is and its size (data::Video) to video.json, what its file says about itself (data::Facts) to facts.json. Done once: a folder already there is kept. Needs ffmpeg. `stop`: asked while ffmpeg runs; true stops it (the folder isn't
/// made, Ok(false)). `at`: told how far into the video it has got (seconds), as it goes. Ok(true): done.
fn video_frames(ffmpeg: &Path, video: &Path, dir: &Path, stop: &dyn Fn() -> bool, at: &dyn Fn(f32)) -> Result<bool, String> {
    if dir.is_dir() {
        return Ok(true);
    }
    let part = dir.with_extension("part");
    let _ = fs::remove_dir_all(&part);
    fs::create_dir_all(&part).map_err(|e| e.to_string())?;
    // what ffmpeg says goes to a file (a pipe nobody reads while it runs would fill up and stall it)
    let log_path = part.join("ffmpeg.log");
    let log_file = fs::File::create(&log_path).map_err(|e| e.to_string())?;
    let mut child = std::process::Command::new(ffmpeg)
        .args(["-nostdin", "-hide_banner", "-loglevel", "info", "-discard", "nokey", "-skip_frame", "nokey"])
        // a keyframe is dropped unless it is the first or FRAME_GAP after the last one kept (its time in register 0)
        .arg("-bsf:v")
        .arg(format!("noise=drop='if(eq(n\\,0)+gte(if(isnan(pts)\\,dts\\,pts)*tb-ld(0)\\,{})\\,st(0\\,if(isnan(pts)\\,dts\\,pts)*tb)*0\\,1)'", FRAME_GAP - 0.05))
        .arg("-i")
        .arg(video)
        .args(["-map", "0:V:0"])
        // the longer side at most POSTER_SIDE (the other follows, kept even)
        .arg("-vf")
        .arg(format!("showinfo,scale='if(gte(iw,ih),min({0},iw),-2)':'if(gte(iw,ih),-2,min({0},ih))'", crate::data::POSTER_SIDE))
        .args(["-fps_mode", "passthrough", "-q:v", "3", "-start_number", "0"])
        .arg(part.join("%d.jpg"))
        .stdout(std::process::Stdio::null())
        .stderr(log_file)
        .spawn()
        .map_err(|e| format!("{}: {e}", ffmpeg.display()))?;
    // how far it has got: the newest frame's time in what ffmpeg has written so far
    let mut reading = fs::File::open(&log_path).map_err(|e| e.to_string())?;
    let mut tail = String::new();
    let mut follow = || {
        use std::io::Read;
        let mut more = String::new();
        if reading.read_to_string(&mut more).is_ok() && !more.is_empty() {
            tail.push_str(&more);
            if let Some(t) = tail.rsplit("pts_time:").next().filter(|_| tail.contains("pts_time:")).and_then(|r| r.split_whitespace().next()?.parse::<f32>().ok()) {
                at(t);
            }
            let keep = tail.rfind('\n').map_or(0, |i| i + 1);
            tail.drain(..keep);
        }
    };
    loop {
        follow();
        match child.try_wait() {
            Ok(Some(_)) => {
                follow();
                break;
            }
            Ok(None) if stop() => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = fs::remove_dir_all(&part);
                return Ok(false);
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
            Err(e) => return Err(e.to_string()),
        }
    }
    // showinfo says each frame's time, in the order they are written: "... n:   3 pts: 90090 pts_time:3.003 ..."
    let log = fs::read_to_string(&log_path).unwrap_or_default();
    let _ = fs::remove_file(&log_path);
    let times: Vec<f32> = log.lines().filter(|l| l.contains("Parsed_showinfo")).filter_map(|l| l.split("pts_time:").nth(1)?.split_whitespace().next()?.parse().ok()).collect();
    let frames = frame_files(&part);
    if frames.is_empty() || times.len() != frames.len() {
        let _ = fs::remove_dir_all(&part);
        let errors: Vec<&str> = log.lines().filter(|l| !l.contains("Parsed_showinfo")).rev().take(3).collect();
        return Err(format!("{}: no frames ({})", video.display(), errors.join(" / ")));
    }
    let lines: String = times.iter().map(|t| format!("{t}\n")).collect();
    fs::write(part.join("times.txt"), lines).map_err(|e| e.to_string())?;
    let duration = log_duration(&log); // none known: the last frame's time
    // "    Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuvj420p(pc, ...), 3840x2160 [SAR 1:1 DAR 16:9], ..."
    let stored = log.lines().filter(|l| l.contains("Video:")).find_map(|l| {
        l.split(|c: char| c == ',' || c == ' ').find_map(|w| {
            let (a, b) = w.split_once('x')?;
            Some((a.parse::<u32>().ok().filter(|&v| v > 0)?, b.parse::<u32>().ok().filter(|&v| v > 0)?))
        })
    });
    // turned upright as its frames are (a phone's video is stored on its side, with a note to turn it)
    let (fw, fh) = shown_size(&frame_files(&part)[0]).unwrap_or((1, 1));
    let (width, height) = match stored {
        Some((w, h)) if (w > h) != (fw > fh) && w != h => (h, w),
        Some(size) => size,
        None => (fw, fh),
    };
    let video = crate::data::Video { duration: duration.unwrap_or(*times.last().expect("a frame")), width, height };
    fs::write(part.join("video.json"), serde_json::to_vec(&video).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    // what the file says about itself (its camera, date, place), as a photo's EXIF does
    if let Some(facts) = crate::files::video_facts(&log) {
        fs::write(part.join("facts.json"), serde_json::to_vec(&facts).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    }
    fs::rename(&part, dir).map_err(|e| e.to_string())?;
    Ok(true)
}

/// How long a video is, from what ffmpeg says reading it: "  Duration: 00:00:52.49, start: ...".
fn log_duration(log: &str) -> Option<f32> {
    log.lines().find_map(|l| {
        let hms = l.trim().strip_prefix("Duration:")?.split(',').next()?.trim();
        hms.split(':').map(|p| p.parse::<f32>().ok()).try_fold(0.0, |sum, p| Some(sum * 60.0 + p?))
    })
}

/// How long a video is (seconds), from its header (ffmpeg reads it and stops: a few ms).
fn video_length(ffmpeg: &Path, video: &Path) -> Option<f32> {
    let out = std::process::Command::new(ffmpeg).args(["-nostdin", "-hide_banner", "-i"]).arg(video).stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).output().ok()?;
    log_duration(&String::from_utf8_lossy(&out.stderr))
}

/// A video's frames, in order (0.jpg, 1.jpg...).
fn frame_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<(u32, PathBuf)> = fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter_map(|p| Some((p.file_stem()?.to_str()?.parse().ok()?, p))).collect();
    files.sort();
    files.into_iter().map(|(_, p)| p).collect()
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

/// Find the photos (once: a scan paused while reading what they say keeps its list, so each photo keeps its number), then write
/// items.json.
fn scan(ctx: &mut Ctx) -> Result<Flow, String> {
    if !ctx.dir.join("files.json").exists() {
        match find_photos(ctx)? {
            Flow::Done => {}
            paused => return Ok(paused),
        }
    }
    write_items(ctx)
}

/// Where the kth video's frames go while the map is made.
fn video_dir(k: usize) -> String {
    format!("videos/{k}")
}

/// The videos among the sources, in order: video k's frames go to video_dir(k).
fn videos(sources: &[PathBuf], every: usize) -> Vec<PathBuf> {
    collect_photos(sources, every).into_iter().filter(|p| is_video(p)).collect()
}

/// Takes the keyframes out of every video (video_frames), a few videos at a time. A video it can't read is left out of the map.
fn read_videos(ctx: &mut Ctx) -> Result<Flow, String> {
    ctx.progress(0, 0, "looking for videos");
    let videos = videos(&ctx.job.sources, ctx.job.every);
    if videos.is_empty() {
        return Ok(Flow::Done);
    }
    let Some(ffmpeg) = crate::runtime::ffmpeg(&ctx.root) else {
        return Err(format!("{} videos need ffmpeg to be read: download it in Runtime (on the Maps page), then carry on", videos.len()));
    };
    // counted in seconds of video (how much there is to read, more than how many videos): each one's length first, from its
    // header (one already read: its video.json)
    ctx.progress(0, 0, format!("{} videos, how long they are", videos.len()));
    let dirs: Vec<PathBuf> = (0..videos.len()).map(|k| ctx.dir.join(video_dir(k))).collect();
    let lengths: Vec<f32> = videos
        .par_iter()
        .zip(&dirs)
        .map(|(video, dir)| {
            let read: Option<crate::data::Video> = fs::read(dir.join("video.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
            read.map(|v| v.duration).or_else(|| video_length(&ffmpeg, video)).unwrap_or(0.0).max(0.0)
        })
        .collect();
    shared_state(&ctx.shared).unit = "seconds of video";
    ctx.progress(0, lengths.iter().sum::<f32>().round() as usize, "");
    let done = std::sync::atomic::AtomicU64::new(0); // milliseconds
    videos.par_iter().zip(&dirs).zip(&lengths).for_each(|((video, dir), &length)| {
        if ctx.paused() {
            return;
        }
        let counted = std::sync::atomic::AtomicU64::new(0);
        let count = |t: f32| {
            let t = (t.clamp(0.0, length) * 1000.0) as u64;
            let before = counted.fetch_max(t, Ordering::Relaxed);
            if t > before {
                ctx.set_done(((done.fetch_add(t - before, Ordering::Relaxed) + t - before) / 1000) as usize);
            }
        };
        match video_frames(&ffmpeg, video, dir, &|| ctx.paused(), &count) {
            Ok(true) => count(length),
            Ok(false) => {} // paused
            Err(e) => {
                eprintln!("{e}");
                count(length);
            }
        }
    });
    if ctx.paused() {
        return Ok(Flow::Paused);
    }
    // the ones with no frames: said on the build's card
    let unread: Vec<String> = videos.iter().zip(&dirs).filter(|(_, dir)| !dir.is_dir()).map(|(v, _)| v.file_name().map_or_else(|| v.display().to_string(), |n| n.to_string_lossy().into_owned())).collect();
    shared_state(&ctx.shared).unread_videos = unread.clone();
    ctx.job.unread_videos = unread;
    Ok(Flow::Done)
}

fn find_photos(ctx: &mut Ctx) -> Result<Flow, String> {
    ctx.progress(0, 0, "looking for photos");
    let files = collect_photos(&ctx.job.sources, ctx.job.every);
    // a video's frames, taken out by read_videos: its size is theirs (ffmpeg turns them upright)
    let videos: std::collections::HashMap<PathBuf, String> = videos(&ctx.job.sources, ctx.job.every).into_iter().enumerate().map(|(k, v)| (v, video_dir(k))).collect();
    let total = files.len();
    ctx.progress(0, total, format!("{total} files, checking them"));
    let done = AtomicUsize::new(0);
    let found: Vec<Option<Found>> = files
        .into_par_iter()
        .map(|src| {
            if ctx.paused() {
                return None;
            }
            let frames = videos.get(&src).cloned();
            let video = frames.as_ref().and_then(|rel| serde_json::from_slice(&fs::read(ctx.dir.join(rel).join("video.json")).ok()?).ok());
            let size = match &frames {
                Some(rel) => frame_files(&ctx.dir.join(rel)).first().and_then(|f| shown_size(f)), // none: the video couldn't be read
                None => shown_size(&src),
            };
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 64 == 0 {
                ctx.set_done(d);
            }
            size.map(|(width, height)| Found { src, width, height, frames, video })
        })
        .collect();
    if ctx.paused() {
        return Ok(Flow::Paused);
    }
    let found: Vec<Found> = found.into_iter().flatten().collect();
    if found.is_empty() {
        return Err(if !videos.is_empty() { "no photos in that folder, and no frames could be taken out of its videos (is ffmpeg installed?)" } else { "no photos in that folder" }.into());
    }
    write_atomic(&ctx.dir.join("files.json"), &serde_json::to_vec(&found).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    ctx.job.photos = found.len();
    Ok(Flow::Done)
}

fn read_found(dir: &Path) -> Result<Vec<Found>, String> {
    serde_json::from_slice(&fs::read(dir.join("files.json")).map_err(|e| format!("files.json: {e}"))?).map_err(|e| e.to_string())
}

/// Whether a hard link from one to the other can be made: the same file system (on btrfs, the same subvolume). Unknown: yes.
pub fn same_disk(a: &Path, b: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (fs::metadata(a), fs::metadata(b)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev(),
            _ => true,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (a, b);
        true
    }
}

fn write_items(ctx: &mut Ctx) -> Result<Flow, String> {
    let found = read_found(&ctx.dir)?;
    ctx.job.photos = found.len();
    let total = found.len();
    // what each file says about itself (camera, when, where...: a photo's EXIF, files::facts, the header only; a video's tags, read
    // with its frames)
    ctx.progress(0, total, "reading when and where the photos were taken");
    let done = AtomicUsize::new(0);
    let items: Option<Result<Vec<crate::data::Item>, String>> = found
        .par_iter()
        .map(|f| {
            if ctx.paused() {
                return None;
            }
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 256 == 0 {
                ctx.set_done(d);
            }
            let source = f.src.to_string_lossy().into_owned();
            Some(match (&f.frames, f.video) {
                (Some(dir), Some(video)) => {
                    let dir = ctx.dir.join(dir);
                    let times = match fs::read_to_string(dir.join("times.txt")) {
                        Ok(t) => t.lines().filter_map(|l| l.parse().ok()).collect(),
                        Err(e) => return Some(Err(format!("{}: {e}", dir.display()))),
                    };
                    let facts = fs::read(dir.join("facts.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
                    Ok(crate::data::Item { source, width: video.width, height: video.height, facts, video: Some(crate::data::Clip { duration: video.duration, times }) })
                }
                _ => Ok(crate::data::Item { source, width: f.width, height: f.height, facts: crate::files::facts(&f.src), video: None }),
            })
        })
        .collect();
    let Some(items) = items else { return Ok(Flow::Paused) };
    let items = items?;
    write_atomic(&ctx.dir.join("items.json"), &serde_json::to_vec(&items).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

fn shard_path(dir: &Path, k: usize) -> PathBuf {
    dir.join("pictures").join(format!("{k:05}.f16"))
}

/// The 32 px pictures of shard `k` (bake::TILE_BYTES each; all zeros for a picture not decoded: reused, or unreadable).
fn tiles_path(dir: &Path, k: usize) -> PathBuf {
    dir.join("pictures").join(format!("{k:05}.tiles"))
}

fn shard_len(pictures: usize, k: usize) -> usize {
    (pictures - k * SHARD).min(SHARD)
}

fn embed_photos(ctx: &mut Ctx) -> Result<Flow, String> {
    let n = ctx.job.photos;
    let dim = ctx.dim();
    // what is embedded: one list of pictures, each photo's file or a video's frames, photo after photo (photo i's are
    // pictures[start[i]..start[i + 1]]); a frame is a picture like any photo. Their vectors go to shards of SHARD pictures, and when
    // every picture has its vector the photos get theirs from them (put_in_place)
    let found = read_found(&ctx.dir)?;
    let frame_dirs: Vec<Option<PathBuf>> = found.iter().map(|f| f.frames.as_ref().map(|d| ctx.dir.join(d))).collect();
    let files: Vec<Vec<PathBuf>> = found.iter().zip(&frame_dirs).map(|(f, d)| match d {
        Some(d) => frame_files(d),
        None => vec![f.src.clone()],
    }).collect();
    let start: Vec<usize> = std::iter::once(0).chain(files.iter().scan(0, |s, f| {
        *s += f.len();
        Some(*s)
    })).collect();
    let owner: Vec<usize> = files.iter().enumerate().flat_map(|(i, f)| std::iter::repeat_n(i, f.len())).collect();
    let pictures: Arc<Vec<PathBuf>> = Arc::new(files.into_iter().flatten().collect());
    let total = pictures.len();
    let shards = total.div_ceil(SHARD);
    fs::create_dir_all(ctx.dir.join("pictures")).map_err(|e| e.to_string())?;
    let finished = |k: usize| fs::metadata(shard_path(&ctx.dir, k)).is_ok_and(|m| m.len() as usize == shard_len(total, k) * dim * 2);
    let todo: Vec<usize> = (0..shards).filter(|&k| !finished(k)).collect();
    let already = total - todo.iter().map(|&k| shard_len(total, k)).sum::<usize>();
    if frame_dirs.iter().any(Option::is_some) {
        shared_state(&ctx.shared).unit = "pictures";
    }
    ctx.progress(already, total, "loading the model (the first time it is prepared for this computer: up to a few minutes)");

    // the model is loaded by the search engine's thread; wait for it (not if every picture has its vector already)
    if !todo.is_empty() && !ctx.wait_for_model()? {
        return Ok(Flow::Paused);
    }
    let (model_id, spec) = (ctx.job.model.clone(), ctx.spec);
    if !todo.is_empty() {
        ctx.hub.with(&model_id, |m| m.prepare_build())??;
    }

    // photos an earlier map already has vectors for (the same file, unchanged): copied, not embedded (not videos: their frames'
    // vectors are needed too, and an earlier map keeps only the poster's)
    ctx.progress(already, total, "looking for photos embedded before");
    let mut in_todo = vec![false; shards];
    for &k in &todo {
        in_todo[k] = true;
    }
    let looked_for = |i: usize| found[i].frames.is_none() && start[i] < start[i + 1] && in_todo[start[i] / SHARD];
    let wanted: std::collections::HashSet<PathBuf> = (0..n).filter(|&i| looked_for(i)).map(|i| found[i].src.clone()).collect();
    let earlier = crate::reuse::Found::open(ctx.dir.parent().ok_or("no maps folder")?, &wanted, &model_id, dim);
    drop(wanted);
    let hits: Vec<Option<crate::reuse::Hit>> = (0..n).into_par_iter().map(|i| if looked_for(i) { earlier.find(&found[i].src) } else { None }).collect();
    drop(found);
    let reused: Arc<Vec<bool>> = Arc::new(owner.iter().map(|&i| hits[i].is_some()).collect());
    let reused_count = reused.iter().filter(|&&r| r).count();
    // the reused photos cost nothing: they count as done from the start, so the speed and the time left are about the pictures the model does
    ctx.progress(already + reused_count, total, if reused_count > 0 { format!("{reused_count} photos were embedded in earlier maps; embedding the rest") } else { "reading the photos and embedding them".to_string() });

    // CPU side ahead of the GPU: the next shard's pictures are decoded and cut into patches while these are embedded. Decoded
    // anyway, each is made 32 px for the bake too (the shard's tiles), and a video's frame FRAME_SIDE (videos/<k>/small/), so that
    // the bake reads none of them again
    let is_frame: Arc<Vec<bool>> = Arc::new(owner.iter().map(|&i| frame_dirs[i].is_some()).collect());
    let (tx, rx) = std::sync::mpsc::sync_channel::<(usize, Vec<Option<(Prepared, image::RgbaImage)>>, Vec<u8>)>(1);
    let stop = Arc::new(AtomicBool::new(false));
    let producer = {
        let (todo, pictures, stop, reused, is_frame) = (todo.clone(), pictures.clone(), stop.clone(), reused.clone(), is_frame.clone());
        std::thread::spawn(move || {
            for k in todo {
                let read: Vec<Option<(Prepared, image::RgbaImage, Vec<u8>)>> = (k * SHARD..k * SHARD + shard_len(total, k))
                    .into_par_iter()
                    .map(|p| {
                        if reused[p] || stop.load(Ordering::Relaxed) {
                            return None;
                        }
                        let img = crate::photo::load_for(&pictures[p], |w, h| spec.longest_side_needed(w, h))?;
                        if is_frame[p] {
                            small_frame_beside(&pictures[p], &img); // if it fails, the bake makes it small instead
                        }
                        Some((spec.prepare(&img).ok()?, thumbnail(&img), crate::bake::tile_of(&img)))
                    })
                    .collect();
                let tiles: Vec<u8> = read.iter().flat_map(|r| r.as_ref().map_or_else(|| vec![0; crate::bake::TILE_BYTES], |r| r.2.clone())).collect();
                let read = read.into_iter().map(|r| r.map(|(patches, thumb, _)| (patches, thumb))).collect();
                if stop.load(Ordering::Relaxed) || tx.send((k, read, tiles)).is_err() {
                    return;
                }
            }
        })
    };

    let mut processed = 0usize; // pictures that were not reused, seen to so far
    let mut embedded = 0usize; // ... and of those, the ones that got a vector
    let mut outcome = Ok(Flow::Done);
    'shards: for (k, mut read, tiles) in rx {
        let first = k * SHARD;
        let mut rows = vec![0u8; read.len() * dim * 2]; // the shard's, f16; a picture that couldn't be read keeps a zero vector
        for (j, row) in rows.chunks_exact_mut(dim * 2).enumerate() {
            if let Some(hit) = hits[owner[first + j]] {
                row.copy_from_slice(earlier.row(hit));
            }
        }
        let ready: Vec<usize> = (0..read.len()).filter(|&j| read[j].is_some()).collect();
        let seen_before = processed;
        // BATCH pictures at a time, so that a pause is never far off
        for batch in ready.chunks(BATCH) {
            if ctx.paused() {
                outcome = Ok(Flow::Paused);
                break 'shards;
            }
            let (patches, thumbs): (Vec<Prepared>, Vec<image::RgbaImage>) = batch.iter().map(|&j| read[j].take().expect("ready")).unzip();
            let results = match ctx.hub.with(&model_id, |m| m.embed_prepared(patches)) {
                Ok(results) => results,
                Err(e) => vec![Err(e); batch.len()],
            };
            let mut head = None;
            for (&j, r) in batch.iter().zip(results) {
                match r {
                    Ok(v) => {
                        for (b, x) in rows[j * dim * 2..(j + 1) * dim * 2].chunks_exact_mut(2).zip(&v) {
                            b.copy_from_slice(&half::f16::from_f32(*x).to_le_bytes());
                        }
                        embedded += 1;
                        head = Some(v[..HEAD].to_vec());
                    }
                    Err(e) if e.contains("running") || e.contains("loading") => {
                        outcome = Err(e);
                        break 'shards;
                    }
                    Err(_) => {} // a picture it couldn't embed keeps a zero vector
                }
            }
            ctx.show_recent(thumbs, head);
            processed += batch.len();
            ctx.set_done(already + reused_count + processed);
        }
        processed = seen_before + (first..first + read.len()).filter(|&p| !reused[p]).count();
        ctx.set_done(already + reused_count + processed);
        // the tiles first: the vectors' shard is what says the shard is done
        write_atomic(&tiles_path(&ctx.dir, k), &tiles).map_err(|e| e.to_string())?;
        write_atomic(&shard_path(&ctx.dir, k), &rows).map_err(|e| e.to_string())?;
    }
    stop.store(true, Ordering::Relaxed);
    let _ = producer.join();
    let flow = outcome?;
    if matches!(flow, Flow::Paused) {
        return Ok(flow);
    }
    // not one picture could be read (say, the disk they are on went away): a map of them would be all in one spot
    if processed > 0 && embedded == 0 {
        for k in todo {
            let _ = fs::remove_file(shard_path(&ctx.dir, k));
        }
        return Err(format!("none of the {processed} photos could be read (the first: {})", pictures.iter().zip(reused.iter()).find(|(_, r)| !**r).map_or(String::new(), |(f, _)| f.display().to_string())));
    }
    ctx.progress(total, total, "putting the vectors in place");
    put_in_place(&ctx.dir, &start, &frame_dirs, &pictures, dim).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

/// look.bin from the pictures' vectors (the shards, read once in order): every item's, then every video's frames', video after video.
/// A photo's is its picture's; a video's is its poster's, the frame nearest the average of its frames (copied to poster.jpg). A real
/// picture's vector, like every photo's: an average of many frames is like no picture, and drifts toward the other videos'. A frame
/// that couldn't be read keeps a zero vector.
fn put_in_place(dir: &Path, start: &[usize], frame_dirs: &[Option<PathBuf>], pictures: &[PathBuf], dim: usize) -> io::Result<()> {
    use std::io::Write;
    let row = dim * 2;
    let mut look = io::BufWriter::new(fs::File::create(dir.join("look.tmp"))?);
    // the frames' rows go after all the items': kept aside until then
    let mut frames = io::BufWriter::new(fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(dir.join("frames.tmp"))?);
    let (mut shard, mut at, mut k) = (Vec::new(), 0, 0);
    let mut rows = Vec::new();
    for (i, frame_dir) in frame_dirs.iter().enumerate() {
        let count = start[i + 1] - start[i];
        rows.clear();
        while rows.len() < count * row {
            if at == shard.len() {
                (shard, at, k) = (fs::read(shard_path(dir, k))?, 0, k + 1);
            }
            let take = (count * row - rows.len()).min(shard.len() - at);
            rows.extend_from_slice(&shard[at..at + take]);
            at += take;
        }
        let Some(frame_dir) = frame_dir else {
            look.write_all(&rows)?;
            continue;
        };
        let vectors: Vec<Vec<f32>> = rows.chunks_exact(row).map(|r| r.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()).collect();
        let mut mean = vec![0f32; dim];
        for v in &vectors {
            for (m, x) in mean.iter_mut().zip(v) {
                *m += x;
            }
        }
        let dot = |v: &Vec<f32>| v.iter().zip(&mean).map(|(a, b)| a * b).sum::<f32>();
        let best = (0..count).max_by(|&a, &b| dot(&vectors[a]).total_cmp(&dot(&vectors[b]))).expect("a video has frames");
        fs::copy(&pictures[start[i] + best], frame_dir.join("poster.jpg"))?;
        look.write_all(&rows[best * row..(best + 1) * row])?;
        frames.write_all(&rows)?;
    }
    let mut frames = frames.into_inner().map_err(|e| e.into_error())?;
    std::io::Seek::rewind(&mut frames)?;
    io::copy(&mut frames, &mut look)?;
    drop(frames);
    fs::remove_file(dir.join("frames.tmp"))?;
    look.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    fs::rename(dir.join("look.tmp"), dir.join("look.bin"))
}

fn read_layout(dir: &Path, n: usize) -> Result<Vec<f32>, String> {
    let raw = fs::read(dir.join("layout.bin")).map_err(|e| format!("layout.bin: {e}"))?;
    if raw.len() != n * 8 {
        return Err("layout.bin is not the size of the photos".into());
    }
    Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

fn lay_out(ctx: &mut Ctx) -> Result<Flow, String> {
    ctx.progress(0, 0, "preparing the vectors");
    let vectors = crate::cells::HalfRows::open(&ctx.dir.join("look.bin"), ctx.job.photos, ctx.dim())?;
    let xy = crate::layout::layout(vectors, &|what, done, total| ctx.progress(done, total, what), &|| ctx.paused())?;
    let Some(xy) = xy else { return Ok(Flow::Paused) };
    // moved apart so that no photo covers another, at the size the viewer will draw them
    let points: &[crate::data::Point] = bytemuck::cast_slice(&xy);
    let side = crate::median_neighbour_distance(points);
    let points = crate::data::spread(points, side);
    let bytes: Vec<u8> = points.iter().flat_map(|p| [p.x, p.y]).flat_map(|v| v.to_le_bytes()).collect();
    write_atomic(&ctx.dir.join("layout.bin"), &bytes).map_err(|e| e.to_string())?;
    ctx.job.photo_side = Some(side);
    Ok(Flow::Done)
}

/// The tree that paths between photos follow (tree.rs).
fn make_tree(ctx: &mut Ctx) -> Result<Flow, String> {
    ctx.progress(0, 0, "preparing the vectors");
    // f16, from the file where it is: an f32 copy would be twice the memory, for the same numbers
    let vectors = crate::cells::HalfRows::open(&ctx.dir.join("look.bin"), ctx.job.photos, ctx.dim())?;
    let Some(parent) = crate::tree::build(&vectors, &|what, done, total| ctx.progress(done, total, what), &|| ctx.paused()) else { return Ok(Flow::Paused) };
    crate::tree::write(&ctx.dir, &parent).map_err(|e| e.to_string())?;
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
    let phrases = crate::naming::phrases();

    // the vocabulary's vectors: cached after the first map (no model needed then); made with the shared model otherwise
    ctx.progress(0, phrases.len(), "reading the vocabulary");
    let vocabulary = match crate::naming::cached(&ctx.root, &ctx.job.model, ctx.dim(), &phrases) {
        Some(cached) => cached,
        None => {
            if !ctx.wait_for_model()? {
                return Ok(Flow::Paused);
            }
            let (hub, model_id) = (ctx.hub.clone(), ctx.job.model.clone());
            let mut embed = |text: &str| hub.with(&model_id, |m| m.embed_phrase(text))?;
            let vocabulary = crate::naming::vectors(&ctx.root, &ctx.job.model, ctx.dim(), &phrases, &mut embed, &|done, total| ctx.progress(done, total, "learning the vocabulary (first map only)"), &|| ctx.paused())?;
            let Some(vocabulary) = vocabulary else { return Ok(Flow::Paused) };
            vocabulary
        }
    };

    ctx.progress(0, 0, "naming the regions");
    let vectors = crate::cells::HalfRows::open(&ctx.dir.join("look.bin"), n, ctx.dim())?;
    let xy = read_layout(&ctx.dir, n)?;
    let named = crate::naming::name_levels(ctx.dim(), &vectors, &xy, &levels, &phrases, &vocabulary);
    let json = serde_json::to_vec(&serde_json::json!({ "levels": named })).map_err(|e| e.to_string())?;
    write_atomic(&ctx.dir.join("regions.json"), &json).map_err(|e| e.to_string())?;
    Ok(Flow::Done)
}

/// The small pictures the viewer draws from (bake.rs), made now so that the finished map opens at once: the items' (a video's
/// poster), then the videos' frames.
fn bake_pages(ctx: &mut Ctx) -> Result<Flow, String> {
    let found = read_found(&ctx.dir)?;
    let frames = keep_frames(ctx, &found)?;
    let pictures: Vec<PathBuf> = found.iter().map(|f| f.frames.as_ref().map_or_else(|| f.src.clone(), |d| ctx.dir.join(d).join("poster.jpg"))).chain(frames).collect();
    // the 32 px pictures embed made: picture b here is picture `embedded[b]` there (a photo; a frame, there among its video's
    // photo's place); a poster has none (it was picked after), nor has a photo reused from an earlier map: those are read
    let mut embedded = vec![usize::MAX; pictures.len()];
    let (mut at, mut frame) = (0, found.len());
    for (i, f) in found.iter().enumerate() {
        let count = f.frames.as_ref().map_or(1, |d| frame_files(&ctx.dir.join(d)).len());
        match f.frames {
            None => embedded[i] = at,
            Some(_) => {
                for j in 0..count {
                    embedded[frame + j] = at + j;
                }
                frame += count;
            }
        }
        at += count;
    }
    let shards = at.div_ceil(SHARD);
    let tiles: Vec<Option<memmap2::Mmap>> = (0..shards)
        .map(|k| {
            let map = unsafe { memmap2::Mmap::map(&fs::File::open(tiles_path(&ctx.dir, k)).ok()?) }.ok()?;
            (map.len() == shard_len(at, k) * crate::bake::TILE_BYTES).then_some(map)
        })
        .collect();
    let ready = |b: usize| {
        let p = *embedded.get(b).filter(|&&p| p != usize::MAX)?;
        let at = p % SHARD * crate::bake::TILE_BYTES;
        Some(tiles[p / SHARD].as_ref()?[at..at + crate::bake::TILE_BYTES].to_vec())
    };
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
        let result = crate::bake::bake(&ctx.dir, &pictures, &ready, &progress);
        finished.store(true, Ordering::Relaxed);
        result
    });
    result.map_err(|e| e.to_string())?;
    drop(tiles);
    for k in 0..shards {
        let _ = fs::remove_file(tiles_path(&ctx.dir, k));
    }
    Ok(Flow::Done)
}

/// The videos' frames as the map keeps them, made small (FRAME_SIDE): their files, video after video.
fn keep_frames(ctx: &mut Ctx, found: &[Found]) -> Result<Vec<PathBuf>, String> {
    let videos: Vec<PathBuf> = found.iter().filter_map(|f| Some(ctx.dir.join(f.frames.as_ref()?))).collect();
    if videos.is_empty() {
        return Ok(Vec::new());
    }
    ctx.progress(0, videos.len(), "putting the videos' small frames in place");
    let done = AtomicUsize::new(0);
    let frames: Vec<Vec<PathBuf>> = videos
        .par_iter()
        .map(|dir| {
            let files = frame_files(dir);
            // embed made them small beside them (videos/<k>/small/); one it didn't make is made small here
            let small_dir = dir.join("small");
            for file in &files {
                let small = small_dir.join(file.file_name().expect("a frame's file"));
                if small.is_file() {
                    fs::rename(&small, file).map_err(|e| e.to_string())?;
                } else if let Some(bytes) = small_frame(file)? {
                    write_atomic(file, &bytes).map_err(|e| e.to_string())?;
                }
            }
            let _ = fs::remove_dir_all(&small_dir);
            ctx.set_done(done.fetch_add(1, Ordering::Relaxed) + 1);
            Ok(files)
        })
        .collect::<Result<_, String>>()?;
    Ok(frames.into_iter().flatten().collect())
}

/// A frame made small (FRAME_SIDE) as a JPEG, or None if it is that small already. A big JPEG is decoded at a fraction of its
/// size (photo::load_for: twice FRAME_SIDE, then shrunk), not whole.
pub(crate) fn small_frame(file: &Path) -> Result<Option<Vec<u8>>, String> {
    let img = crate::photo::load_for(file, |_, _| crate::data::FRAME_SIDE).ok_or_else(|| format!("{}: can't read", file.display()))?;
    frame_jpeg(&img)
}

/// A frame's small copy (made from the frame embed decodes anyway) to videos/<k>/small/, beside it. Not made if it can't be written.
fn small_frame_beside(file: &Path, img: &image::RgbaImage) {
    let (Some(dir), Some(name)) = (file.parent(), file.file_name()) else { return };
    if let Ok(Some(bytes)) = frame_jpeg(img) {
        let small = dir.join("small");
        let _ = fs::create_dir_all(&small).and_then(|_| write_atomic(&small.join(name), &bytes));
    }
}

/// A decoded frame as the map keeps it: FRAME_SIDE, a JPEG; None if it is that small already.
fn frame_jpeg(img: &image::RgbaImage) -> Result<Option<Vec<u8>>, String> {
    let side = crate::data::FRAME_SIDE;
    if img.width().max(img.height()) <= side {
        return Ok(None);
    }
    // written by libjpeg-turbo (several times quicker than image's encoder), 4:2:0 as cameras and ffmpeg write them
    let small = crate::photo::fit(&img, side, false);
    use libjpeg_turbo_rs::{PixelFormat, Subsampling};
    let bytes = libjpeg_turbo_rs::compress(small.as_raw(), small.width() as usize, small.height() as usize, PixelFormat::Rgba, 85, Subsampling::S420).map_err(|e| e.to_string())?;
    Ok(Some(bytes))
}

fn finish(ctx: &mut Ctx) -> Result<Flow, String> {
    let found = read_found(&ctx.dir)?;
    let manifest = crate::data::Manifest {
        name: ctx.job.name.clone(),
        created: ctx.job.created.clone(),
        photos: ctx.job.photos,
        videos: found.iter().filter(|f| f.frames.is_some()).count(),
        photo_side: ctx.job.photo_side.ok_or("the layout's photo size is missing: lay out the map again")?,
        look: crate::data::Vectors { model: ctx.job.model.clone(), dim: ctx.dim() },
    };
    write_atomic(&ctx.dir.join("manifest.json"), &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    // the videos' folders become item i's frames/<i>/, without what is in items.json and look.bin now (a folder already
    // moved is not there any more: a finish stopped half way carries on)
    for (i, f) in found.iter().enumerate() {
        let Some(rel) = &f.frames else { continue };
        let from = ctx.dir.join(rel);
        if !from.is_dir() {
            continue;
        }
        for name in ["times.txt", "video.json", "facts.json"] {
            let _ = fs::remove_file(from.join(name));
        }
        fs::create_dir_all(ctx.dir.join("frames")).map_err(|e| e.to_string())?;
        fs::rename(&from, ctx.dir.join(format!("frames/{i}"))).map_err(|e| e.to_string())?;
    }
    let _ = fs::remove_dir_all(ctx.dir.join("videos"));
    let _ = fs::remove_dir_all(ctx.dir.join("pictures"));
    for name in ["files.json", "levels.bin", "job.json", ".lock"] {
        let _ = fs::remove_file(ctx.dir.join(name));
    }
    let maps = ctx.dir.parent().ok_or("no maps folder")?;
    fs::rename(&ctx.dir, finished_dir(maps, &ctx.job.name)).map_err(|e| e.to_string())?;
    ctx.shared.state.lock().unwrap().finished = true;
    Ok(Flow::Done)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Turns maps of the format before items.json into today's (data.rs), in place: photos.json and frames.json become items.json;
    /// embeddings.bin and frames.bin look.bin; region_names.json regions.json; pages_bc7.bin
    /// pages.bin; frames/<k, 6 digits>/<n, 6 digits>.jpg frames/<i>/<n - 1>.jpg. The old json files are kept in `backup`/<map>/
    /// first. A map stopped half way is carried on. Maps with photos copied into them are left alone.
    /// cargo test --release convert_maps -- --ignored --nocapture (PHOTO_MAP_MAPS: the maps folder, else the app's)
    #[test]
    #[ignore]
    fn convert_maps() {
        let maps = std::env::var_os("PHOTO_MAP_MAPS").map_or_else(|| crate::data_root().join("maps"), PathBuf::from);
        let backup = maps.parent().unwrap().join("format-backup");
        let mut dirs: Vec<PathBuf> = fs::read_dir(&maps).unwrap().flatten().map(|e| e.path()).filter(|p| p.join("photos.json").exists()).collect();
        dirs.sort();
        for dir in dirs {
            let started = Instant::now();
            match convert(&dir, &backup.join(dir.file_name().unwrap())) {
                Ok(()) => {
                    let data = crate::data::MapData::load(&dir, &crate::data::Progress::default()).expect("the map opens");
                    let look = fs::metadata(dir.join("look.bin")).unwrap().len() as usize;
                    assert_eq!(look, data.pictures() * data.manifest.dim() * 2, "look.bin");
                    assert!((0..data.pictures()).all(|i| i < data.n() || data.picture_path(i).exists()), "every frame's file");
                    assert!((0..data.n()).all(|i| data.photos[i].video.is_none() || data.path(i).exists()), "every poster");
                    eprintln!("{}: {} items, {} frames ({:.1} s)", dir.display(), data.n(), data.frames.len(), started.elapsed().as_secs_f32());
                }
                Err(e) => eprintln!("{}: left as it was: {e}", dir.display()),
            }
        }
    }

    fn convert(dir: &Path, backup: &Path) -> Result<(), String> {
        use serde_json::Value;
        let read = |name: &str| -> Result<Value, String> { serde_json::from_slice(&fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?).map_err(|e| format!("{name}: {e}")) };
        let old = read("manifest.json")?;
        let photos = read("photos.json")?;
        let photos = photos.as_array().ok_or("photos.json is not a list")?;
        let frames = if dir.join("frames.json").exists() { read("frames.json")? } else { Value::Array(Vec::new()) };
        let frames = frames.as_array().ok_or("frames.json is not a list")?;
        let n = photos.len();
        if old["photos"].as_u64() != Some(n as u64) {
            return Err("manifest.json and photos.json don't agree".into());
        }
        if photos.iter().any(|p| p.get("video").is_none() && p["file"].as_str().is_some_and(|f| !f.is_empty())) {
            return Err("its photos were copied into it".into());
        }
        let look = crate::data::Vectors {
            model: old["embedding"]["model"].as_str().unwrap_or(models::DEFAULT).to_string(),
            dim: old["embedding"]["dim"].as_u64().map_or(DIM, |d| d as usize),
        };
        let photo_side = old["photo_side"].as_f64().ok_or("no photo_side")? as f32;
        // the frames come video after video
        let mut times: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
        let mut last = 0;
        for f in frames {
            let (video, time) = (f["video"].as_u64().unwrap() as usize, f["time"].as_f64().unwrap() as f32);
            if video < last || video >= n || photos[video].get("video").is_none() {
                return Err(format!("frames out of order at video {video}"));
            }
            last = video;
            times.entry(video).or_default().push(time);
        }
        let mut items = Vec::with_capacity(n);
        for (i, p) in photos.iter().enumerate() {
            let source = p["source"].as_str().ok_or(format!("photo {i} has no source"))?.to_string();
            let facts = p.get("facts").map(|f| serde_json::from_value(f.clone())).transpose().map_err(|e| e.to_string())?;
            let size = |v: &Value| (v["width"].as_u64().unwrap() as u32, v["height"].as_u64().unwrap() as u32);
            items.push(match p.get("video") {
                Some(v) => {
                    let (width, height) = size(v);
                    let times = times.get(&i).cloned().ok_or(format!("video {i} has no frames"))?;
                    let clip = crate::data::Clip { duration: v["duration"].as_f64().unwrap() as f32, times };
                    crate::data::Item { source, width, height, facts, video: Some(clip) }
                }
                None => {
                    let (width, height) = size(p);
                    crate::data::Item { source, width, height, facts, video: None }
                }
            });
        }
        let row = look.dim * 2;
        let len = |name: &str| fs::metadata(dir.join(name)).map(|m| m.len() as usize).ok();
        if len("embeddings.bin").is_some_and(|l| l != n * row && l != (n + frames.len()) * row) || len("frames.bin").is_some_and(|l| l != frames.len() * row) {
            return Err("embeddings.bin or frames.bin is not the size of the pictures".into());
        }
        // the old json files first, as they were
        fs::create_dir_all(backup).map_err(|e| e.to_string())?;
        for name in ["manifest.json", "photos.json", "frames.json", "region_names.json", "regions.json"] {
            if dir.join(name).exists() && !backup.join(name).exists() {
                fs::copy(dir.join(name), backup.join(name)).map_err(|e| e.to_string())?;
            }
        }
        // the frames: frames/<k>/<n>.jpg (n from 1, as ffmpeg numbered them) to frames/<i>/<n - 1>.jpg, the poster with them
        if !frames.is_empty() {
            let (old_frames, new_frames) = (dir.join("frames.old"), dir.join("frames"));
            if !old_frames.exists() {
                fs::rename(&new_frames, &old_frames).map_err(|e| e.to_string())?;
            }
            let mut index = 0;
            for (i, p) in photos.iter().enumerate() {
                if p.get("video").is_none() {
                    continue;
                }
                let to = new_frames.join(i.to_string());
                fs::create_dir_all(&to).map_err(|e| e.to_string())?;
                let poster = p["file"].as_str().unwrap().strip_prefix("frames/").ok_or("a poster outside frames/")?;
                let from = old_frames.join(poster);
                if from.exists() {
                    fs::rename(&from, to.join("poster.jpg")).map_err(|e| e.to_string())?;
                }
                for k in 0..times[&i].len() {
                    let from = old_frames.join(frames[index]["file"].as_str().unwrap().strip_prefix("frames/").ok_or("a frame outside frames/")?);
                    if from.exists() {
                        fs::rename(&from, to.join(format!("{k}.jpg"))).map_err(|e| e.to_string())?;
                    }
                    index += 1;
                }
            }
            fs::remove_dir_all(&old_frames).map_err(|e| e.to_string())?;
        }
        // the vectors: the frames' rows after the items' (appended where they are: no copy of the items')
        if len("embeddings.bin") == Some(n * row) && !frames.is_empty() {
            let mut out = fs::OpenOptions::new().append(true).open(dir.join("embeddings.bin")).map_err(|e| e.to_string())?;
            io::copy(&mut fs::File::open(dir.join("frames.bin")).map_err(|e| e.to_string())?, &mut out).map_err(|e| e.to_string())?;
            out.sync_all().map_err(|e| e.to_string())?;
        }
        let rename = |from: &str, to: &str| -> Result<(), String> { if dir.join(from).exists() { fs::rename(dir.join(from), dir.join(to)).map_err(|e| e.to_string()) } else { Ok(()) } };
        rename("embeddings.bin", "look.bin")?;
        rename("pages_bc7.bin", "pages.bin")?;
        if dir.join("region_names.json").exists() {
            rename("region_names.json", "regions.json")?;
        }
        let videos = items.iter().filter(|i| i.video.is_some()).count();
        let manifest = crate::data::Manifest { name: old["name"].as_str().unwrap_or_default().to_string(), created: old["created"].as_str().unwrap_or_default().to_string(), photos: n, videos, photo_side, look };
        write_atomic(&dir.join("items.json"), &serde_json::to_vec(&items).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        write_atomic(&dir.join("manifest.json"), &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        for name in ["photos.json", "frames.json", "frames.bin"] {
            let _ = fs::remove_file(dir.join(name));
        }
        Ok(())
    }

    const DIM: usize = 768;

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

    /// Photos come from folders and single files, each once.
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
        fs::remove_dir_all(&temp).unwrap();
    }

    /// A photo an earlier map already holds a vector for (same file, not changed since) is not embedded again: its row is copied.
    /// Proved by changing a row in the first map to a vector the model would never make, and finding it in the second.
    #[cfg(unix)]
    /// A video's keyframes come out with their times (portrait: the longer side 1024 px), once; a stop kills ffmpeg and leaves
    /// nothing behind. Needs ffmpeg; no model.
    #[test]
    fn video_frames_and_their_times() {
        let temp = std::env::temp_dir().join(format!("photo-map-frames-test-{}", std::process::id()));
        fs::create_dir_all(&temp).unwrap();
        let video = temp.join("tall.mp4");
        let made = std::process::Command::new(crate::runtime::ffmpeg(&crate::data_root()).expect("ffmpeg"))
            .args(["-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=4:size=1080x1920:rate=25", "-g", "10", "-sc_threshold", "0"])
            .arg(&video)
            .status()
            .unwrap();
        assert!(made.success());
        let dir = temp.join(video_dir(0));
        let ffmpeg = crate::runtime::ffmpeg(&crate::data_root()).expect("ffmpeg");
        assert_eq!(video_frames(&ffmpeg, &video, &dir, &|| true, &|_| {}), Ok(false), "stopped");
        assert!(!dir.exists() && !dir.with_extension("part").exists(), "nothing left of a stopped one");
        let got_to = std::cell::Cell::new(0.0f32);
        assert_eq!(video_frames(&ffmpeg, &video, &dir, &|| false, &|t| got_to.set(got_to.get().max(t))), Ok(true));
        assert!((got_to.get() - 2.0).abs() < 0.2, "it says how far it has got: {}", got_to.get());
        let frames = frame_files(&dir);
        assert_eq!(frames.len(), 2, "of a keyframe every 0.4 s, one every 2 s");
        assert_eq!(shown_size(&frames[0]), Some((576, 1024)));
        let times: Vec<f32> = fs::read_to_string(dir.join("times.txt")).unwrap().lines().map(|l| l.parse().unwrap()).collect();
        assert!(times.len() == 2 && times.iter().enumerate().all(|(k, t)| (t - 2.0 * k as f32).abs() < 0.2), "2 s apart: {times:?}");
        assert!(!dir.join("ffmpeg.log").exists());
        let info: crate::data::Video = serde_json::from_slice(&fs::read(dir.join("video.json")).unwrap()).unwrap();
        assert!((info.duration - 4.0).abs() < 0.1, "{info:?}");
        assert_eq!((info.width, info.height), (1080, 1920), "the video's own size, upright");
        assert_eq!(video_frames(&ffmpeg, &video, &dir, &|| true, &|_| {}), Ok(true), "done once: kept");
        fs::remove_dir_all(&temp).unwrap();
    }

    /// A video is one photo on the map: its picture and vector the frame nearest its frames' average, its
    /// frames' vectors kept for search. Needs ffmpeg (it makes the videos, too).
    #[test]
    fn a_video_is_one_photo() {
        let root = crate::data_root();
        let temp = std::env::temp_dir().join(format!("photo-map-video-test-{}", std::process::id()));
        let (source, maps) = (temp.join("media"), temp.join("maps"));
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&maps).unwrap();
        for i in 0..6 {
            let img = image::RgbImage::from_fn(96, 80, |x, y| image::Rgb([(i * 40) as u8, (x * 2) as u8, (y * 3) as u8]));
            img.save(source.join(format!("p{i}.png"))).unwrap();
        }
        // 12 s of test pattern then 6 s of colour bars, a keyframe every second: 18 frames
        let made = std::process::Command::new(crate::runtime::ffmpeg(&crate::data_root()).expect("ffmpeg"))
            .args(["-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=12:size=320x180:rate=25", "-f", "lavfi", "-i", "smptebars=duration=6:size=320x180:rate=25", "-filter_complex", "[0][1]concat=n=2", "-g", "25", "-sc_threshold", "0"])
            .arg(source.join("clip.mp4"))
            .status()
            .unwrap();
        assert!(made.success());
        let model = models::test_hub();
        let dir = create(&maps, "videos", &[source.clone()], 1, models::DEFAULT).unwrap();
        let run = Run::start(dir, root, model, || {}).unwrap();
        while run.running() {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(run.state().error.is_none(), "{:?}", run.state().error);
        let done = finished_dir(&maps, "videos");
        let items: Vec<crate::data::Item> = serde_json::from_slice(&fs::read(done.join("items.json")).unwrap()).unwrap();
        assert_eq!(items.len(), 7, "six photos and one video");
        let video = items.iter().position(|p| p.source.ends_with("clip.mp4")).unwrap();
        assert_eq!((items[video].width, items[video].height), (320, 180), "the video's own size");
        let clip = items[video].video.as_ref().unwrap();
        assert!((clip.duration - 18.0).abs() < 0.1, "a video says how long it is");
        assert!(items.iter().filter(|p| p.video.is_some()).count() == 1, "a photo doesn't");
        // of its 18 keyframes, one every 2 s, made small; their vectors after the items' in look.bin, in the same order
        assert_eq!(clip.times.len(), 9);
        assert!(clip.times.iter().enumerate().all(|(k, t)| (t - 2.0 * k as f32).abs() < 0.2), "{:?}", clip.times);
        let folder = done.join(format!("frames/{video}"));
        assert!(folder.join("poster.jpg").exists());
        assert_eq!(fs::read_dir(&folder).unwrap().count(), 10, "the poster and the frames, nothing else");
        assert!(!done.join("videos").exists() && !done.join("files.json").exists(), "no working files left");
        let data = crate::data::MapData::load(&done, &crate::data::Progress::default()).unwrap();
        assert_eq!(data.pictures(), 7 + 9);
        for (k, f) in data.frames.iter().enumerate() {
            assert_eq!(f.video, video);
            assert_eq!(crate::photo::load(&done.join(&f.file)).map(|i| (i.width(), i.height())), Some((f.width, f.height)), "frame {k} is {:?}", (f.width, f.height));
        }
        assert_eq!((data.photos[video].width, data.photos[video].height), (320, 180), "a poster no bigger than its video");
        assert_eq!(fs::metadata(done.join("look.bin")).unwrap().len() as usize, (7 + 9) * DIM * 2);
        // the map opens with them, and a search finds the video by its best frame and shows that frame
        let vectors = crate::search::Vectors::open(&done, 7, 9, DIM).unwrap();
        let frame: Vec<f32> = fs::read(done.join("look.bin")).unwrap()[(7 + 3) * DIM * 2..(7 + 4) * DIM * 2].chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        let mut scores = vectors.cosines(&frame);
        let shown = crate::search::best_frames(&mut scores, &vectors.frame_cosines(&frame), &data.frames);
        assert!(scores[video] > 0.99, "frame 3 (6 s) against itself: {}", scores[video]);
        assert!(shown[video] >= 7 && data.frames[shown[video] - 7].time < 12.0, "a frame of the test pattern, not of the bars: {}", shown[video]);
        fs::remove_dir_all(&temp).unwrap();
    }

    #[test]
    fn reuses_vectors_of_earlier_maps() {
        let root = crate::data_root();
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
        let model = models::test_hub();
        let build = |name: &str| {
            let dir = create(&maps, name, &[source.clone()], 1, models::DEFAULT).unwrap();
            let run = Run::start(dir, root.clone(), model.clone(), || {}).unwrap();
            while run.running() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            assert!(run.state().error.is_none(), "{:?}", run.state().error);
            finished_dir(&maps, name)
        };
        let first = build("first");
        let photos: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(first.join("items.json")).unwrap()).unwrap();
        assert!(Path::new(photos[0]["source"].as_str().unwrap()).is_absolute() && photos[0]["source"].as_str().unwrap().ends_with(".png"), "the full path of the file");

        // the first map is the only one the second can reuse from: its row 0 becomes a vector the model would not give
        let mut raw = fs::read(first.join("look.bin")).unwrap();
        let marker: Vec<u8> = (0..DIM).flat_map(|d| half::f16::from_f32(if d == 0 { 1.0 } else { 0.0 }).to_le_bytes()).collect();
        raw[..DIM * 2].copy_from_slice(&marker);
        fs::write(first.join("look.bin"), &raw).unwrap();

        let second = build("second");
        let again = fs::read(second.join("look.bin")).unwrap();
        assert_eq!(&again[..DIM * 2], &marker[..], "row 0 was copied from the first map");
        assert_eq!(&again[DIM * 2..], &raw[DIM * 2..], "and the others");

        // a file changed since the first map was started is embedded again
        let changed = first_source_path(&photos);
        fs::File::options().write(true).open(&changed).unwrap().set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(3600)).unwrap();
        let third = build("third");
        let last = fs::read(third.join("look.bin")).unwrap();
        assert_ne!(&last[..DIM * 2], &marker[..], "a changed file is embedded again");
        fs::remove_dir_all(&temp).unwrap();

        fn first_source_path(photos: &[serde_json::Value]) -> PathBuf {
            PathBuf::from(photos[0]["source"].as_str().unwrap())
        }
    }

    /// A folder of photos becomes a map's vectors, also when the build is paused and continued several times on the way.
    #[test]
    fn builds_from_a_folder_and_resumes() {
        let root = crate::data_root();
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

        let model = models::test_hub();
        let dir = create(&maps, "test map", &[source.clone()], 1, models::DEFAULT).unwrap();
        assert!(create(&maps, "test map", &[source.clone()], 1, models::DEFAULT).is_err(), "a second map with the same name");
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
        assert_eq!(fs::metadata(done.join("look.bin")).unwrap().len() as usize, count * DIM * 2);
        let photos: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(done.join("items.json")).unwrap()).unwrap();
        assert_eq!(photos.len(), count);
        // read where they are
        assert!(Path::new(photos[count - 1]["source"].as_str().unwrap()).exists());
        // the vectors are unit length and not all alike
        let raw = fs::read(done.join("look.bin")).unwrap();
        let vector = |i: usize| -> Vec<f32> { raw[i * DIM * 2..(i + 1) * DIM * 2].chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect() };
        let (a, b) = (vector(0), vector(count - 1));
        let norm = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cos: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        eprintln!("norm {norm:.3}, cosine between the first and last photo {cos:.3}");
        assert!((norm - 1.0).abs() < 0.01 && cos < 0.999);
        // laid out, with named regions, and the viewer's loader takes it
        assert_eq!(fs::metadata(done.join("layout.bin")).unwrap().len() as usize, count * 8);
        let names: serde_json::Value = serde_json::from_slice(&fs::read(done.join("regions.json")).unwrap()).unwrap();
        let levels = names["levels"].as_array().unwrap();
        let named: usize = levels.iter().map(|l| l.as_array().unwrap().len()).sum();
        eprintln!("{} levels, {named} named regions, e.g. {:?}", levels.len(), levels.iter().flat_map(|l| l.as_array().unwrap().iter().take(2).map(|r| r["name"].as_str().unwrap().to_string())).collect::<Vec<_>>());
        assert!(named > 0);
        assert!(done.join("pages.bin").exists(), "baked before the map is finished");
        let data = crate::data::MapData::load(&done, &crate::data::Progress::default()).unwrap();
        assert_eq!(data.n(), count);
        fs::remove_dir_all(&temp).unwrap();
    }

    /// Goes on with a build (BUILD: its .building folder, run from the stage in its job.json) to the end, and prints each stage's time
    /// and the most memory the process held during it: for seeing how a big map's build scales. Linux (it reads /proc/self/status).
    /// A build whose photos are all embedded doesn't load the model (nor touch the graphics card) if the vocabulary is cached.
    /// BUILD=... cargo test --release stages_of_a_real_build -- --ignored --nocapture
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore]
    fn stages_of_a_real_build() {
        let dir = PathBuf::from(std::env::var("BUILD").expect("BUILD: a map's .building folder"));
        // the process's own memory (not the files it maps: the system can drop those whenever it needs the room)
        let rss = || -> usize {
            let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
            status.lines().find_map(|l| l.strip_prefix("RssAnon:")).and_then(|v| v.trim().trim_end_matches(" kB").parse().ok()).unwrap_or(0) / 1024
        };
        // what freed memory the allocator still holds is given back between stages, so that each stage's peak is its own
        let trim = || unsafe { libc::malloc_trim(0) };
        // no model is loaded unless a stage needs it (photos still to embed, a vocabulary not yet learned)
        let hub = models::Hub::start(crate::data_root(), || {});
        trim();
        let base = rss();
        eprintln!("before the build: {base} MB; below, the most above that during each stage");
        let run = Run::start(dir, crate::data_root(), hub, || {}).expect("a job in BUILD");
        let (mut stage, mut started, mut peak) = (run.state().stage, Instant::now(), 0);
        // and the steps inside a stage, by what it says it is doing
        let (mut step, mut step_started, mut step_peak) = (String::new(), Instant::now(), 0);
        let report = |stage: Stage, started: Instant, peak: usize| eprintln!("{:<32} {:7.1}s  {:6} MB", stage.label(), started.elapsed().as_secs_f32(), peak.saturating_sub(base));
        let report_step = |step: &str, started: Instant, peak: usize| eprintln!("    {:<44} {:7.1}s  {:6} MB", step, started.elapsed().as_secs_f32(), peak.saturating_sub(base));
        while run.running() {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let state = run.state();
            let now_step: String = state.message.chars().take(44).collect();
            if now_step != step || state.stage != stage {
                if !step.is_empty() {
                    report_step(&step, step_started, step_peak);
                }
                (step, step_started, step_peak) = (now_step, Instant::now(), 0);
            }
            if state.stage != stage {
                report(stage, started, peak);
                trim();
                (stage, started, peak) = (state.stage, Instant::now(), 0);
            }
            let now = rss();
            peak = peak.max(now);
            step_peak = step_peak.max(now);
        }
        report_step(&step, step_started, step_peak);
        report(stage, started, peak);
        assert!(run.state().error.is_none(), "{:?}", run.state().error);
    }

    /// The same photos made into a map with each model: the second model's map is made with its own vectors (nothing copied from the
    /// first model's map, though the files are the same and unchanged), has its own size of vector in the manifest, and is named.
    /// Swaps the loaded model, so run it alone: cargo test --release another_model -- --ignored --test-threads=1
    #[test]
    #[ignore]
    fn another_model_makes_its_own_vectors() {
        let root = crate::data_root();
        let temp = std::env::temp_dir().join(format!("photo-map-models-test-{}", std::process::id()));
        let (source, maps) = (temp.join("photos"), temp.join("maps"));
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&maps).unwrap();
        let hour_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
        for i in 0..24 {
            let img = image::RgbImage::from_fn(96 + (i % 4) as u32 * 8, 80, |x, y| image::Rgb([(i * 9) as u8, (x * 2) as u8, (y * 3 + i as u32) as u8]));
            let path = source.join(format!("p{i:02}.png"));
            img.save(&path).unwrap();
            fs::File::options().write(true).open(&path).unwrap().set_modified(hour_ago).unwrap();
        }
        let hub = models::test_hub();
        let build = |name: &str, model: &str| {
            let dir = create(&maps, name, &[source.clone()], 1, model).unwrap();
            let run = Run::start(dir, root.clone(), hub.clone(), || {}).unwrap();
            while run.running() {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            assert!(run.state().error.is_none(), "{:?}", run.state().error);
            finished_dir(&maps, name)
        };
        let other = build("by siglip 256", models::SIGLIP_BASE_256);
        // a row the model would never make, to see whether it is copied
        let mut raw = fs::read(other.join("look.bin")).unwrap();
        let marker: Vec<u8> = (0..DIM).flat_map(|d| half::f16::from_f32(if d == 0 { 1.0 } else { 0.0 }).to_le_bytes()).collect();
        raw[..DIM * 2].copy_from_slice(&marker);
        fs::write(other.join("look.bin"), &raw).unwrap();

        let siglip = build("by siglip", models::SIGLIP_BASE_512);
        let spec = models::spec(models::SIGLIP_BASE_512).unwrap();
        let manifest: serde_json::Value = serde_json::from_slice(&fs::read(siglip.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["embedding"]["model"], models::SIGLIP_BASE_512);
        assert_eq!(manifest["embedding"]["dim"], spec.dim);
        let vectors = fs::read(siglip.join("look.bin")).unwrap();
        assert_eq!(vectors.len(), 24 * spec.dim * 2);
        assert_ne!(&vectors[..DIM * 2], &marker[..], "row 0 was not copied from the other model's map");
        assert!(fs::metadata(siglip.join("regions.json")).is_ok(), "the regions are named");
        assert!(root.join(".cache/vocab/curated2.google_siglip2-base-patch16-512.f16").exists(), "its own vocabulary vectors");
        // and a second map by the same model does reuse them
        let again = build("by siglip again", models::SIGLIP_BASE_512);
        assert_eq!(fs::read(again.join("look.bin")).unwrap(), vectors);
        fs::remove_dir_all(&temp).unwrap();
    }
}
