//! A map: maps/<name>.photomap/, made by build.rs. Its items are the photos and videos on it, in order (item i is row i of every
//! per-item file); its pictures are everything with pixels: the items (a video's poster), then every video's frames, video after
//! video (picture n + k is frame k).
//!     manifest.json   name, when it was made, how many items (videos included) and videos, a photo's side on the map, and which
//!                     model made the vectors (Manifest)
//!     items.json      per item: where its file is, its size, what the file says about itself, and a video's length and frame
//!                     times (Item)
//!     regions.json    named regions per zoom level (Regions)
//!     look.bin        per picture: float16 vector (an item's places it, search and Groups; a video's is its poster frame's)
//!     layout.bin      per item: x, y (float32, 0 ... WORLD)
//!     tree.bin        per item: its parent in the tree that paths between photos follow (tree.rs)
//!     pages.bin       per picture: its baked levels 1 ... 32 px (bake.rs)
//!     cover.png       the map's picture on the Maps page
//!     frames/<i>/     video i's poster.jpg (its picture on the map, POSTER_SIDE) and its frames 0.jpg, 1.jpg... (FRAME_SIDE)
//! A photo is read where it is (its source); nothing is stored that can be worked out (titles, poster and frame sizes, file names).

use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// What opening a map is doing, for the window to show (the thread opening it writes, the window reads).
#[derive(Default)]
pub struct Progress {
    text: Mutex<String>,
    done: AtomicUsize,
    total: AtomicUsize,
}

impl Progress {
    /// Starting a stage of `total` steps (0 = not countable).
    pub fn stage(&self, text: impl Into<String>, total: usize) {
        *self.text.lock().unwrap() = text.into();
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn set_done(&self, done: usize) {
        self.done.store(done, Ordering::Relaxed);
    }

    /// The stage's text and how far along it is (0..1, None if not countable).
    pub fn now(&self) -> (String, Option<f32>) {
        let total = self.total.load(Ordering::Relaxed);
        let fraction = (total > 0).then(|| self.done.load(Ordering::Relaxed) as f32 / total as f32);
        (self.text.lock().unwrap().clone(), fraction)
    }
}

/// World coordinates run 0..WORLD on both axes.
pub const WORLD: f32 = 10000.0;

/// manifest.json.
#[derive(Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    /// "2026-10-13 21:40" (UTC).
    pub created: String,
    /// Items: photos and videos.
    pub photos: usize,
    pub videos: usize,
    /// A photo's longer side on the map, world units: the layout was spread to it (spread).
    pub photo_side: f32,
    /// The model of look.bin's vectors (the map is searched with it).
    pub look: Vectors,
}

/// Which model made a kind of vectors, and how many numbers each has.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Vectors {
    pub model: String,
    pub dim: usize,
}

impl Manifest {
    /// The id of the model the map was made with (and has to be searched with).
    pub fn model(&self) -> &str {
        &self.look.model
    }

    /// How many numbers each vector of the map has.
    pub fn dim(&self) -> usize {
        self.look.dim
    }
}

/// An item of items.json: a photo or a video.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Item {
    /// The full path of its file: read from there (and builds look for vectors made before by it: reuse.rs).
    pub source: String,
    /// Its size as it is shown (turned upright), px; a video's own.
    pub width: u32,
    pub height: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facts: Option<Box<Facts>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video: Option<Clip>,
}

/// What items.json says of a video.
#[derive(Serialize, Deserialize, Debug, PartialEq, Default)]
pub struct Clip {
    /// Seconds.
    pub duration: f32,
    /// Its frames' times (seconds from the start), in order: frame k is frames/<i>/<k>.jpg.
    pub times: Vec<f32>,
}

/// A photo or a video as the app holds it (from items.json).
pub struct Photo {
    /// The full path of its file.
    pub source: String,
    /// Its file's name without the extension.
    pub title: String,
    /// The size of the picture it is drawn from, px: a video's poster's (`video` has its own).
    pub width: u32,
    pub height: u32,
    /// What its EXIF says that search can pick by (boxed: most photos of a big map have none, and this keeps them small).
    pub facts: Option<Box<Facts>>,
    /// A video: how long it is and its own size. None: a photo.
    pub video: Option<Video>,
}

/// What a video is, apart from its poster.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Video {
    /// Seconds.
    pub duration: f32,
    /// Its frames' size as it is shown (turned upright), px.
    pub width: u32,
    pub height: u32,
}

/// What a photo's file says about itself, read once when the map is made (a photo's EXIF: files::facts; a video's container,
/// as ffmpeg tells it: files::video_facts): search picks by camera, date and place, and Details shows it all. Empty: not known.
#[derive(Serialize, Deserialize, Default, Debug, PartialEq)]
pub struct Facts {
    /// "Google Pixel 7", "Canon EOS 550D".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub camera: String,
    /// "2019-07-04 14:00" (local time, as the camera had it; a video's as its file says, often UTC).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub taken: String,
    /// The nearest town to where it was taken: "Lviv, Lviv Oblast, Ukraine" (places.rs).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub place: String,
    /// Where it was taken: latitude, longitude.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<[f32; 2]>,
    /// "EF-S 18-55mm".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub lens: String,
    /// "1/120 s, f/1.8, 4.4 mm, ISO 100".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub exposure: String,
}

/// A file's name without its extension.
pub fn title(source: &str) -> String {
    Path::new(source).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Where item `i` is drawn from: a photo's own file, a video's poster in the map.
pub fn item_path(bundle: &Path, i: usize, item: &Item) -> PathBuf {
    if item.video.is_some() { poster(bundle, i) } else { PathBuf::from(&item.source) }
}

/// Video `i`'s poster.
pub fn poster(bundle: &Path, i: usize) -> PathBuf {
    bundle.join(format!("frames/{i}/poster.jpg"))
}

/// Video `i`'s frame `k`, in the map.
pub fn frame_file(i: usize, k: usize) -> String {
    format!("frames/{i}/{k}.jpg")
}

/// A poster's longer side at most (px): a video's frames are taken out at this size (build::video_frames), the poster is one of them.
pub const POSTER_SIDE: u32 = 1024;

/// The size of a video's poster (and of its frames as they are taken out), from the video's: ffmpeg's scale in video_frames, the
/// longer side at most POSTER_SIDE, the other kept even.
pub fn poster_size(width: u32, height: u32) -> (u32, u32) {
    let even = |v: f64| ((v / 2.0).round() as u32 * 2).max(2);
    if width >= height {
        let w = width.min(POSTER_SIDE);
        (w, even(height as f64 * w as f64 / width.max(1) as f64))
    } else {
        let h = height.min(POSTER_SIDE);
        (even(width as f64 * h as f64 / height.max(1) as f64), h)
    }
}

/// The size of a video's frames as kept in the map, from its poster's: made small as photo::fit does.
pub fn frame_size(poster: (u32, u32)) -> (u32, u32) {
    let long = poster.0.max(poster.1);
    if long <= FRAME_SIDE {
        return poster;
    }
    let s = FRAME_SIDE as f32 / long as f32;
    (((poster.0 as f32 * s).round() as u32).max(1), ((poster.1 as f32 * s).round() as u32).max(1))
}

/// A frame of a video: kept small (FRAME_SIDE), shown beside the world when it is a search's best match in its video. It is
/// picture n + k.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The item (the video) it is from.
    pub video: usize,
    /// Seconds from the video's start.
    pub time: f32,
    /// Its picture, in the map.
    pub file: String,
    pub width: u32,
    pub height: u32,
}

/// A frame's longer side, as kept in the map (px).
pub const FRAME_SIDE: u32 = 256;

/// A named area of the map (regions.json).
#[derive(Deserialize)]
pub struct Label {
    pub name: String,
    /// The region's most central photo.
    pub x: f32,
    pub y: f32,
    /// Root-mean-square distance of its photos from its centre, in world units.
    pub spread: f32,
    pub photos: usize,
}

#[derive(Deserialize, Default)]
pub struct Regions {
    /// Coarse first; each level's regions biggest first.
    pub levels: Vec<Vec<Label>>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct MapData {
    pub bundle: PathBuf,
    pub manifest: Manifest,
    pub photos: Vec<Photo>,
    /// The videos' frames (none if the map has no videos).
    pub frames: Vec<Frame>,
    pub points: Vec<Point>,
    /// How big each photo is drawn, times `side` (the Groups view: the most typical photos biggest); empty: all 1.
    pub sizes: Vec<f32>,
    /// A photo's longer side on the map (world units): the median distance to the nearest photo before they were spread.
    pub side: f32,
    /// The baked levels as BC7 pages (bake.rs): 1 ... 16 px uploaded to the GPU as they are, 32 px copied out as needed (stream.rs).
    pub pages: std::sync::Arc<Mmap>,
    pub grid: Grid,
    /// Empty if the map has no regions.json.
    pub regions: Regions,
}

impl MapData {
    pub fn load(bundle: &Path, progress: &Progress) -> std::io::Result<Self> {
        progress.stage("reading the map's files", 0);
        let read = |name: &str| std::fs::read(bundle.join(name));
        let manifest: Manifest = serde_json::from_slice(&read("manifest.json")?)?;
        let items: Vec<Item> = serde_json::from_slice(&read("items.json")?)?;
        let points: Vec<Point> = bytemuck::pod_collect_to_vec(&read("layout.bin")?);
        let n = manifest.photos;
        if items.len() != n || points.len() != n {
            return Err(std::io::Error::other("the map's files don't match"));
        }
        // the frames, video after video
        let mut frames = Vec::new();
        let photos: Vec<Photo> = items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                let title = title(&item.source);
                let (width, height, video) = match item.video {
                    Some(clip) => {
                        let poster = poster_size(item.width, item.height);
                        let (fw, fh) = frame_size(poster);
                        frames.extend(clip.times.iter().enumerate().map(|(k, &time)| Frame { video: i, time, file: frame_file(i, k), width: fw, height: fh }));
                        (poster.0, poster.1, Some(Video { duration: clip.duration, width: item.width, height: item.height }))
                    }
                    None => (item.width, item.height, None),
                };
                Photo { source: item.source, title, width, height, facts: item.facts, video }
            })
            .collect();
        let pages = unsafe { Mmap::map(&File::open(bundle.join("pages.bin"))?)? };
        if pages.len() != crate::pages::baked_bytes(n + frames.len()) {
            return Err(std::io::Error::other("pages.bin is not the size of the pictures"));
        }
        let grid = Grid::new(&points);
        let mut regions: Regions = read("regions.json").ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        for level in &mut regions.levels {
            level.sort_by(|a, b| b.photos.cmp(&a.photos));
        }
        let side = manifest.photo_side;
        Ok(Self { bundle: bundle.to_path_buf(), manifest, photos, frames, points, sizes: Vec::new(), side, pages: std::sync::Arc::new(pages), grid, regions })
    }

    pub fn n(&self) -> usize {
        self.photos.len()
    }

    /// Put the photos somewhere else (another view of the same map), at these sizes (empty: all 1): the grid follows.
    pub fn set_points(&mut self, points: Vec<Point>, sizes: Vec<f32>) {
        self.grid = Grid::new(&points);
        self.points = points;
        self.sizes = sizes;
    }

    /// How big the photo is drawn (1 but in the Groups view).
    pub fn size(&self, id: usize) -> f32 {
        self.sizes.get(id).copied().unwrap_or(1.0)
    }

    /// The biggest any photo is drawn.
    pub fn max_size(&self) -> f32 {
        self.sizes.iter().copied().fold(1.0, f32::max)
    }

    /// Where photo `id` is drawn from: its file, or a video's poster.
    pub fn path(&self, id: usize) -> PathBuf {
        let p = &self.photos[id];
        if p.video.is_some() { poster(&self.bundle, id) } else { PathBuf::from(&p.source) }
    }

    /// Photos and frames: everything with pixels (pages.rs, stream.rs).
    pub fn pictures(&self) -> usize {
        self.photos.len() + self.frames.len()
    }

    /// Where picture `id`'s file is (a photo's, or a frame's in the map).
    pub fn picture_path(&self, id: usize) -> PathBuf {
        match id.checked_sub(self.n()) {
            Some(k) => self.bundle.join(&self.frames[k].file),
            None => self.path(id),
        }
    }

    /// Picture `id`'s width and height, in pixels.
    fn picture_size(&self, id: usize) -> (u32, u32) {
        match id.checked_sub(self.n()) {
            Some(k) => (self.frames[k].width, self.frames[k].height),
            None => (self.photos[id].width, self.photos[id].height),
        }
    }

    /// Picture `id`'s longer side, in pixels.
    pub fn picture_long(&self, id: usize) -> u32 {
        let (w, h) = self.picture_size(id);
        w.max(h)
    }

    /// Picture `id`'s shape, its longer side 1.
    pub fn picture_form(&self, id: usize) -> glam::Vec2 {
        let (w, h) = self.picture_size(id);
        glam::Vec2::new(w as f32, h as f32) / w.max(h).max(1) as f32
    }

    /// Longer side of the original, in pixels.
    pub fn native_long(&self, id: usize) -> u32 {
        self.photos[id].width.max(self.photos[id].height)
    }

    /// Photo size on the map relative to a square of `side`: (1, 0.67) for a 3:2 landscape photo, times its size.
    pub fn shape(&self, id: usize) -> glam::Vec2 {
        self.form(id) * self.size(id)
    }

    /// The photo's shape alone (its longer side 1): for its copies beside the world, which are never bigger.
    pub fn form(&self, id: usize) -> glam::Vec2 {
        let p = &self.photos[id];
        glam::Vec2::new(p.width as f32, p.height as f32) / self.native_long(id).max(1) as f32
    }
}

/// The photos moved apart so that none covers another: in turn, each stays where it is if its square (`side`, the photos' longer side on
/// the map) overlaps none placed before, else it goes to the nearest spot that is free, on rings around it. No grid: a grid shimmers
/// when zoomed out. Measured on three maps: 36-39% of photos move, half of them by under 0.75 of a side, the farthest 2-6 sides;
/// 88-97% keep their 10 nearest.
pub fn spread(points: &[Point], side: f32) -> Vec<Point> {
    let g = side * GAP;
    let mut placed: std::collections::HashMap<(i32, i32), Vec<Point>> = std::collections::HashMap::with_capacity(points.len());
    let square = |p: Point| ((p.x / g).floor() as i32, (p.y / g).floor() as i32);
    let free = |placed: &std::collections::HashMap<(i32, i32), Vec<Point>>, p: Point| {
        let (cx, cy) = square(p);
        (cy - 1..=cy + 1).all(|y| (cx - 1..=cx + 1).all(|x| placed.get(&(x, y)).is_none_or(|v| v.iter().all(|q| (q.x - p.x).abs().max((q.y - p.y).abs()) >= g))))
    };
    points
        .iter()
        .map(|&p| {
            let mut at = p;
            // rings a quarter of the gap apart, points on each a quarter of the gap apart
            let mut r = 0.0;
            while !free(&placed, at) {
                r += g * 0.25;
                let steps = ((std::f32::consts::TAU * r / (g * 0.25)) as usize).max(6);
                if let Some(q) = (0..steps).map(|k| k as f32 * std::f32::consts::TAU / steps as f32).map(|a| Point { x: p.x + r * a.cos(), y: p.y + r * a.sin() }).find(|&q| free(&placed, q)) {
                    at = q;
                }
            }
            placed.entry(square(at)).or_default().push(at);
            at
        })
        .collect()
}

/// How close two photos' centres may be along x or y, in photo sizes (1: they may touch).
const GAP: f32 = 1.0;

/// G x G cells over the world; each cell lists the photos inside it.
pub struct Grid {
    start: Vec<u32>,
    order: Vec<u32>,
}

const G: usize = 64;

impl Grid {
    fn cell(x: f32, y: f32) -> (usize, usize) {
        let c = |v: f32| ((v / WORLD * G as f32) as usize).min(G - 1);
        (c(x), c(y))
    }

    fn new(points: &[Point]) -> Self {
        let mut start = vec![0u32; G * G + 1];
        let index = |p: &Point| {
            let (cx, cy) = Self::cell(p.x, p.y);
            cy * G + cx
        };
        for p in points {
            start[index(p) + 1] += 1;
        }
        for c in 0..G * G {
            start[c + 1] += start[c];
        }
        let mut fill = start.clone();
        let mut order = vec![0u32; points.len()];
        for (i, p) in points.iter().enumerate() {
            let c = index(p);
            order[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        Self { start, order }
    }

    /// Photos whose centre lies inside the world box.
    pub fn within(&self, points: &[Point], min: glam::Vec2, max: glam::Vec2, mut f: impl FnMut(usize)) {
        let (x0, y0) = Self::cell(min.x.max(0.0), min.y.max(0.0));
        let (x1, y1) = Self::cell(max.x.max(0.0), max.y.max(0.0));
        for cy in y0..=y1 {
            for c in cy * G + x0..=cy * G + x1 {
                for &i in &self.order[self.start[c] as usize..self.start[c + 1] as usize] {
                    let p = points[i as usize];
                    if p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y {
                        f(i as usize);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Photos piled on one spot end up side by side, near where they were; photos apart enough stay where they are.
    #[test]
    fn spread_photos_hide_none() {
        let side = 10.0;
        let mut points: Vec<Point> = (0..200).map(|_| Point { x: 500.0, y: 500.0 }).collect();
        points.extend((0..300).map(|i| Point { x: (i % 20) as f32 * 3.0 + 100.0, y: (i / 20) as f32 * 3.0 + 100.0 }));
        points.extend((0..50).map(|i| Point { x: 900.0 + i as f32 * side, y: 900.0 }));
        let spread = spread(&points, side);
        for (i, a) in spread.iter().enumerate() {
            for b in &spread[i + 1..] {
                assert!((a.x - b.x).abs().max((a.y - b.y).abs()) >= side * 0.999, "overlap");
            }
            let moved = ((a.x - points[i].x).powi(2) + (a.y - points[i].y).powi(2)).sqrt();
            assert!(moved < side * 12.0, "{i} moved {moved}");
            if i >= 500 {
                assert_eq!(moved, 0.0, "{i} was apart enough");
            }
        }
    }
}
