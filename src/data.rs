//! A map bundle, plus a spatial grid for "which photo is under the mouse". Bundles are made in Python
//! (src/photomap.py: maps/<name>.photomap/):
//!     manifest.json   name, photo count
//!     photos.json     per photo: file (originals/NNNNNN.ext if copied into the map, else empty), source, width, height, title, caption (optional), meta (any strings, for people to read),
//!                     facts (optional: camera, taken, place from its EXIF, for search's camera: date: place:)
//!     originals/      the photos copied into the map (maps made before kept links to all of them here); photo i = position i in photos.json
//!     layout.bin      per photo: x, y (float32, 0 ... WORLD)
//!     regions.json    optional: named regions per zoom level (Label below); region_names.json, if there, is used instead
//!     embeddings.bin  per photo: float16 vector of the picture (search and Groups)
//!     tree.bin        per photo: its parent in the tree that paths between photos follow (tree.rs)
//!     pages_bc7.bin   the baked levels (1 ... 32 px), made while the map is built (bake.rs)

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

#[derive(Deserialize)]
pub struct Manifest {
    pub name: String,
    pub photos: usize,
    #[serde(default)]
    embedding: Embedding,
    /// A photo's longer side on the map, world units: set when the layout was spread (spread) while the map was made. A map made
    /// before has its photos spread when it is opened.
    #[serde(default)]
    photo_side: Option<f32>,
}

/// What the vectors of a map are: which model made them, and how many numbers each has. A map without it (made before models could
/// be chosen) was made with the default.
#[derive(Deserialize)]
struct Embedding {
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_dim")]
    dim: usize,
}

impl Default for Embedding {
    fn default() -> Self {
        Self { model: default_model(), dim: default_dim() }
    }
}

fn default_model() -> String {
    crate::models::DEFAULT.to_string()
}

fn default_dim() -> usize {
    crate::models::spec(crate::models::DEFAULT).map_or(768, |s| s.dim)
}

impl Manifest {
    /// The id of the model the map was made with (and has to be searched with).
    pub fn model(&self) -> &str {
        &self.embedding.model
    }

    /// How many numbers each vector of the map has.
    pub fn dim(&self) -> usize {
        self.embedding.dim
    }
}

#[derive(Deserialize)]
pub struct Photo {
    /// Path inside the bundle (originals/NNNNNN.ext) if the photo was copied into the map; empty: it is read at `source`.
    pub file: String,
    /// The full path of the file the photo was made from: where it is read from if it wasn't copied. Builds use it to find vectors
    /// already made for the same file (reuse.rs).
    #[serde(default)]
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub title: String,
    /// Shown above the photo when zoomed in (never searched: search.rs); empty if the photo has none.
    #[serde(default)]
    pub caption: String,
    /// Whatever the image file said about itself: description, date taken, camera, location...
    #[serde(default)]
    pub meta: serde_json::Map<String, serde_json::Value>,
    /// What its EXIF says that search can pick by (boxed: most photos of a big map have none, and this keeps them small).
    #[serde(default)]
    pub facts: Option<Box<Facts>>,
}

/// A photo's camera, when it was taken and where, read from its EXIF when the map was made (files::facts). Empty: not known.
#[derive(Serialize, Deserialize, Default, Debug, PartialEq)]
pub struct Facts {
    /// "Google Pixel 7", "Canon EOS 550D".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub camera: String,
    /// "2019-07-04 14:00" (local time, as the camera had it).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub taken: String,
    /// The nearest town to where it was taken: "Lviv, Lviv Oblast, Ukraine" (places.rs).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub place: String,
}

impl Photo {
    /// Where the photo is read from: its copy in the map `bundle`, else where it came from.
    pub fn path(&self, bundle: &Path) -> PathBuf {
        if self.file.is_empty() { PathBuf::from(&self.source) } else { bundle.join(&self.file) }
    }
}

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
    pub points: Vec<Point>,
    /// How big each photo is drawn, times `side` (the Groups view: the most typical photos biggest); empty: all 1.
    pub sizes: Vec<f32>,
    /// A photo's longer side on the map (world units): the median distance to the nearest photo before they were spread.
    pub side: f32,
    /// The baked levels as BC7 pages (bake.rs): 1 ... 16 px uploaded to the GPU as they are, 32 px copied out as needed (stream.rs).
    pub pages: std::sync::Arc<Mmap>,
    pub grid: Grid,
    /// Empty if the bundle has no regions.json.
    pub regions: Regions,
}

impl MapData {
    pub fn load(bundle: &Path, progress: &Progress) -> std::io::Result<Self> {
        progress.stage("reading the map's files", 0);
        let read = |name: &str| std::fs::read(bundle.join(name));
        let manifest: Manifest = serde_json::from_slice(&read("manifest.json")?)?;
        let photos: Vec<Photo> = serde_json::from_slice(&read("photos.json")?)?;
        let points: Vec<Point> = bytemuck::pod_collect_to_vec(&read("layout.bin")?);
        let n = manifest.photos;
        if photos.len() != n || points.len() != n {
            return Err(std::io::Error::other("bundle files don't match"));
        }
        let pages = unsafe { Mmap::map(&File::open(bundle.join("pages_bc7.bin"))?)? };
        if pages.len() != crate::pages::baked_bytes(n) {
            return Err(std::io::Error::other("pages_bc7.bin is not the size of the photos"));
        }
        let (side, points) = match manifest.photo_side {
            Some(side) => (side, points),
            None => {
                let side = crate::median_neighbour_distance(&points);
                (side, spread(&points, side))
            }
        };
        let grid = Grid::new(&points);
        let mut regions: Regions = read("region_names.json").or_else(|_| read("regions.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        for level in &mut regions.levels {
            level.sort_by(|a, b| b.photos.cmp(&a.photos));
        }
        Ok(Self { bundle: bundle.to_path_buf(), manifest, photos, points, sizes: Vec::new(), side, pages: std::sync::Arc::new(pages), grid, regions })
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

    pub fn path(&self, id: usize) -> PathBuf {
        self.photos[id].path(&self.bundle)
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
