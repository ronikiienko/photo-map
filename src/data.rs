//! A map bundle, plus a spatial grid for "which photo is under the mouse". Bundles are made in Python
//! (src/photomap.py: maps/<name>.photomap/):
//!     manifest.json   name, photo count
//!     photos.json     per photo: file (originals/NNNNNN.ext), source, width, height, title, caption (optional), meta (any strings, for people to read)
//!     originals/      the images; photo i = position i in photos.json
//!     layout.bin      per photo: x, y (float32, 0 ... WORLD)
//!     regions.json    optional: named regions per zoom level (Label below); region_names.json, if there, is used instead
//!     embeddings.bin  per photo: float16 vector (not used by the viewer)
//!     pages_bc7.bin   the always-loaded levels, made while the map is built (bake.rs)

use memmap2::Mmap;
use serde::Deserialize;
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
    /// Path inside the bundle (originals/NNNNNN.ext).
    pub file: String,
    /// The full path of the file the photo was made from. Builds use it to find vectors already made for the same file (reuse.rs).
    #[serde(default)]
    #[allow(dead_code)]
    pub source: String,
    pub width: u32,
    pub height: u32,
    pub title: String,
    /// Shown above the photo when zoomed in, and searched along with the picture (captions.bin); empty if the photo has none.
    #[serde(default)]
    pub caption: String,
    /// Whatever the image file said about itself: description, date taken, camera, location...
    #[serde(default)]
    #[allow(dead_code)] // not shown yet
    pub meta: serde_json::Map<String, serde_json::Value>,
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
    /// The always-loaded levels as BC7 pages (bake.rs), uploaded to the GPU as they are.
    pub pages: Mmap,
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
        if pages.len() != crate::pages::resident_bytes(n) {
            return Err(std::io::Error::other("pages_bc7.bin is not the size of the photos"));
        }
        let grid = Grid::new(&points);
        let mut regions: Regions = read("region_names.json").or_else(|_| read("regions.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        for level in &mut regions.levels {
            level.sort_by(|a, b| b.photos.cmp(&a.photos));
        }
        Ok(Self { bundle: bundle.to_path_buf(), manifest, photos, points, pages, grid, regions })
    }

    pub fn n(&self) -> usize {
        self.photos.len()
    }

    /// Put the photos somewhere else (another view of the same map): the grid follows.
    pub fn set_points(&mut self, points: Vec<Point>) {
        self.grid = Grid::new(&points);
        self.points = points;
    }

    pub fn path(&self, id: usize) -> PathBuf {
        self.bundle.join(&self.photos[id].file)
    }

    /// Longer side of the original, in pixels.
    pub fn native_long(&self, id: usize) -> u32 {
        self.photos[id].width.max(self.photos[id].height)
    }

    /// Photo size relative to a square: (1, 0.67) for a 3:2 landscape photo.
    pub fn shape(&self, id: usize) -> glam::Vec2 {
        let p = &self.photos[id];
        glam::Vec2::new(p.width as f32, p.height as f32) / self.native_long(id).max(1) as f32
    }
}

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
