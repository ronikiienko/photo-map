//! The map files written by src/prep_photo_map.py, plus a spatial grid for "which photo is under the mouse".

use memmap2::Mmap;
use serde::Deserialize;
use std::fs::File;
use std::path::{Path, PathBuf};

/// World coordinates run 0..WORLD on both axes.
pub const WORLD: f32 = 10000.0;

#[derive(Deserialize)]
pub struct Meta {
    pub n: usize,
    pub k: usize,
    pub tile: u32,
    pub side: u32,
    pub mips: u32,
    pub cap: Vec<String>,
    pub img: Vec<String>,
    pub art: Vec<Vec<String>>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RawPoint {
    pub x: f32,
    pub y: f32,
    pub region: u32,
}

pub struct MapData {
    pub root: PathBuf,
    pub meta: Meta,
    pub points: Vec<RawPoint>,
    pub nn: Vec<u32>,
    pub sim: Vec<u8>,
    /// Width and height of each photo inside its tile, in pixels (longer side = tile size).
    pub fill: Vec<[u8; 2]>,
    /// Width and height of each original file: how sharp it can be streamed.
    pub native: Vec<[u16; 2]>,
    /// Raw RGBA tiles: per layer, mip 0 then each smaller mip. Memory-mapped; only read for side-panel thumbnails.
    pub atlas: Mmap,
    /// The always-loaded levels as BC7 pages (bake_pages), uploaded to the GPU as they are.
    pub pages: Mmap,
    /// Lowercased captions + article names, for search.
    pub search_text: Vec<String>,
    pub grid: Grid,
}

impl MapData {
    /// `root` is the EmbeddingTest folder: image paths in meta.json are relative to it.
    pub fn load(root: &Path) -> std::io::Result<Self> {
        let dir = root.join("data/wikivoyage/photo_map");
        let meta: Meta = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
        let points: Vec<RawPoint> = bytemuck::pod_collect_to_vec(&std::fs::read(dir.join("points.bin"))?);
        let nn: Vec<u32> = bytemuck::pod_collect_to_vec(&std::fs::read(dir.join("nn.bin"))?);
        let sim = std::fs::read(dir.join("sim.bin"))?;
        let fill: Vec<[u8; 2]> = bytemuck::pod_collect_to_vec(&std::fs::read(dir.join("fill.bin"))?);
        let native: Vec<[u16; 2]> = bytemuck::pod_collect_to_vec(&std::fs::read(dir.join("native.bin"))?);
        let atlas = unsafe { Mmap::map(&File::open(dir.join("atlas.bin"))?)? };
        let pages = unsafe { Mmap::map(&File::open(dir.join("pages_bc7.bin"))?)? };
        assert_eq!(points.len(), meta.n);
        assert_eq!(nn.len(), meta.n * meta.k);
        let search_text = (0..meta.n)
            .map(|i| format!("{}\n{}", meta.cap[i], meta.art[i].join("\n")).to_lowercase())
            .collect();
        let grid = Grid::new(&points);
        Ok(Self { root: root.to_path_buf(), meta, points, nn, sim, fill, native, atlas, pages, search_text, grid })
    }

    /// One mip level of one atlas layer: raw RGBA (4 bytes per pixel).
    pub fn atlas_level(&self, layer: u32, mip: u32) -> &[u8] {
        self.level(&self.atlas, 4, layer, mip)
    }

    fn level<'a>(&self, file: &'a [u8], bytes_per_pixel: usize, layer: u32, mip: u32) -> &'a [u8] {
        let side = self.meta.side as usize;
        let level_bytes = |m: u32| (side >> m) * (side >> m) * bytes_per_pixel;
        let layer_bytes: usize = (0..self.meta.mips).map(level_bytes).sum();
        let start = layer as usize * layer_bytes + (0..mip).map(level_bytes).sum::<usize>();
        &file[start..start + level_bytes(mip)]
    }

    /// Longer side of the original file, in pixels.
    pub fn native_long(&self, id: usize) -> u32 {
        self.native[id][0].max(self.native[id][1]) as u32
    }

    /// Photo size relative to a square tile: (1, 0.67) for a 3:2 landscape photo.
    pub fn shape(&self, id: usize) -> glam::Vec2 {
        glam::Vec2::new(self.fill[id][0] as f32, self.fill[id][1] as f32) / self.meta.tile as f32
    }

    /// One photo's pixels from the atlas (without the tile's empty part), for the side panel: (width, height, RGBA).
    pub fn tile_rgba(&self, id: usize) -> (usize, usize, Vec<u8>) {
        let (tile, side) = (self.meta.tile as usize, self.meta.side as usize);
        let per_row = side / tile;
        let t = id % (per_row * per_row);
        let level = self.atlas_level((id / (per_row * per_row)) as u32, 0);
        let (x0, y0) = ((t % per_row) * tile, (t / per_row) * tile);
        let (w, h) = (self.fill[id][0] as usize, self.fill[id][1] as usize);
        let mut out = Vec::with_capacity(w * h * 4);
        for y in y0..y0 + h {
            out.extend_from_slice(&level[(y * side + x0) * 4..(y * side + x0 + w) * 4]);
        }
        (w, h, out)
    }

    pub fn neighbours(&self, id: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let k = self.meta.k;
        (0..k).map(move |j| (self.nn[id * k + j] as usize, self.sim[id * k + j] as f32 / 100.0))
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

    fn new(points: &[RawPoint]) -> Self {
        let mut start = vec![0u32; G * G + 1];
        let index = |p: &RawPoint| {
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
    pub fn within(&self, points: &[RawPoint], min: glam::Vec2, max: glam::Vec2, mut f: impl FnMut(usize)) {
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
