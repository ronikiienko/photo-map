//! All photo pixels on the GPU, one unified system: every level (1 px ... 1024 px) of every photo is a picture
//! in a slot of that level's pool (see pages.rs). A table gives, per photo, how many levels are loaded and the
//! slot of each. Only level 0 (1 px, the photo's colour) is always loaded; levels from 64 px up are streamed for the photos on
//! screen (from 32 px, when the 64 px picture starts to fade in):
//! worker threads decode the original file, fit each wanted level and BC7-compress it, and the main thread copies
//! the blocks into a free or least-recently-seen slot. A photo's streamed levels are always contiguous from 64 px up, so
//! evicting a level also drops the ones above it. The table's first entry is the loaded level count: 1 (just the colour),
//! or the top streamed level + 1 (levels 1 ... 5 never exist). The table is re-uploaded whole whenever it changed.

use crate::data::MapData;
use crate::photo;
use crate::pages::{LEVELS, Layout, PAGE, RESIDENT, STREAMED, slot_size};
use glam::Vec2;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

const WORKERS: usize = 6;
const MAX_IN_FLIGHT: usize = 24;
/// u32s per photo in the table: loaded level count, original's longer side, 2 unused, then one slot per level.
pub const ENTRY: usize = 4 + LEVELS;
const NONE: u32 = u32::MAX;

/// Slot bookkeeping of one streamed level.
struct Pool {
    holder: Vec<u32>,
    last_seen: Vec<u64>,
}

impl Pool {
    /// A free slot, else the one seen longest ago that isn't on screen this frame.
    fn take(&self, frame: u64) -> Option<usize> {
        if let Some(s) = self.holder.iter().position(|&h| h == NONE) {
            return Some(s);
        }
        (0..self.holder.len()).filter(|&s| self.last_seen[s] < frame).min_by_key(|&s| self.last_seen[s])
    }
}

struct Job {
    id: usize,
    path: PathBuf,
    from: usize,
    to: usize,
}

/// One level of one photo, ready to copy: BC7 blocks of a w x h area (both multiples of 4).
struct Level {
    level: usize,
    w: u32,
    h: u32,
    blocks: Vec<u8>,
}

struct Loaded {
    id: usize,
    levels: Vec<Level>,
}

pub struct Streamer {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub table: wgpu::Buffer,
    /// First page of each level, for the shader's address maths.
    pub first_pages: wgpu::Buffer,
    layout: Layout,
    table_cpu: Vec<u32>,
    dirty: bool,
    pools: Vec<Pool>,
    loading: Vec<bool>,
    in_flight: usize,
    jobs: mpsc::Sender<Job>,
    results: mpsc::Receiver<Loaded>,
    frame: u64,
}

impl Streamer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, data: &MapData) -> Self {
        let n = data.n();
        let layout = Layout::new(n);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pages"),
            size: wgpu::Extent3d { width: PAGE, height: PAGE, depth_or_array_layers: layout.total_pages },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bc7RgbaUnorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // the always-loaded levels, baked as whole pages, straight from the file
        let started = std::time::Instant::now();
        let page_bytes = (PAGE * PAGE) as usize; // BC7: 1 byte per pixel
        let resident: u32 = (0..STREAMED).map(|l| layout.pages[l]).sum();
        assert_eq!(data.pages.len(), resident as usize * page_bytes, "pages_bc7.bin doesn't match the photos: delete it, the viewer bakes it again");
        for page in 0..resident {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: page }, aspect: wgpu::TextureAspect::All },
                &data.pages[page as usize * page_bytes..(page as usize + 1) * page_bytes],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(PAGE * 4), rows_per_image: Some(PAGE / 4) },
                wgpu::Extent3d { width: PAGE, height: PAGE, depth_or_array_layers: 1 },
            );
            if page % 8 == 7 {
                queue.submit([]); // flush staging memory now and then
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
        }
        eprintln!("{resident} always-loaded pages uploaded ({:.1}s), {} pages in total", started.elapsed().as_secs_f32(), layout.total_pages);

        let mut table_cpu = vec![NONE; n * ENTRY];
        for i in 0..n {
            let e = &mut table_cpu[i * ENTRY..(i + 1) * ENTRY];
            e[0] = RESIDENT as u32;
            e[1] = data.native_long(i);
            e[2] = 0;
            e[3] = 0;
            for l in 0..RESIDENT {
                e[4 + l] = i as u32; // the always-loaded levels: slot = photo id
            }
        }
        let table = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("level table"),
            size: (table_cpu.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut first = [0u32; 16];
        first[..LEVELS].copy_from_slice(&layout.first_page);
        let first_pages = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
            label: Some("first pages"),
            contents: bytemuck::cast_slice(&first),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pools = (STREAMED..LEVELS)
            .map(|l| {
                let slots = layout.slots(l) as usize;
                Pool { holder: vec![NONE; slots], last_seen: vec![0; slots] }
            })
            .collect();

        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (result_tx, results) = mpsc::channel();
        let job_rx = Arc::new(Mutex::new(job_rx));
        for _ in 0..WORKERS {
            let (job_rx, result_tx) = (job_rx.clone(), result_tx.clone());
            std::thread::spawn(move || {
                loop {
                    let Ok(job) = job_rx.lock().unwrap().recv() else { return };
                    if result_tx.send(decode(&job)).is_err() {
                        return;
                    }
                }
            });
        }
        Self { texture, view, table, first_pages, layout, table_cpu, dirty: true, pools, loading: vec![false; n], in_flight: 0, jobs, results, frame: 1 }
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// (slot size, used, total) of each streamed level.
    pub fn usage(&self) -> Vec<(u32, usize, usize)> {
        self.pools
            .iter()
            .enumerate()
            .map(|(k, p)| (slot_size(STREAMED + k), p.holder.iter().filter(|&&h| h != NONE).count(), p.holder.len()))
            .collect()
    }

    /// The sharpest level worth having: the one whose slot size is closest to the original's longer side.
    fn max_level(&self, id: usize) -> usize {
        let long = self.table_cpu[id * ENTRY + 1].max(1) as f32;
        (long.log2().round() as usize).clamp(STREAMED, LEVELS - 1)
    }

    /// Drop levels `from..` of a photo (keeps its loaded levels contiguous).
    fn drop_levels(&mut self, id: usize, from: usize) {
        let count = self.table_cpu[id * ENTRY] as usize;
        for l in from.max(STREAMED)..count {
            let slot = self.table_cpu[id * ENTRY + 4 + l];
            self.pools[l - STREAMED].holder[slot as usize] = NONE;
            self.table_cpu[id * ENTRY + 4 + l] = NONE;
        }
        if from < count {
            self.table_cpu[id * ENTRY] = if from <= STREAMED { RESIDENT as u32 } else { from as u32 };
            self.dirty = true;
        }
    }

    /// Once per frame: take in finished loads, then request what the current view needs.
    /// `view_min`/`view_max`: the visible world box; `size_px`: a photo's longer side on screen.
    pub fn update(&mut self, queue: &wgpu::Queue, data: &MapData, view_min: Vec2, view_max: Vec2, center: Vec2, size_px: f32) {
        self.frame += 1;
        let frame = self.frame;
        // the level matching the on-screen size, plus the next one up to blend towards
        let want = (size_px.max(1.0).log2().floor() as usize + 1).min(LEVELS - 1);

        let mut wanted: Vec<(f32, usize, usize, usize)> = Vec::new();
        if want >= STREAMED {
            let mut visible = Vec::new();
            data.grid.within(&data.points, view_min, view_max, |i| visible.push(i));
            for i in visible {
                let count = self.table_cpu[i * ENTRY] as usize;
                for l in STREAMED..count {
                    let slot = self.table_cpu[i * ENTRY + 4 + l] as usize;
                    self.pools[l - STREAMED].last_seen[slot] = frame;
                }
                let need = want.min(self.max_level(i));
                if need >= STREAMED && count <= need && !self.loading[i] {
                    let p = data.points[i];
                    wanted.push((Vec2::new(p.x, p.y).distance_squared(center), i, count, need));
                }
            }
        }

        // finished loads, level by level from the bottom: each into a free or least-recently-seen slot
        while let Ok(done) = self.results.try_recv() {
            self.in_flight -= 1;
            self.loading[done.id] = false;
            for lv in done.levels {
                let id = done.id;
                if self.table_cpu[id * ENTRY].max(STREAMED as u32) as usize != lv.level {
                    break; // a level below went missing meanwhile: the rest would leave a gap
                }
                let pool = lv.level - STREAMED;
                let Some(slot) = self.pools[pool].take(frame) else { break };
                let old = self.pools[pool].holder[slot];
                if old != NONE {
                    self.drop_levels(old as usize, lv.level);
                }
                let (page, x, y) = self.layout.place(lv.level, slot as u32);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: page }, aspect: wgpu::TextureAspect::All },
                    &lv.blocks,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(lv.w * 4), rows_per_image: Some(lv.h / 4) },
                    wgpu::Extent3d { width: lv.w, height: lv.h, depth_or_array_layers: 1 },
                );
                self.pools[pool].holder[slot] = id as u32;
                self.pools[pool].last_seen[slot] = frame;
                self.table_cpu[id * ENTRY + 4 + lv.level] = slot as u32;
                self.table_cpu[id * ENTRY] = lv.level as u32 + 1;
                self.dirty = true;
            }
        }

        // nearest to the screen centre first; never more than the smallest streamed pool can hold
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, id, from, to) in wanted.iter().take(self.pools[0].holder.len() * 3 / 4) {
            if self.in_flight >= MAX_IN_FLIGHT {
                break;
            }
            if self.jobs.send(Job { id, path: data.path(id), from: from.max(STREAMED), to }).is_ok() {
                self.in_flight += 1;
                self.loading[id] = true;
            }
        }

        if self.dirty {
            queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&self.table_cpu));
            self.dirty = false;
        }
    }
}

/// Worker: decode the original once; for each level fit it (longer side = min(slot size, original)), pad to
/// whole 4 x 4 blocks by repeating the edge, and BC7-compress.
fn decode(job: &Job) -> Loaded {
    let mut loaded = Loaded { id: job.id, levels: Vec::new() };
    let Some(rgba) = photo::load_for(&job.path, |_, _| slot_size(job.to)) else { return loaded };
    let long = rgba.width().max(rgba.height());
    let settings = intel_tex_2::bc7::opaque_ultra_fast_settings();
    for level in job.from..=job.to {
        let fitted = photo::fit(&rgba, slot_size(level).min(long), false);
        let (w, h) = fitted.dimensions();
        let (w4, h4) = (w.div_ceil(4) * 4, h.div_ceil(4) * 4);
        let mut padded = vec![0u8; (w4 * h4 * 4) as usize];
        for y in 0..h4 {
            for x in 0..w4 {
                let p = fitted.get_pixel(x.min(w - 1), y.min(h - 1));
                let o = ((y * w4 + x) * 4) as usize;
                padded[o..o + 4].copy_from_slice(&p.0);
            }
        }
        let surface = intel_tex_2::RgbaSurface { data: &padded, width: w4, height: h4, stride: w4 * 4 };
        let blocks = intel_tex_2::bc7::compress_blocks(&settings, &surface);
        loaded.levels.push(Level { level, w: w4, h: h4, blocks });
    }
    loaded
}
