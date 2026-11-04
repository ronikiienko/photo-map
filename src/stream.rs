//! All photo pixels on the GPU, one unified system: every level (1 px ... 1024 px) of every photo is a picture
//! in a slot of that level's pool (see pages.rs). A table gives, per photo, how many levels are loaded and the
//! slot of each. Levels 1 ... 16 px are always loaded (slot = photo id); from 32 px up they are streamed for the photos on
//! screen (from 16 px on screen, when the 32 px picture starts to fade in), each into a free or least-recently-seen slot of
//! its level: 32 px is read from the baked file (pages_bc7.bin) by reader threads (it may have to come from disk); from 64 px up, worker threads decode
//! the original file, fit each wanted level and BC7-compress it, and the main thread copies the blocks in. A photo's
//! streamed levels are always contiguous from 32 px up, so evicting a level also drops the ones above it. The table's first
//! entry is the loaded level count (levels 0 ... count - 1 are there). Only the entries that changed are uploaded again.

use crate::data::{MapData, Point};
use crate::photo;
use crate::pages::{BAKED, LEVELS, Layout, PAGE, RESIDENT, slot_size};
use glam::Vec2;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};

const WORKERS: usize = 6;
const MAX_IN_FLIGHT: usize = 24;
/// Threads reading baked 32 px pictures (a KB each, in 8 rows of blocks: from disk, a few reads at once are faster).
const READERS: usize = 4;
/// Baked 32 px pictures asked for and not back yet, at most (few enough that a zoom elsewhere doesn't wait behind them).
const MAX_READS: usize = 1024;
/// u32s per photo in the table: loaded level count, original's longer side, 2 unused, then one slot per level.
pub const ENTRY: usize = 4 + LEVELS;
const NONE: u32 = u32::MAX;
/// A photo whose file couldn't be read is tried again after this many frames (a disk may be plugged in meanwhile).
const RETRY_FRAMES: u64 = 300;

/// Slot bookkeeping of one streamed level (32 px up).
struct Pool {
    holder: Vec<u32>,
    last_seen: Vec<u64>,
    /// Slots no photo holds.
    free: Vec<usize>,
    /// Slots not on screen this frame, the one seen longest ago last (listed once a frame, when the first is needed).
    old: Vec<usize>,
    old_frame: u64,
}

impl Pool {
    fn new(slots: usize) -> Self {
        Self { holder: vec![NONE; slots], last_seen: vec![0; slots], free: (0..slots).rev().collect(), old: Vec::new(), old_frame: 0 }
    }

    /// A free slot, else the one seen longest ago that isn't on screen this frame.
    fn take(&mut self, frame: u64) -> Option<usize> {
        if let Some(s) = self.free.pop() {
            return Some(s);
        }
        if self.old_frame != frame {
            self.old = (0..self.holder.len()).filter(|&s| self.last_seen[s] < frame).collect();
            self.old.sort_unstable_by_key(|&s| std::cmp::Reverse(self.last_seen[s]));
            self.old_frame = frame;
        }
        // taken since it was listed: seen this frame now
        std::iter::from_fn(|| self.old.pop()).find(|&s| self.last_seen[s] < frame)
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
    /// The file could be read.
    found: bool,
    levels: Vec<Level>,
}

/// What the lens shows (lens.rs): the world box under it, its centre, and a photo's and a copy's longer side on screen in it.
#[derive(Clone, Copy)]
pub struct LensRequest {
    pub min: Vec2,
    pub max: Vec2,
    pub center: Vec2,
    pub size_px: f32,
    pub copy_px: f32,
}

pub struct Streamer {
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub table: wgpu::Buffer,
    /// First page of each level, for the shader's address maths.
    pub first_pages: wgpu::Buffer,
    layout: Layout,
    table_cpu: Vec<u32>,
    /// The whole table still has to be uploaded (at the start).
    upload_all: bool,
    /// Photos whose entries changed since the last upload.
    changed: Vec<u32>,
    pools: Vec<Pool>,
    loading: Vec<bool>,
    /// The frame a photo's file was found not to be there (0: it was, or it hasn't been asked for).
    missing: Vec<u64>,
    in_flight: usize,
    jobs: mpsc::Sender<Job>,
    results: mpsc::Receiver<Loaded>,
    /// 32 px pictures to read from the baked file, and those read: the photo and its BC7 blocks.
    reads: mpsc::Sender<usize>,
    read: mpsc::Receiver<(usize, Vec<u8>)>,
    reading: Vec<bool>,
    reads_in_flight: usize,
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

        // the always-loaded levels, baked as whole pages, straight from the file (the 32 px pages after them are copied from as needed)
        let started = std::time::Instant::now();
        let page_bytes = (PAGE * PAGE) as usize; // BC7: 1 byte per pixel
        let resident: u32 = (0..RESIDENT).map(|l| layout.pages[l]).sum();
        assert_eq!(data.pages.len(), crate::pages::baked_bytes(n), "pages_bc7.bin doesn't match the photos: delete it, the viewer bakes it again");
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
        let pools = (RESIDENT..LEVELS)
            .map(|l| Pool::new(layout.slots(l) as usize))
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
        let (reads, read_rx) = mpsc::channel::<usize>();
        let (read_tx, read) = mpsc::channel();
        let read_rx = Arc::new(Mutex::new(read_rx));
        for _ in 0..READERS {
            let (read_rx, read_tx, pages) = (read_rx.clone(), read_tx.clone(), data.pages.clone());
            std::thread::spawn(move || {
                loop {
                    let Ok(id) = read_rx.lock().unwrap().recv() else { return };
                    if read_tx.send((id, read_baked(&pages, n, id))).is_err() {
                        return;
                    }
                }
            });
        }
        Self {
            texture,
            view,
            table,
            first_pages,
            layout,
            table_cpu,
            upload_all: true,
            changed: Vec::new(),
            pools,
            loading: vec![false; n],
            missing: vec![0; n],
            in_flight: 0,
            jobs,
            results,
            reads,
            read,
            reading: vec![false; n],
            reads_in_flight: 0,
            frame: 1,
        }
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// (slot size, used, total) of each streamed level.
    pub fn usage(&self) -> Vec<(u32, usize, usize)> {
        self.pools
            .iter()
            .enumerate()
            .map(|(k, p)| (slot_size(RESIDENT + k), p.holder.iter().filter(|&&h| h != NONE).count(), p.holder.len()))
            .collect()
    }

    /// A photo's sharpest loaded picture's longer side, and the original's.
    pub fn loaded_px(&self, id: usize) -> (u32, u32) {
        let e = &self.table_cpu[id * ENTRY..(id + 1) * ENTRY];
        let long = e[1].max(1);
        (slot_size(e[0] as usize - 1).min(long), long)
    }

    /// The sharpest level worth having: the one whose slot size is closest to the original's longer side (at least 64 px).
    fn max_level(&self, id: usize) -> usize {
        let long = self.table_cpu[id * ENTRY + 1].max(1) as f32;
        (long.log2().round() as usize).clamp(BAKED, LEVELS - 1)
    }

    /// The photo's file couldn't be read the last time it was wanted at full size (moved, deleted, or on a disk not plugged in).
    pub fn missing(&self, id: usize) -> bool {
        self.missing[id] != 0
    }

    /// Drop levels `from..` of a photo (keeps its loaded levels contiguous).
    fn drop_levels(&mut self, id: usize, from: usize) {
        let count = self.table_cpu[id * ENTRY] as usize;
        let from = from.max(RESIDENT);
        for l in from..count {
            let slot = self.table_cpu[id * ENTRY + 4 + l];
            self.pools[l - RESIDENT].holder[slot as usize] = NONE;
            self.pools[l - RESIDENT].free.push(slot as usize);
            self.table_cpu[id * ENTRY + 4 + l] = NONE;
        }
        if from < count {
            self.table_cpu[id * ENTRY] = from as u32;
            self.changed.push(id as u32);
        }
    }

    /// Put a level of a photo into a free or least-recently-seen slot of its pool (the level below must be loaded): its BC7
    /// blocks, `w` x `h` px, `bytes_per_row` apart. False if every slot is on screen this frame.
    fn place(&mut self, queue: &wgpu::Queue, id: usize, level: usize, blocks: &[u8], w: u32, h: u32, bytes_per_row: u32) -> bool {
        let pool = level - RESIDENT;
        let Some(slot) = self.pools[pool].take(self.frame) else { return false };
        let old = self.pools[pool].holder[slot];
        if old != NONE {
            self.drop_levels(old as usize, level);
            self.pools[pool].free.pop(); // `slot` itself, freed last in this pool
        }
        let (page, x, y) = self.layout.place(level, slot as u32);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: page }, aspect: wgpu::TextureAspect::All },
            blocks,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: Some(h / 4) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.pools[pool].holder[slot] = id as u32;
        self.pools[pool].last_seen[slot] = self.frame;
        self.table_cpu[id * ENTRY + 4 + level] = slot as u32;
        self.table_cpu[id * ENTRY] = level as u32 + 1;
        self.changed.push(id as u32);
        true
    }

    /// Once per frame: take in finished loads, then request what the current view needs.
    /// `view_min`/`view_max`: the visible world box; `size_px`: a photo's longer side on screen at size 1 (data.size). `copies`: the matches' copies
    /// beside the world (beside.rs), which photo is where, and their longer side on screen. `lens`: what the lens shows, if it is
    /// open; its photos come first.
    pub fn update(&mut self, queue: &wgpu::Queue, data: &MapData, view_min: Vec2, view_max: Vec2, center: Vec2, size_px: f32, copies: (&[usize], &[Point], f32), lens: Option<LensRequest>) {
        self.frame += 1;
        let frame = self.frame;
        // the level matching an on-screen size, plus the next one up to blend towards
        let level = |px: f32| (px.max(1.0).log2().floor() as usize + 1).min(LEVELS - 1);

        // the photos on screen (a photo and its copy may both be, and the lens may show it bigger): the level each wants, and
        // how soon (the lens's first, then nearest the middle)
        let mut visible: Vec<(usize, usize, (u8, f32))> = Vec::new();
        let mut look = |view_min: Vec2, view_max: Vec2, center: Vec2, size_px: f32, copy_px: f32, group: u8| {
            if level(size_px * data.max_size()) >= RESIDENT {
                data.grid.within(&data.points, view_min, view_max, |i| {
                    let want = level(size_px * data.size(i));
                    if want >= RESIDENT {
                        visible.push((i, want, (group, Vec2::new(data.points[i].x, data.points[i].y).distance_squared(center))));
                    }
                });
            }
            let (photos, at, _) = copies;
            let want = level(copy_px);
            if want >= RESIDENT {
                for (&i, p) in photos.iter().zip(at) {
                    let p = Vec2::new(p.x, p.y);
                    if p.cmpge(view_min).all() && p.cmple(view_max).all() {
                        visible.push((i, want, (group, p.distance_squared(center))));
                    }
                }
            }
        };
        if let Some(l) = lens {
            look(l.min, l.max, l.center, l.size_px, l.copy_px, 0);
        }
        look(view_min, view_max, center, size_px, copies.2, 1);
        let mut wanted: Vec<((u8, f32), usize, usize)> = Vec::new();
        for (i, want, soon) in visible {
            let count = self.table_cpu[i * ENTRY] as usize;
            for l in RESIDENT..count {
                let slot = self.table_cpu[i * ENTRY + 4 + l] as usize;
                self.pools[l - RESIDENT].last_seen[slot] = frame;
            }
            let need = want.min(self.max_level(i));
            if count <= need {
                wanted.push((soon, i, need));
            }
        }
        // a photo wanted twice (itself and its copy): once, at the bigger of the two
        wanted.sort_by(|a, b| a.1.cmp(&b.1).then(b.2.cmp(&a.2)).then(a.0.0.cmp(&b.0.0)));
        wanted.dedup_by_key(|w| w.1);

        // finished loads, level by level from the bottom: each into a free or least-recently-seen slot
        while let Ok(done) = self.results.try_recv() {
            self.in_flight -= 1;
            self.loading[done.id] = false;
            self.missing[done.id] = if done.found { 0 } else { frame };
            for lv in done.levels {
                // a level below went missing meanwhile (the rest would leave a gap), or every slot is on screen
                if self.table_cpu[done.id * ENTRY] as usize != lv.level || !self.place(queue, done.id, lv.level, &lv.blocks, lv.w, lv.h, lv.w * 4) {
                    break;
                }
            }
        }

        // the lens's first, then nearest the screen centre; never more than the pools can hold
        wanted.sort_by(|a, b| a.0.0.cmp(&b.0.0).then(a.0.1.total_cmp(&b.0.1)));
        wanted.truncate(self.pools[BAKED - RESIDENT].holder.len() * 3 / 4);
        // 32 px: read from the baked file by the readers; in as they come back (unless the photo already has it)
        let s = slot_size(RESIDENT);
        while let Ok((id, blocks)) = self.read.try_recv() {
            self.reads_in_flight -= 1;
            self.reading[id] = false;
            if self.table_cpu[id * ENTRY] as usize == RESIDENT {
                self.place(queue, id, RESIDENT, &blocks, s, s, s / 4 * 16);
            }
        }
        for &(_, id, _) in &wanted {
            if self.reads_in_flight >= MAX_READS {
                break;
            }
            if self.table_cpu[id * ENTRY] as usize == RESIDENT && !self.reading[id] && self.reads.send(id).is_ok() {
                self.reads_in_flight += 1;
                self.reading[id] = true;
            }
        }
        // from 64 px: decoded from the original, once the 32 px picture is in
        for &(_, id, to) in &wanted {
            if self.in_flight >= MAX_IN_FLIGHT {
                break;
            }
            let from = self.table_cpu[id * ENTRY] as usize;
            let retry = self.missing[id] == 0 || frame >= self.missing[id] + RETRY_FRAMES;
            if from < BAKED || from > to || self.loading[id] || !retry {
                continue;
            }
            if self.jobs.send(Job { id, path: data.path(id), from, to }).is_ok() {
                self.in_flight += 1;
                self.loading[id] = true;
            }
        }

        // the table: whole at the start, then only the changed entries, neighbours joined into one write
        if self.upload_all {
            queue.write_buffer(&self.table, 0, bytemuck::cast_slice(&self.table_cpu));
            self.upload_all = false;
            self.changed.clear();
        }
        self.changed.sort_unstable();
        self.changed.dedup();
        let mut k = 0;
        while k < self.changed.len() {
            let first = self.changed[k] as usize;
            let mut last = first;
            while k + 1 < self.changed.len() && self.changed[k + 1] as usize == last + 1 {
                k += 1;
                last += 1;
            }
            let entries = &self.table_cpu[first * ENTRY..(last + 1) * ENTRY];
            queue.write_buffer(&self.table, (first * ENTRY * 4) as u64, bytemuck::cast_slice(entries));
            k += 1;
        }
        self.changed.clear();
    }
}

/// Reader: photo `id`'s 32 px picture from the baked file (`photos` photos), a row of BC7 blocks at a time.
fn read_baked(pages: &[u8], photos: usize, id: usize) -> Vec<u8> {
    let row = (slot_size(RESIDENT) / 4 * 16) as usize;
    let (at, step) = crate::pages::baked_at(RESIDENT, photos, id);
    (0..slot_size(RESIDENT) as usize / 4).flat_map(|k| &pages[at + k * step..at + k * step + row]).copied().collect()
}

/// Worker: decode the original once; for each level fit it (longer side = min(slot size, original)), pad to
/// whole 4 x 4 blocks by repeating the edge, and BC7-compress.
fn decode(job: &Job) -> Loaded {
    let mut loaded = Loaded { id: job.id, found: false, levels: Vec::new() };
    let Some(rgba) = photo::load_for(&job.path, |_, _| slot_size(job.to)) else { return loaded };
    loaded.found = true;
    let long = rgba.width().max(rgba.height());
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
        let blocks = crate::bc7::compress(&padded, w4, h4);
        loaded.levels.push(Level { level, w: w4, h: h4, blocks });
    }
    loaded
}
