//! The always-loaded levels (1 px ... 64 px) as BC7 pages: pages_bc7.bin, uploaded to the GPU as it is.
//! Made by the viewer the first time it opens a bundle without one (bundles come from Python, which has no fast
//! BC7 encoder), then kept in the bundle.
//! Every level L holds each photo in a slot of 2^L px (photo fitted into the top-left corner, real shape kept,
//! small photos enlarged). Slots sit on PAGE x PAGE pages in a grid of PAGE / 2^L per row; slot = photo index.
//! Pages are written level 0 first, each as BC7 blocks (16 bytes per 4 x 4 pixels).

use crate::data::Photo;
use crate::pages::{PAGE as PAGE_U32, STREAMED};
use crate::photo;
use rayon::prelude::*;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const PAGE: usize = PAGE_U32 as usize;
const LEVELS: usize = STREAMED; // 1, 2, 4, ... 64 px
const TOP: usize = 1 << (LEVELS - 1);
/// Rows compressed per task: a multiple of 4 (a BC7 block row).
const STRIP: usize = 64;

pub fn bake(bundle: &Path, photos: &[Photo]) -> io::Result<()> {
    let n = photos.len();
    let started = Instant::now();
    let step = |msg: String| eprintln!("[{:5.0}s] {msg}", started.elapsed().as_secs_f32());
    step(format!("{}: baking the always-loaded levels of {n} photos", bundle.display()));

    // 1. every photo fitted to 64 px (longer side) in the corner of a 64 x 64 tile
    let done = AtomicUsize::new(0);
    let mut tiles = vec![0u8; n * TOP * TOP * 4];
    tiles.par_chunks_mut(TOP * TOP * 4).enumerate().for_each(|(i, tile)| {
        if let Some(img) = photo::load(&bundle.join(&photos[i].file)) {
            let img = photo::fit(&img, TOP as u32, true);
            for (y, row) in img.as_raw().chunks(img.width() as usize * 4).enumerate() {
                tile[y * TOP * 4..y * TOP * 4 + row.len()].copy_from_slice(row);
            }
        }
        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 10_000 == 0 {
            step(format!("decoded {d} / {n}"));
        }
    });

    // 2. from 64 px down: pack each level's tiles onto pages and compress; halve the tiles for the next level
    let settings = intel_tex_2::bc7::opaque_basic_settings();
    let mut levels: Vec<Vec<u8>> = vec![Vec::new(); LEVELS];
    for level in (0..LEVELS).rev() {
        let s = 1 << level;
        let per_row = PAGE / s;
        let per_page = per_row * per_row;
        let pages = n.div_ceil(per_page);
        let mut out = Vec::with_capacity(pages * PAGE * PAGE);
        for page in 0..pages {
            let mut px = vec![0u8; PAGE * PAGE * 4];
            let first = page * per_page;
            for i in first..(first + per_page).min(n) {
                let local = i - first;
                let (dx, dy) = ((local % per_row) * s, (local / per_row) * s);
                for row in 0..s {
                    let a = (i * s * s + row * s) * 4;
                    let b = ((dy + row) * PAGE + dx) * 4;
                    px[b..b + s * 4].copy_from_slice(&tiles[a..a + s * 4]);
                }
            }
            let mut blocks = vec![0u8; PAGE * PAGE];
            px.par_chunks(PAGE * STRIP * 4).zip(blocks.par_chunks_mut(PAGE / 4 * STRIP / 4 * 16)).for_each(|(src, dst)| {
                let surface = intel_tex_2::RgbaSurface { data: src, width: PAGE as u32, height: STRIP as u32, stride: (PAGE * 4) as u32 };
                intel_tex_2::bc7::compress_blocks_into(&settings, &surface, dst);
            });
            out.extend_from_slice(&blocks);
        }
        levels[level] = out;
        step(format!("level {level} ({s} px): {pages} page(s)"));
        if level > 0 {
            // average 2 x 2 pixels: tiles of s px -> s/2 px
            let h = s / 2;
            let mut next = vec![0u8; n * h * h * 4];
            next.par_chunks_mut(h * h * 4).enumerate().for_each(|(i, t)| {
                let src = &tiles[i * s * s * 4..(i + 1) * s * s * 4];
                for y in 0..h {
                    for x in 0..h {
                        for c in 0..4 {
                            let at = |yy: usize, xx: usize| src[(yy * s + xx) * 4 + c] as u32;
                            let sum = at(2 * y, 2 * x) + at(2 * y, 2 * x + 1) + at(2 * y + 1, 2 * x) + at(2 * y + 1, 2 * x + 1);
                            t[(y * h + x) * 4 + c] = ((sum + 2) / 4) as u8;
                        }
                    }
                }
            });
            tiles = next;
        }
    }
    let tmp = bundle.join("pages_bc7.tmp");
    let mut file = io::BufWriter::new(std::fs::File::create(&tmp)?);
    for l in &levels {
        file.write_all(l)?;
    }
    file.flush()?;
    drop(file);
    std::fs::rename(tmp, bundle.join("pages_bc7.bin"))?;
    step("pages_bc7.bin written".into());
    Ok(())
}
