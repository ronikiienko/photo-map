//! The baked levels (1 px ... 32 px) as BC7 pages: pages_bc7.bin; 1 ... 16 px are uploaded to the GPU as they are, 32 px pictures
//! are copied out as they come on screen (stream.rs).
//! Made as the last step of building a map (build.rs), and kept in the bundle.
//! Every level L holds each picture (the photos, then the videos' frames) in a slot of 2^L px (fitted into the top-left corner,
//! real shape kept, small ones enlarged). Slots sit on PAGE x PAGE pages in a grid of PAGE / 2^L per row; slot = picture index.
//! Smaller levels are averaged from the photo's own pixels only, never from the empty rest of the tile.
//! Pages are written level 0 first, each as BC7 blocks (16 bytes per 4 x 4 pixels).

use crate::data::Progress;
use crate::pages::{BAKED, PAGE as PAGE_U32};
use crate::photo;
use rayon::prelude::*;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const PAGE: usize = PAGE_U32 as usize;
const LEVELS: usize = BAKED; // 1, 2, 4, ... 32 px
const TOP: usize = 1 << (LEVELS - 1);
/// Rows compressed per task: a multiple of 4 (a BC7 block row).
const STRIP: usize = 64;

/// Bakes the pictures in these files, in this order (picture i: `pictures[i]`).
pub fn bake(bundle: &Path, pictures: &[PathBuf], progress: &Progress) -> io::Result<()> {
    let n = pictures.len();
    let started = Instant::now();
    let step = |msg: String| eprintln!("[{:5.0}s] {msg}", started.elapsed().as_secs_f32());
    step(format!("{}: baking the small levels of {n} photos", bundle.display()));
    progress.stage("preparing small pictures (reading every photo, then compressing)", n);

    // what the encoder makes of a block of zeros (transparent black), the same wherever it is
    let empty = {
        let mut block = [0u8; 16];
        block.copy_from_slice(&crate::bc7::compress(&[0u8; 4 * 4 * 4], 4, 4));
        block
    };
    // one page compressed: `px` holds its rows from the top, as far as photos reach (rounded up to whole strips); the photos fill
    // the first `used_rows` rows, the rest is zeros, which compress to one block, repeated
    let compress = |px: &[u8], used_rows: usize| -> Vec<u8> {
        let mut blocks = vec![0u8; PAGE * PAGE];
        blocks.par_chunks_mut(PAGE / 4 * STRIP / 4 * 16).enumerate().for_each(|(strip, dst)| {
            if strip * STRIP >= used_rows {
                for block in dst.chunks_exact_mut(16) {
                    block.copy_from_slice(&empty);
                }
                return;
            }
            dst.copy_from_slice(&crate::bc7::compress(&px[strip * PAGE * STRIP * 4..(strip + 1) * PAGE * STRIP * 4], PAGE as u32, STRIP as u32));
        });
        blocks
    };

    // A batch is a page of the biggest level: its photos are read and fitted to 32 px, that page is compressed, and their halved
    // tiles go on to the pages of the smaller levels (4x as many photos a page each time), which are compressed when full. So
    // only a batch of tiles and a page per level are held, whatever the number of photos. Each level goes to a file of its own,
    // put together (level 0 first) at the end.
    let batch = (PAGE / TOP) * (PAGE / TOP);
    let level_file = |level: usize| bundle.join(format!("pages_bc7.{level}.tmp"));
    let mut files = (0..LEVELS).map(|l| std::fs::File::create(level_file(l)).map(io::BufWriter::new)).collect::<io::Result<Vec<_>>>()?;
    let mut pages: Vec<Vec<u8>> = vec![Vec::new(); LEVELS];
    let done = AtomicUsize::new(0);
    for first in (0..n).step_by(batch) {
        let end = (first + batch).min(n);
        // 1. every photo of the batch fitted to 32 px (longer side) in the corner of a 32 x 32 tile
        // `content`: how much of each tile is the photo (width, height); the rest of the tile is padding, which no average may use
        let mut tiles = vec![0u8; (end - first) * TOP * TOP * 4];
        let mut content = vec![(0usize, 0usize); end - first];
        tiles.par_chunks_mut(TOP * TOP * 4).zip(content.par_iter_mut()).enumerate().for_each(|(k, (tile, size))| {
            if let Some(img) = photo::load_for(&pictures[first + k], |_, _| TOP as u32) {
                let img = photo::fit(&img, TOP as u32, true);
                *size = (img.width() as usize, img.height() as usize);
                for (y, row) in img.as_raw().chunks(img.width() as usize * 4).enumerate() {
                    tile[y * TOP * 4..y * TOP * 4 + row.len()].copy_from_slice(row);
                }
            }
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            if d % 64 == 0 {
                progress.set_done(d);
            }
            if d % 10_000 == 0 {
                step(format!("decoded {d} / {n}"));
            }
        });

        // 2. from 32 px down: the tiles onto their level's page, a page compressed when it is full (or has the last photo); the
        // tiles halved for the next level
        for level in (0..LEVELS).rev() {
            let s = 1 << level;
            let per_row = PAGE / s;
            let per_page = per_row * per_row;
            let page_first = first / per_page * per_page;
            let px = &mut pages[level];
            let rows = |photos: usize| photos.div_ceil(per_row) * s;
            let reach = rows(end - page_first).div_ceil(STRIP) * STRIP;
            px.resize(reach * PAGE * 4, 0);
            for i in first..end {
                let (local, k) = (i - page_first, i - first);
                let (dx, dy) = ((local % per_row) * s, (local / per_row) * s);
                for row in 0..s {
                    let a = (k * s * s + row * s) * 4;
                    let b = ((dy + row) * PAGE + dx) * 4;
                    px[b..b + s * 4].copy_from_slice(&tiles[a..a + s * 4]);
                }
            }
            if end % per_page == 0 || end == n {
                files[level].write_all(&compress(px, rows(end - page_first)))?;
                px.clear();
            }
            if level > 0 {
                // average 2 x 2 pixels: tiles of s px -> s/2 px, using only the photo's own pixels
                let h = s / 2;
                let mut next = vec![0u8; (end - first) * h * h * 4];
                next.par_chunks_mut(h * h * 4).zip(content.par_iter_mut()).enumerate().for_each(|(k, (t, size))| {
                    *size = halve(&tiles[k * s * s * 4..(k + 1) * s * s * 4], s, *size, t);
                });
                tiles = next;
            }
        }
    }
    let tmp = bundle.join("pages_bc7.tmp");
    let mut file = io::BufWriter::new(std::fs::File::create(&tmp)?);
    for (level, part) in files.into_iter().enumerate() {
        drop(part.into_inner().map_err(|e| e.into_error())?);
        io::copy(&mut std::fs::File::open(level_file(level))?, &mut file)?;
        std::fs::remove_file(level_file(level))?;
    }
    file.flush()?;
    drop(file);
    std::fs::rename(tmp, bundle.join("pages_bc7.bin"))?;
    step("pages_bc7.bin written".into());
    Ok(())
}

/// One tile of `s` x `s` px (RGBA) halved into `out` (s/2 x s/2). Each new pixel is the average of the photo's pixels among its 2 x 2
/// (`content`: the photo's width and height in the tile; the rest is padding and counts for nothing). Returns the photo's size in the
/// halved tile. A pixel with no photo in it stays 0: the viewer never reads it.
fn halve(src: &[u8], s: usize, content: (usize, usize), out: &mut [u8]) -> (usize, usize) {
    let h = s / 2;
    for y in 0..h {
        for x in 0..h {
            let mut sum = [0u32; 4];
            let mut count = 0;
            for (yy, xx) in [(2 * y, 2 * x), (2 * y, 2 * x + 1), (2 * y + 1, 2 * x), (2 * y + 1, 2 * x + 1)] {
                if xx < content.0 && yy < content.1 {
                    for c in 0..4 {
                        sum[c] += src[(yy * s + xx) * 4 + c] as u32;
                    }
                    count += 1;
                }
            }
            for c in 0..4 {
                out[(y * h + x) * 4 + c] = if count > 0 { ((sum[c] + count / 2) / count) as u8 } else { 0 };
            }
        }
    }
    (content.0.div_ceil(2), content.1.div_ceil(2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Progress;

    /// A wide white photo (32 x 16 in a 32 x 32 tile) is still white at 1 px: the padding is not in the average.
    #[test]
    fn padding_stays_out_of_the_average() {
        let mut tile = vec![0u8; 32 * 32 * 4];
        for y in 0..16 {
            for x in 0..32 {
                tile[(y * 32 + x) * 4..(y * 32 + x) * 4 + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let (mut size, mut s) = ((32, 16), 32);
        while s > 1 {
            let mut next = vec![0u8; s / 2 * (s / 2) * 4];
            size = halve(&tile, s, size, &mut next);
            tile = next;
            s /= 2;
        }
        assert_eq!(tile, vec![255, 255, 255, 255]);
        assert_eq!(size, (1, 1));
    }

    /// Baking a few photos writes the baked levels, in pages.rs's size.
    #[test]
    fn bakes_the_resident_levels() {
        let dir = std::env::temp_dir().join(format!("photo-map-bake-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("originals")).unwrap();
        let photos: Vec<PathBuf> = (0..5)
            .map(|i| {
                let file = dir.join(format!("originals/{i}.png"));
                image::RgbaImage::from_pixel(100 + i * 20, 80, image::Rgba([50 * i as u8, 120, 200, 255])).save(&file).unwrap();
                file
            })
            .collect();
        bake(&dir, &photos, &Progress::default()).unwrap();
        let len = std::fs::metadata(dir.join("pages_bc7.bin")).unwrap().len() as usize;
        assert_eq!(len, crate::pages::baked_bytes(photos.len()));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The shortcut for empty rows: a strip of zeros compresses to one block, the same everywhere.
    #[test]
    fn empty_strips_are_one_repeated_block() {
        let encode = |data: &[u8], width: usize, height: usize, out: &mut [u8]| out.copy_from_slice(&crate::bc7::compress(data, width as u32, height as u32));
        let mut one = [0u8; 16];
        encode(&[0u8; 64], 4, 4, &mut one);
        let mut strip = vec![0u8; PAGE / 4 * STRIP / 4 * 16];
        encode(&vec![0u8; PAGE * STRIP * 4], PAGE, STRIP, &mut strip);
        assert!(strip.chunks_exact(16).all(|b| b == one));
    }
}
