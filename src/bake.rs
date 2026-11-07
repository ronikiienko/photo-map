//! The baked levels (1 px ... 32 px) as BC7 pages: pages.bin; 1 ... 16 px are uploaded to the GPU as they are, 32 px pictures
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
    let level_file = |level: usize| bundle.join(format!("pages.{level}.tmp"));
    let mut files = (0..LEVELS).map(|l| std::fs::File::create(level_file(l)).map(io::BufWriter::new)).collect::<io::Result<Vec<_>>>()?;
    let mut pages: Vec<Vec<u8>> = vec![Vec::new(); LEVELS];
    let done = AtomicUsize::new(0);
    // where the time goes, for the log: reading the pictures, compressing the pages, halving the tiles
    let (mut reading, mut compressing, mut halving) = (0f32, 0f32, 0f32);
    for first in (0..n).step_by(batch) {
        let end = (first + batch).min(n);
        let t = Instant::now();
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
        let read_s = t.elapsed().as_secs_f32();
        reading += read_s;
        let (mut compress_s, mut halve_s) = (0f32, 0f32);

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
                let t = Instant::now();
                files[level].write_all(&compress(px, rows(end - page_first)))?;
                compress_s += t.elapsed().as_secs_f32();
                px.clear();
            }
            if level > 0 {
                let t = Instant::now();
                // average 2 x 2 pixels: tiles of s px -> s/2 px, using only the photo's own pixels
                let h = s / 2;
                let mut next = vec![0u8; (end - first) * h * h * 4];
                next.par_chunks_mut(h * h * 4).zip(content.par_iter_mut()).enumerate().for_each(|(k, (t, size))| {
                    *size = halve(&tiles[k * s * s * 4..(k + 1) * s * s * 4], s, *size, t);
                });
                tiles = next;
                halve_s += t.elapsed().as_secs_f32();
            }
        }
        step(format!("pictures {first}..{end}: read {read_s:.1}s, compressed {compress_s:.1}s, halved {halve_s:.2}s"));
        (compressing, halving) = (compressing + compress_s, halving + halve_s);
    }
    step(format!("in all: reading {reading:.1}s, compressing {compressing:.1}s, halving {halving:.2}s"));
    let tmp = bundle.join("pages.tmp");
    let mut file = io::BufWriter::new(std::fs::File::create(&tmp)?);
    for (level, part) in files.into_iter().enumerate() {
        drop(part.into_inner().map_err(|e| e.into_error())?);
        io::copy(&mut std::fs::File::open(level_file(level))?, &mut file)?;
        std::fs::remove_file(level_file(level))?;
    }
    file.flush()?;
    drop(file);
    std::fs::rename(tmp, bundle.join("pages.bin"))?;
    step("pages.bin written".into());
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

    /// How long baking a real map's pictures takes, and where the time goes (the log lines); and what making its videos' frames
    /// small would cost (build::keep_frames: the 1024 px frame read, shrunk to 256 px and written as JPEG), timed on its posters.
    #[test]
    #[ignore = "reads a whole real map's photos: minutes"]
    fn time_real_map() {
        let bundle = crate::data_root().join("maps/takeout-tagged.photomap");
        let data = crate::data::MapData::load(&bundle, &Progress::default()).expect("the map");
        let pictures: Vec<PathBuf> = (0..data.pictures()).map(|i| data.picture_path(i)).collect();
        let posters: Vec<PathBuf> = (0..data.n()).filter(|&i| data.photos[i].video.is_some()).map(|i| data.path(i)).collect();
        let t = Instant::now();
        posters.par_iter().for_each(|file| {
            crate::build::small_frame(file).unwrap(); // None for a small video's: already a frame's size
        });
        let per = t.elapsed().as_secs_f32() / posters.len() as f32;
        // its parts, each over all the posters: decoding (at 512 px), shrinking to 256, writing the JPEG
        let side = crate::data::FRAME_SIDE;
        let t = Instant::now();
        let decoded: Vec<image::RgbaImage> = posters.par_iter().map(|f| photo::load_for(f, |_, _| side).unwrap()).collect();
        let decode_ms = t.elapsed().as_secs_f32() * 1e3 / posters.len() as f32;
        let t = Instant::now();
        let small: Vec<image::RgbaImage> = decoded.par_iter().map(|img| photo::fit(img, side, false)).collect();
        let fit_ms = t.elapsed().as_secs_f32() * 1e3 / posters.len() as f32;
        let t = Instant::now();
        small.par_iter().for_each(|img| {
            libjpeg_turbo_rs::compress(img.as_raw(), img.width() as usize, img.height() as usize, libjpeg_turbo_rs::PixelFormat::Rgba, 85, libjpeg_turbo_rs::Subsampling::S420).unwrap();
        });
        let write_ms = t.elapsed().as_secs_f32() * 1e3 / posters.len() as f32;
        eprintln!("frame parts: decode {decode_ms:.3} ms, shrink {fit_ms:.3} ms, write {write_ms:.3} ms (each on 12 threads)");
        // the photos and posters' parts: reading the files, decoding them by kind (photo::load_for, for 32 px), fitting to 32 px
        let items = &pictures[..data.n()];
        let t = Instant::now();
        let bytes: usize = items.par_iter().map(|f| std::fs::read(f).map_or(0, |b| b.len())).sum();
        eprintln!("items' files read: {:.1}s for {} ({:.1} GB)", t.elapsed().as_secs_f32(), items.len(), bytes as f32 / 1e9);
        // decoding and fitting, by kind: nothing kept (a PNG is decoded whole, and thousands of them don't fit in memory); the time
        // each part took on each thread, added up, over the threads: the share of the wall time it had
        let threads = rayon::current_num_threads() as f64;
        for jpeg in [true, false] {
            let group: Vec<&PathBuf> = items.iter().filter(|f| f.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg")) == jpeg).collect();
            let (decode_ns, fit_ns, mp) = (std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0));
            let t = Instant::now();
            group.par_iter().for_each(|f| {
                let t = Instant::now();
                let Some(img) = photo::load_for(f, |_, _| TOP as u32) else { return };
                decode_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if let Ok((w, h)) = image::image_dimensions(f) {
                    mp.fetch_add((w as u64 * h as u64) / 1000, Ordering::Relaxed);
                }
                let t = Instant::now();
                photo::fit(&img, TOP as u32, true);
                fit_ns.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            });
            let share = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9 / threads;
            eprintln!(
                "{} ({}, {:.1} MP on average): {:.1}s in all; reading and decoding ~{:.1}s, fitting ~{:.1}s",
                if jpeg { "JPEGs" } else { "PNG and others" },
                group.len(),
                mp.load(Ordering::Relaxed) as f64 / 1000.0 / group.len().max(1) as f64,
                t.elapsed().as_secs_f32(),
                share(&decode_ns),
                share(&fit_ns)
            );
        }
        // one to look at
        let sample = posters.iter().find_map(|f| crate::build::small_frame(f).unwrap()).unwrap();
        std::fs::write(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/shots/frame-small.jpg"), sample).unwrap();
        eprintln!("frames made small: {:.2} ms each on 12 threads ({} posters), {:.1}s for the map's {} frames", per * 1e3, posters.len(), per * data.frames.len() as f32, data.frames.len());
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/bake-timing");
        std::fs::create_dir_all(&dir).unwrap();
        let t = Instant::now();
        bake(&dir, &pictures, &Progress::default()).unwrap();
        eprintln!("baked {} pictures in {:.1}s", pictures.len(), t.elapsed().as_secs_f32());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Baking a few photos writes the baked levels, in pages.rs's size.
    #[test]
    fn bakes_the_resident_levels() {
        let dir = std::env::temp_dir().join(format!("photo-map-bake-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("photos")).unwrap();
        let photos: Vec<PathBuf> = (0..5)
            .map(|i| {
                let file = dir.join(format!("photos/{i}.png"));
                image::RgbaImage::from_pixel(100 + i * 20, 80, image::Rgba([50 * i as u8, 120, 200, 255])).save(&file).unwrap();
                file
            })
            .collect();
        bake(&dir, &photos, &Progress::default()).unwrap();
        let len = std::fs::metadata(dir.join("pages.bin")).unwrap().len() as usize;
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
