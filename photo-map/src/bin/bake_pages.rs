//! One-off: the always-loaded levels (1 px ... 64 px) as BC7 pages for the viewer.
//!     cargo run --bin bake_pages   ->  data/wikivoyage/photo_map/pages_bc7.bin
//! Every level L holds each photo in a slot of 2^L px (photo in the top-left corner, real shape). Slots sit on
//! PAGE x PAGE pages in a grid of PAGE / 2^L per row; slot = photo id. Pages are written level 0 first, each
//! as BC7 blocks (16 bytes per 4 x 4 pixels). Source: atlas.bin from prep_photo_map.py (64 px tiles + their mips).

use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

// intel_tex_2's precompiled ISPC kernels need the C++ runtime (__gxx_personality_v0)
#[link(name = "stdc++")]
unsafe extern "C" {}

const PAGE: usize = 4096;
const RESIDENT_LEVELS: usize = 7; // 1, 2, 4, ... 64 px
/// Rows compressed per task: a multiple of 4 (a BC7 block row).
const STRIP: usize = 64;

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let dir = root.join("data/wikivoyage/photo_map");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let get = |k: &str| meta[k].as_u64().unwrap() as usize;
    let (n, side, mips, tile) = (get("n"), get("side"), get("mips"), get("tile"));
    assert_eq!(tile, 1 << (RESIDENT_LEVELS - 1));
    let raw = unsafe { Mmap::map(&File::open(dir.join("atlas.bin")).unwrap()).unwrap() };
    let per_atlas_row = side / tile;
    let per_atlas_layer = per_atlas_row * per_atlas_row;
    // atlas.bin: per atlas layer, mip 0 (side x side) then each half-size mip
    let level_bytes = |m: usize| (side >> m) * (side >> m) * 4;
    let layer_bytes: usize = (0..mips).map(level_bytes).sum();
    let atlas_level = |layer: usize, m: usize| {
        let start = layer * layer_bytes + (0..m).map(level_bytes).sum::<usize>();
        &raw[start..start + level_bytes(m)]
    };

    let mut out = std::io::BufWriter::new(File::create(dir.join("pages_bc7.bin")).unwrap());
    let settings = intel_tex_2::bc7::opaque_basic_settings();
    let started = Instant::now();
    for level in 0..RESIDENT_LEVELS {
        let s = 1 << level; // slot size
        let m = RESIDENT_LEVELS - 1 - level; // the atlas mip that has tiles of this size
        let per_row = PAGE / s;
        let per_page = per_row * per_row;
        let pages = n.div_ceil(per_page);
        for page in 0..pages {
            // gather this page's photos from the atlas mip, then compress it
            let mut px = vec![0u8; PAGE * PAGE * 4];
            let first = page * per_page;
            for id in first..(first + per_page).min(n) {
                let (layer, t) = (id / per_atlas_layer, id % per_atlas_layer);
                let src = atlas_level(layer, m);
                let src_side = side >> m;
                let (sx, sy) = ((t % per_atlas_row) * s, (t / per_atlas_row) * s);
                let local = id - first;
                let (dx, dy) = ((local % per_row) * s, (local / per_row) * s);
                for row in 0..s {
                    let a = ((sy + row) * src_side + sx) * 4;
                    let b = ((dy + row) * PAGE + dx) * 4;
                    px[b..b + s * 4].copy_from_slice(&src[a..a + s * 4]);
                }
            }
            let mut blocks = vec![0u8; PAGE * PAGE];
            blocks.par_chunks_mut(PAGE * STRIP).zip(px.par_chunks(PAGE * 4 * STRIP)).for_each(|(dst, src)| {
                let surface = intel_tex_2::RgbaSurface { data: src, width: PAGE as u32, height: STRIP as u32, stride: (PAGE * 4) as u32 };
                intel_tex_2::bc7::compress_blocks_into(&settings, &surface, dst);
            });
            out.write_all(&blocks).unwrap();
        }
        eprintln!("level {level} ({s} px): {pages} page(s) done ({:.0}s)", started.elapsed().as_secs_f32());
    }
    out.flush().unwrap();
    eprintln!("BAKE_DONE in {:.0}s", started.elapsed().as_secs_f32());
}
