//! The photos split into cells of similar photos (k-means by cosine, about 2 cells per square root of the number of photos),
//! to find each photo's nearest photos among the photos of its nearest cells only, not among all: a photo is compared with
//! a few thousand others, and the time grows with n^1.5, not n². A photo whose near ones are in a cell it doesn't look in
//! misses them (and takes the next nearest instead): the more cells looked in, the fewer missed.

use half::f16;
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Where the photos' vectors are read from, a row at a time: an f32 matrix, or a map's f16 vectors straight from its file (half the
/// memory of f32, and the same numbers once widened: f16 to f32 is exact). Nothing makes an f32 copy of all of them.
pub trait Rows: Sync {
    /// How a number is kept (f32, or f16 as its bits).
    type Elem: Copy + Default + Send + Sync;
    fn count(&self) -> usize;
    fn dim(&self) -> usize;
    /// Row `i` as it is kept.
    fn raw(&self, i: usize) -> &[Self::Elem];
    /// Kept numbers as f32, into `out` (as long).
    fn widen(raw: &[Self::Elem], out: &mut [f32]);

    /// Row `i` as f32, into `out` (`dim` long).
    fn row_into(&self, i: usize, out: &mut [f32]) {
        Self::widen(self.raw(i), out);
    }
}

impl Rows for Array2<f32> {
    type Elem = f32;

    fn count(&self) -> usize {
        self.nrows()
    }

    fn dim(&self) -> usize {
        self.ncols()
    }

    fn raw(&self, i: usize) -> &[f32] {
        let dim = self.ncols();
        &self.as_slice().expect("a matrix row by row")[i * dim..(i + 1) * dim]
    }

    fn widen(raw: &[f32], out: &mut [f32]) {
        out.copy_from_slice(raw);
    }
}

/// A map's items' vectors as f16, read from look.bin where it is (mapped, not copied: the system keeps what it reads of it in its
/// file cache, which it can drop when memory is short, and reads again).
pub struct HalfRows {
    map: memmap2::Mmap,
    n: usize,
    dim: usize,
}

impl HalfRows {
    /// The first `n` rows (the items'; the videos' frames' come after) of `dim` numbers of look.bin; an error if it is shorter.
    pub fn open(path: &std::path::Path, n: usize, dim: usize) -> Result<HalfRows, String> {
        const { assert!(cfg!(target_endian = "little"), "look.bin is little-endian") };
        let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // the file is the build's own, and nothing writes it while it is read
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
        if map.len() < n * dim * 2 || map.len() % (dim * 2) != 0 {
            return Err(format!("{} is not the size of the photos", path.display()));
        }
        Ok(HalfRows { map, n, dim })
    }
}

impl Rows for HalfRows {
    type Elem = u16;

    fn count(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn raw(&self, i: usize) -> &[u16] {
        // a mapping starts on a page, so it is aligned for u16
        &bytemuck::cast_slice(&self.map[..])[i * self.dim..(i + 1) * self.dim]
    }

    fn widen(raw: &[u16], out: &mut [f32]) {
        let raw: &[f16] = raw.reinterpret_cast();
        raw.convert_to_f32_slice(out);
    }
}

/// The rows `ids` of `v`, one per row, as f32.
pub fn gather<R: Rows + ?Sized>(v: &R, ids: impl ExactSizeIterator<Item = usize>) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((ids.len(), v.dim()));
    for (mut row, i) in out.axis_iter_mut(Axis(0)).zip(ids) {
        v.row_into(i, row.as_slice_mut().expect("a new matrix is row by row"));
    }
    out
}

/// Photos of one cell looked up at once (a cell has about 200: mostly one block, so its nearest cells are read once).
const BLOCK: usize = 256;
/// Cells per square root of the number of photos.
const CELLS: f32 = 2.0;
/// k-means rounds, and the photos it is worked out on (per cell).
const ROUNDS: usize = 10;
const SAMPLE: usize = 64;

/// The photos split into cells of similar photos (k-means by cosine). `R`: where the vectors came from.
pub struct Cells<R: Rows> {
    /// Each cell's centre (unit length), one per row.
    centres: Array2<f32>,
    members: Vec<Vec<u32>>,
    /// Each photo's cell, and its row in `sorted`.
    of: Vec<u32>,
    row: Vec<u32>,
    /// The vectors, as they were kept (an f16 map's as f16), cell by cell: a cell's rows are together, from `start[cell]`.
    sorted: Vec<R::Elem>,
    start: Vec<usize>,
    dim: usize,
}

impl<R: Rows> Cells<R> {
    /// The cells of the vectors `v` (which are copied, in the order of the cells: `v` isn't needed after).
    pub fn new(v: &R, progress: &(dyn Fn(&str, usize, usize) + Sync), paused: &(dyn Fn() -> bool + Sync)) -> Option<Cells<R>> {
        let n = v.count();
        let k = ((n as f32).sqrt() * CELLS).round().clamp(1.0, n as f32) as usize;
        // worked out on photos spread through the list, started from k of them
        let every = |count: usize| -> Vec<usize> { (0..count).map(|i| i * n / count).collect() };
        let sample = gather(v, every((k * SAMPLE).min(n)).into_iter());
        let mut centres = gather(v, every(k).into_iter());
        for round in 0..ROUNDS {
            if paused() {
                return None;
            }
            progress("splitting the photos into cells", round, ROUNDS);
            let of = nearest_centre(&sample, &centres);
            let mut sums = Array2::<f32>::zeros(centres.dim());
            for (row, &c) in sample.axis_iter(Axis(0)).zip(&of) {
                sums.row_mut(c as usize).scaled_add(1.0, &row);
            }
            for (mut centre, sum) in centres.axis_iter_mut(Axis(0)).zip(sums.axis_iter(Axis(0))) {
                let norm = sum.dot(&sum).sqrt();
                if norm > 1e-9 {
                    centre.assign(&sum.mapv(|x| x / norm)); // an empty cell keeps its centre
                }
            }
        }
        let of = nearest_centre(v, &centres);
        let mut members = vec![Vec::new(); k];
        for (i, &c) in of.iter().enumerate() {
            members[c as usize].push(i as u32);
        }
        let dim = v.dim();
        let (mut sorted, mut start, mut row) = (Vec::with_capacity(n * dim), Vec::with_capacity(k + 1), vec![0u32; n]);
        for m in &members {
            start.push(sorted.len() / dim.max(1));
            for &i in m {
                row[i as usize] = (sorted.len() / dim.max(1)) as u32;
                sorted.extend_from_slice(v.raw(i as usize));
            }
        }
        start.push(n);
        Some(Cells { centres, members, of, row, sorted, start, dim })
    }

    /// Rows `rows` of `sorted`, as f32, one per row.
    fn widened(&self, rows: impl ExactSizeIterator<Item = usize>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((rows.len(), self.dim));
        for (mut o, r) in out.axis_iter_mut(Axis(0)).zip(rows) {
            R::widen(&self.sorted[r * self.dim..(r + 1) * self.dim], o.as_slice_mut().expect("a new matrix is row by row"));
        }
        out
    }

    /// For each of `rows`: its `nearest` photos outside its group, nearest first, as (1 - cosine, photo), among the photos of
    /// its `probe` nearest cells that hold any outside its group.
    #[allow(clippy::too_many_arguments)]
    pub fn nearest_outside(
        &self,
        probe: usize,
        nearest: usize,
        rows: &[usize],
        group: &[u32],
        what: &str,
        progress: &(dyn Fn(&str, usize, usize) + Sync),
        paused: &(dyn Fn() -> bool + Sync),
    ) -> Option<Vec<Vec<(f32, u32)>>> {
        // a cell whose photos are all in one group has nothing for a photo of that group
        let only: Vec<Option<u32>> = self
            .members
            .iter()
            .map(|m| m.first().filter(|&&a| m.iter().all(|&b| group[b as usize] == group[a as usize])).map(|&a| group[a as usize]))
            .collect();
        // the rows in blocks of one cell
        let mut by_cell = vec![Vec::new(); self.members.len()];
        for (at, &i) in rows.iter().enumerate() {
            by_cell[self.of[i] as usize].push(at);
        }
        let blocks: Vec<&[usize]> = by_cell.iter().flat_map(|b| b.chunks(BLOCK)).collect();
        let (done, stopped) = (AtomicUsize::new(0), AtomicBool::new(false));
        let found: Vec<Vec<(usize, Vec<(f32, u32)>)>> = blocks
            .par_iter()
            .map(|block| {
                if stopped.load(Ordering::Relaxed) || paused() {
                    stopped.store(true, Ordering::Relaxed);
                    return Vec::new();
                }
                let ids: Vec<usize> = block.iter().map(|&at| rows[at]).collect();
                let mine = self.widened(ids.iter().map(|&i| self.row[i] as usize));
                // each photo's nearest cells that hold photos outside its group
                let to_centres = mine.dot(&self.centres.t());
                let mut asking: Vec<Vec<usize>> = vec![Vec::new(); self.members.len()];
                for (r, (&i, sim)) in ids.iter().zip(to_centres.axis_iter(Axis(0))).enumerate() {
                    let mut open: Vec<usize> = (0..self.members.len()).filter(|&c| only[c] != Some(group[i])).collect();
                    let p = probe.min(open.len());
                    if p > 0 && p < open.len() {
                        open.select_nth_unstable_by(p - 1, |&a, &b| sim[b].total_cmp(&sim[a]));
                    }
                    for &c in &open[..p] {
                        asking[c].push(r);
                    }
                }
                // each cell compared with the photos that asked for it; every photo's candidates (thousands) in one buffer for the
                // block, each with the room of the cells it asked for: not a list per photo grown as it goes (in many threads at once,
                // the memory such lists leave behind is kept by the allocator, gigabytes for a big map)
                let mut start = vec![0usize; ids.len() + 1];
                for (c, who) in asking.iter().enumerate() {
                    for &r in who {
                        start[r + 1] += self.members[c].len();
                    }
                }
                for r in 0..ids.len() {
                    start[r + 1] += start[r];
                }
                let mut far = vec![(0f32, 0u32); start[ids.len()]];
                let mut len = vec![0usize; ids.len()];
                for (c, who) in asking.iter().enumerate().filter(|(_, w)| !w.is_empty()) {
                    let theirs = self.widened(self.start[c]..self.start[c + 1]);
                    let sims = mine.select(Axis(0), who).dot(&theirs.t());
                    for (&r, sim) in who.iter().zip(sims.axis_iter(Axis(0))) {
                        let i = ids[r];
                        for (&j, &s) in self.members[c].iter().zip(sim.iter()).filter(|&(&j, _)| group[j as usize] != group[i]) {
                            far[start[r] + len[r]] = (1.0 - s, j);
                            len[r] += 1;
                        }
                    }
                }
                let mut out = Vec::with_capacity(ids.len());
                for (r, &at) in block.iter().enumerate() {
                    let far = &mut far[start[r]..start[r] + len[r]];
                    let k = nearest.min(far.len());
                    if k > 0 && k < far.len() {
                        far.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
                    }
                    let mut list = far[..k].to_vec();
                    list.sort_by(|a, b| a.0.total_cmp(&b.0));
                    out.push((at, list));
                }
                progress(what, done.fetch_add(1, Ordering::Relaxed) + 1, blocks.len());
                out
            })
            .collect();
        if stopped.load(Ordering::Relaxed) {
            return None;
        }
        let mut lists = vec![Vec::new(); rows.len()];
        for (at, list) in found.into_iter().flatten() {
            lists[at] = list;
        }
        Some(lists)
    }
}

/// Each row's nearest centre (by cosine), in blocks so that the cosines held at once stay small.
fn nearest_centre<R: Rows>(v: &R, centres: &Array2<f32>) -> Vec<u32> {
    let n = v.count();
    let blocks: Vec<Vec<u32>> = (0..n.div_ceil(1024))
        .into_par_iter()
        .map(|b| {
            gather(v, b * 1024..((b + 1) * 1024).min(n))
                .dot(&centres.t())
                .axis_iter(Axis(0))
                .map(|sim| (0..sim.len()).max_by(|&a, &b| sim[a].total_cmp(&sim[b]).then(b.cmp(&a))).unwrap_or(0) as u32)
                .collect()
        })
        .collect();
    blocks.concat()
}
