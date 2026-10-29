//! The photos split into cells of similar photos (k-means by cosine, about 2 cells per square root of the number of photos),
//! to find each photo's nearest photos among the photos of its nearest cells only, not among all: a photo is compared with
//! a few thousand others, and the time grows with n^1.5, not n². A photo whose near ones are in a cell it doesn't look in
//! misses them (and takes the next nearest instead): the more cells looked in, the fewer missed.

use ndarray::{Array2, Axis};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Photos of one cell looked up at once.
const BLOCK: usize = 128;
/// Cells per square root of the number of photos.
const CELLS: f32 = 2.0;
/// k-means rounds, and the photos it is worked out on (per cell).
const ROUNDS: usize = 10;
const SAMPLE: usize = 64;

/// The photos split into cells of similar photos (k-means by cosine).
pub struct Cells {
    /// Each cell's centre (unit length), one per row.
    centres: Array2<f32>,
    members: Vec<Vec<u32>>,
    /// Each cell's photos' vectors.
    vectors: Vec<Array2<f32>>,
    /// Each photo's cell.
    of: Vec<u32>,
}

impl Cells {
    pub fn new(v: &Array2<f32>, progress: &(dyn Fn(&str, usize, usize) + Sync), paused: &(dyn Fn() -> bool + Sync)) -> Option<Cells> {
        let n = v.nrows();
        let k = ((n as f32).sqrt() * CELLS).round().clamp(1.0, n as f32) as usize;
        // worked out on photos spread through the list, started from k of them
        let every = |count: usize| -> Vec<usize> { (0..count).map(|i| i * n / count).collect() };
        let sample = v.select(Axis(0), &every((k * SAMPLE).min(n)));
        let mut centres = v.select(Axis(0), &every(k));
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
        let vectors = members.par_iter().map(|m| v.select(Axis(0), &m.iter().map(|&j| j as usize).collect::<Vec<_>>())).collect();
        Some(Cells { centres, members, vectors, of })
    }

    /// For each of `rows`: its `nearest` photos outside its group, nearest first, as (1 - cosine, photo), among the photos of
    /// its `probe` nearest cells that hold any outside its group.
    #[allow(clippy::too_many_arguments)]
    pub fn nearest_outside(
        &self,
        v: &Array2<f32>,
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
                let mine = v.select(Axis(0), &ids);
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
                // each cell compared with the photos that asked for it
                let mut far: Vec<Vec<(f32, u32)>> = vec![Vec::new(); ids.len()];
                for (c, who) in asking.iter().enumerate().filter(|(_, w)| !w.is_empty()) {
                    let sims = mine.select(Axis(0), who).dot(&self.vectors[c].t());
                    for (&r, sim) in who.iter().zip(sims.axis_iter(Axis(0))) {
                        let i = ids[r];
                        far[r].extend(self.members[c].iter().zip(sim.iter()).filter(|&(&j, _)| group[j as usize] != group[i]).map(|(&j, &s)| (1.0 - s, j)));
                    }
                }
                let mut out = Vec::with_capacity(ids.len());
                for (&at, mut far) in block.iter().zip(far) {
                    let k = nearest.min(far.len());
                    if k > 0 && k < far.len() {
                        far.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
                    }
                    far.truncate(k);
                    far.sort_by(|a, b| a.0.total_cmp(&b.0));
                    out.push((at, far));
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
fn nearest_centre(v: &Array2<f32>, centres: &Array2<f32>) -> Vec<u32> {
    let blocks: Vec<Vec<u32>> = v
        .axis_chunks_iter(Axis(0), 1024)
        .into_par_iter()
        .map(|rows| {
            rows.dot(&centres.t())
                .axis_iter(Axis(0))
                .map(|sim| (0..sim.len()).max_by(|&a, &b| sim[a].total_cmp(&sim[b]).then(b.cmp(&a))).unwrap_or(0) as u32)
                .collect()
        })
        .collect();
    blocks.concat()
}
