//! Where each photo goes on the map, similar photos close together:
//! 1. PCA: centre the vectors (what's left is how a photo differs from the average one) and keep the 50 strongest
//!    directions of difference.
//! 2. Each photo's 10 nearest neighbours in those 50 numbers (exact, by cosine).
//! 3. UMAP: turn the neighbours into a weighted graph and lay it out in 2D so neighbours end up close and everything
//!    else apart. Follows umap-learn (n_neighbors 11 counting the photo itself, min_dist 0.05), but starts from the
//!    2 strongest PCA directions instead of a spectral layout.

use super::step;
use super::embed::DIM;
use rayon::prelude::*;
use std::collections::HashMap;

const COMPONENTS: usize = 50;
/// Components padded to a multiple of 8 so the dot products vectorize.
const PAD: usize = 56;
const NEIGHBOURS: usize = 11; // including the photo itself
/// The curve 1 / (1 + a d^2b) that umap-learn fits for min_dist 0.05, spread 1.
const A: f32 = 1.750_225;
const B: f32 = 0.842_055_4;
const NEGATIVE_RATE: f32 = 5.0;

/// Positions in 0 ... world on both axes.
pub fn lay_out(vectors: &[f32], n: usize, world: f32) -> Vec<[f32; 2]> {
    let pcs = pca(vectors, n);
    let k = NEIGHBOURS.min(n);
    let (idx, dist) = nearest(&pcs, n, k);
    step(format!("{} nearest neighbours of each photo found", k - 1));
    let edges = graph(&idx, &dist, n, k);
    let mut xy = umap(&pcs, &edges, n);
    for axis in 0..2 {
        let (lo, hi) = xy.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[axis]), hi.max(p[axis])));
        for p in &mut xy {
            p[axis] = (p[axis] - lo) / (hi - lo).max(1e-9) * world;
        }
    }
    xy
}

/// Rows of PAD numbers: the photo's coordinates along the strongest directions, scaled to length 1.
fn pca(vectors: &[f32], n: usize) -> Vec<f32> {
    let d = DIM;
    let mut mean = vec![0f64; d];
    for row in vectors.chunks_exact(d) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    let mean: Vec<f32> = mean.iter().map(|m| (m / n as f64) as f32).collect();
    let mut centred = vectors.to_vec();
    centred.par_chunks_exact_mut(d).for_each(|row| {
        row.iter_mut().zip(&mean).for_each(|(v, m)| *v -= m);
        normalize(row);
    });

    // covariance C = Xᵀ X (d x d), summed over blocks of rows in parallel
    let cov = centred
        .par_chunks(d * 256)
        .map(|block| {
            let mut c = vec![0f32; d * d];
            for row in block.chunks_exact(d) {
                for (a, &ra) in row.iter().enumerate() {
                    c[a * d..(a + 1) * d].iter_mut().zip(row).for_each(|(c, &rb)| *c += ra * rb);
                }
            }
            c
        })
        .reduce(|| vec![0f32; d * d], |mut x, y| {
            x.iter_mut().zip(&y).for_each(|(x, y)| *x += y);
            x
        });
    let total: f32 = (0..d).map(|a| cov[a * d + a]).sum();

    // the strongest directions: orthogonal iteration (repeatedly multiply by C and re-orthonormalize, in order)
    let comps = COMPONENTS.min(d).min(n);
    let mut rng = Rng(7);
    let mut q: Vec<f32> = (0..comps * d).map(|_| rng.normal()).collect();
    orthonormalize(&mut q, d);
    for _ in 0..150 {
        q = q.par_chunks_exact(d).flat_map_iter(|col| multiply(&cov, col, d)).collect();
        orthonormalize(&mut q, d);
    }
    let kept: f32 = q.chunks_exact(d).map(|col| dot(col, &multiply(&cov, col, d))).sum();
    step(format!("PCA: {comps} directions keep {:.0}% of the variation", kept / total * 100.0));

    let mut pcs = vec![0f32; n * PAD];
    pcs.par_chunks_exact_mut(PAD).zip(centred.par_chunks_exact(d)).for_each(|(out, row)| {
        for (o, col) in out.iter_mut().zip(q.chunks_exact(d)) {
            *o = dot(row, col);
        }
        normalize(out);
    });
    pcs
}

fn multiply(m: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    m.chunks_exact(d).map(|row| dot(row, v)).collect()
}

/// Gram-Schmidt on the columns (each d long), in order, twice for accuracy.
fn orthonormalize(q: &mut [f32], d: usize) {
    for c in 0..q.len() / d {
        for _ in 0..2 {
            for p in 0..c {
                let (before, rest) = q.split_at_mut(c * d);
                let (prev, col) = (&before[p * d..(p + 1) * d], &mut rest[..d]);
                let proj = dot(prev, col);
                col.iter_mut().zip(prev).for_each(|(x, y)| *x -= proj * y);
            }
        }
        normalize(&mut q[c * d..(c + 1) * d]);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// PAD-long dot product in 8 lanes, so it vectorizes.
fn dot_pad(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0f32; 8];
    for (x, y) in a[..PAD].chunks_exact(8).zip(b[..PAD].chunks_exact(8)) {
        for l in 0..8 {
            lanes[l] += x[l] * y[l];
        }
    }
    lanes.iter().sum()
}

fn normalize(v: &mut [f32]) {
    let len = dot(v, v).sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= len);
}

/// Each photo's k nearest (itself first), by cosine distance: indices and distances, n x k, closest first.
/// Exact: compares every pair, a block of rows at a time so the data is read from memory once per block.
fn nearest(pcs: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    const BLOCK: usize = 64;
    let mut idx = vec![0u32; n * k];
    let mut dist = vec![0f32; n * k];
    pcs.par_chunks(BLOCK * PAD)
        .zip(idx.par_chunks_mut(BLOCK * k).zip(dist.par_chunks_mut(BLOCK * k)))
        .for_each(|(rows, (idx, dist))| {
            let m = rows.len() / PAD;
            let mut best = vec![(f32::MIN, 0u32); m * k]; // per row: most similar first
            for (j, other) in pcs.chunks_exact(PAD).enumerate() {
                for r in 0..m {
                    let s = dot_pad(&rows[r * PAD..], other);
                    let list = &mut best[r * k..(r + 1) * k];
                    if s > list[k - 1].0 {
                        let at = list.iter().position(|e| s > e.0).unwrap();
                        list.copy_within(at..k - 1, at + 1);
                        list[at] = (s, j as u32);
                    }
                }
            }
            for (t, (s, j)) in best.into_iter().enumerate() {
                idx[t] = j;
                dist[t] = (1.0 - s).max(0.0);
            }
        });
    (idx, dist)
}

/// UMAP's fuzzy neighbour graph: per photo, weights that fall off past its nearest neighbour's distance, scaled so
/// they sum to log2(k); then made symmetric (a or b: a + b - ab). Edges (from, to, weight) in both directions.
fn graph(idx: &[u32], dist: &[f32], n: usize, k: usize) -> Vec<(u32, u32, f32)> {
    let target = (k as f32).log2();
    let mean_all = dist.iter().sum::<f32>() / dist.len() as f32;
    let mut pairs: HashMap<(u32, u32), [f32; 2]> = HashMap::new();
    for i in 0..n {
        let (ids, ds) = (&idx[i * k..(i + 1) * k], &dist[i * k..(i + 1) * k]);
        let rho = ds.iter().copied().find(|&d| d > 0.0).unwrap_or(0.0);
        let (mut lo, mut hi, mut sigma) = (0f32, f32::INFINITY, 1f32);
        for _ in 0..64 {
            let sum: f32 = ds[1..].iter().map(|&d| if d - rho > 0.0 { (-(d - rho) / sigma).exp() } else { 1.0 }).sum();
            if (sum - target).abs() < 1e-5 {
                break;
            }
            if sum > target {
                hi = sigma;
                sigma = (lo + hi) / 2.0;
            } else {
                lo = sigma;
                sigma = if hi.is_infinite() { sigma * 2.0 } else { (lo + hi) / 2.0 };
            }
        }
        let mean = if rho > 0.0 { ds.iter().sum::<f32>() / k as f32 } else { mean_all };
        sigma = sigma.max(1e-3 * mean);
        for (&j, &d) in ids.iter().zip(ds) {
            if j as usize == i {
                continue;
            }
            let w = if d - rho <= 0.0 { 1.0 } else { (-(d - rho) / sigma).exp() };
            let (a, b, side) = if (i as u32) < j { (i as u32, j, 0) } else { (j, i as u32, 1) };
            pairs.entry((a, b)).or_insert([0.0; 2])[side] = w;
        }
    }
    let mut edges: Vec<(u32, u32, f32)> = pairs
        .into_iter()
        .flat_map(|((a, b), [x, y])| {
            let w = x + y - x * y;
            [(a, b, w), (b, a, w)]
        })
        .collect();
    edges.sort_unstable_by_key(|&(a, b, _)| (a, b)); // a fixed order: the same map every run
    edges
}

/// UMAP's layout: pull edges together (each sampled in proportion to its weight), push random pairs apart.
fn umap(pcs: &[f32], edges: &[(u32, u32, f32)], n: usize) -> Vec<[f32; 2]> {
    let epochs = if n > 10_000 { 200 } else { 500 };
    let max_w = edges.iter().map(|e| e.2).fold(0.0, f32::max);
    let edges: Vec<_> = edges.iter().filter(|e| e.2 >= max_w / epochs as f32).collect();
    let every: Vec<f32> = edges.iter().map(|e| max_w / e.2).collect(); // sampled every this many epochs
    let every_negative: Vec<f32> = every.iter().map(|e| e / NEGATIVE_RATE).collect();
    let mut next = every.clone();
    let mut next_negative = every_negative.clone();

    // start: the 2 strongest PCA directions, scaled to 0 ... 10 with a little noise (as umap-learn)
    let mut rng = Rng(42);
    let mut y: Vec<[f32; 2]> = pcs.chunks_exact(PAD).map(|p| [p[0], p[1]]).collect();
    let max_abs = y.iter().flatten().fold(0f32, |m, v| m.max(v.abs())).max(1e-9);
    for p in &mut y {
        for v in p.iter_mut() {
            *v = *v / max_abs * 10.0 + rng.normal() * 1e-4;
        }
    }
    for axis in 0..2 {
        let (lo, hi) = y.iter().fold((f32::MAX, f32::MIN), |(lo, hi), p| (lo.min(p[axis]), hi.max(p[axis])));
        y.iter_mut().for_each(|p| p[axis] = 10.0 * (p[axis] - lo) / (hi - lo).max(1e-9));
    }

    let clip = |g: f32| g.clamp(-4.0, 4.0);
    for epoch in 0..epochs {
        let alpha = 1.0 - epoch as f32 / epochs as f32;
        let now = epoch as f32;
        for (e, &&(from, to, _)) in edges.iter().enumerate() {
            if next[e] > now {
                continue;
            }
            let (i, j) = (from as usize, to as usize);
            let diff = [y[i][0] - y[j][0], y[i][1] - y[j][1]];
            let d2 = diff[0] * diff[0] + diff[1] * diff[1];
            let pull = if d2 > 0.0 { -2.0 * A * B * d2.powf(B - 1.0) / (A * d2.powf(B) + 1.0) } else { 0.0 };
            for axis in 0..2 {
                let g = clip(pull * diff[axis]) * alpha;
                y[i][axis] += g;
                y[j][axis] -= g;
            }
            next[e] += every[e];

            let negatives = ((now - next_negative[e]) / every_negative[e]).max(0.0) as usize;
            for _ in 0..negatives {
                let other = rng.below(n);
                if other == i {
                    continue;
                }
                let diff = [y[i][0] - y[other][0], y[i][1] - y[other][1]];
                let d2 = diff[0] * diff[0] + diff[1] * diff[1];
                let push = if d2 > 0.0 { 2.0 * B / ((0.001 + d2) * (A * d2.powf(B) + 1.0)) } else { 0.0 };
                if push > 0.0 {
                    for axis in 0..2 {
                        y[i][axis] += clip(push * diff[axis]) * alpha;
                    }
                }
            }
            next_negative[e] += negatives as f32 * every_negative[e];
        }
        if (epoch + 1) % 50 == 0 {
            step(format!("UMAP epoch {}/{epochs}", epoch + 1));
        }
    }
    y
}

/// Small seeded random numbers (SplitMix64): the same map every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn uniform(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        (-2.0 * self.uniform().ln()).sqrt() * (std::f32::consts::TAU * self.uniform()).cos()
    }
}
