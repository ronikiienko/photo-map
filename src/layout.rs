//! Laying out a map: where each photo goes so that similar photos are close. PCA to 50 dimensions, each photo's 10 nearest
//! neighbours (exact, by cosine), then UMAP down to 2 (umap-rs), scaled to 0 ... WORLD on both axes - what src/photomap.py did.

use crate::data::WORLD;
use ndarray::{Array2, Axis};
use rayon::prelude::*;
use umap_rs::{GraphParams, LearnedManifold, Umap, UmapConfig};

const COMPONENTS: usize = 50;
const NEIGHBOURS: usize = 11; // a photo itself and its 10 nearest
const MIN_DIST: f32 = 0.05;

/// A small deterministic random generator (xorshift): the layout is the same every time for the same photos.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    }
}

fn normalise_rows(m: &mut Array2<f32>) {
    m.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            row.mapv_inplace(|x| x / norm);
        }
    });
}

/// `vectors`: n x dim (unit length; a zero row is a photo that couldn't be read). `progress(text, done, total)` is told how far each
/// part is; `paused()` is asked often: true stops it (None). The x, y of every photo, 0 ... WORLD.
pub fn layout(vectors: Array2<f32>, progress: &(dyn Fn(&str, usize, usize) + Sync), paused: &(dyn Fn() -> bool + Sync)) -> Result<Option<Vec<f32>>, String> {
    let n = vectors.nrows();
    if n < 3 {
        // too few photos for a manifold: along a line
        return Ok(Some((0..n).flat_map(|i| [WORLD * (i as f32 + 0.5) / n as f32, WORLD / 2.0]).collect()));
    }
    let dim = vectors.ncols();

    // 1. centre (a photo it couldn't read counts as the average), unit length
    progress("preparing the vectors", 0, 1);
    let readable: Vec<bool> = vectors.axis_iter(Axis(0)).map(|r| r.iter().map(|x| x * x).sum::<f32>() > 0.25).collect();
    let count = readable.iter().filter(|&&r| r).count().max(1);
    let mut mean = vec![0f32; dim];
    for (row, _) in vectors.axis_iter(Axis(0)).zip(&readable).filter(|(_, ok)| **ok) {
        for (m, x) in mean.iter_mut().zip(row.iter()) {
            *m += x / count as f32;
        }
    }
    let mut x = vectors;
    for (mut row, &ok) in x.axis_iter_mut(Axis(0)).zip(&readable) {
        for (v, m) in row.iter_mut().zip(&mean) {
            *v = if ok { *v - m } else { 0.0 };
        }
    }
    normalise_rows(&mut x);

    // 2. the top directions of the photos' spread (subspace iteration on the covariance), the photos in them
    let p = COMPONENTS.min(dim - 1).min(n - 1);
    let cov = x
        .axis_chunks_iter(Axis(0), 4096)
        .into_par_iter()
        .map(|chunk| chunk.t().dot(&chunk))
        .reduce(|| Array2::zeros((dim, dim)), |a, b| a + b);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut q = Array2::from_shape_fn((dim, p), |_| rng.next());
    for round in 0..40 {
        if paused() {
            return Ok(None);
        }
        progress("finding the main directions of the photos", round, 40);
        q = cov.dot(&q);
        for j in 0..p {
            for _ in 0..2 {
                for i in 0..j {
                    let d = q.column(i).dot(&q.column(j));
                    let qi = q.column(i).to_owned();
                    q.column_mut(j).scaled_add(-d, &qi);
                }
            }
            let norm = q.column(j).dot(&q.column(j)).sqrt().max(1e-12);
            q.column_mut(j).mapv_inplace(|v| v / norm);
        }
    }
    let mut pcs = x.dot(&q);
    let init = pcs.slice(ndarray::s![.., ..2]).to_owned();
    normalise_rows(&mut pcs);
    drop(x);

    // 3. each photo's nearest neighbours by cosine
    let k = NEIGHBOURS.min(n - 1).max(2);
    let Some((indices, dists)) = neighbours(&pcs, k, progress, paused) else { return Ok(None) };

    // 4. UMAP
    progress("arranging the photos", 0, 1);
    let epochs = if n > 10_000 { 200 } else { 500 };
    let mut config = UmapConfig::default();
    config.graph = GraphParams { n_neighbors: k, ..Default::default() };
    config.manifold.min_dist = MIN_DIST;
    let umap = Umap::new(config);
    let manifold = umap.learn_manifold(pcs.view(), indices.view(), dists.view());
    let mut start = init;
    for (i, v) in start.iter_mut().enumerate() {
        *v += 1e-4 * Rng(i as u64 + 1).next();
    }
    let Some(embedding) = optimise(&manifold, start, epochs, progress, paused) else { return Ok(None) };
    let xy = embedding.view();

    // 5. scaled to 0 ... WORLD on both axes
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for row in xy.axis_iter(Axis(0)) {
        for a in 0..2 {
            lo[a] = lo[a].min(row[a]);
            hi[a] = hi[a].max(row[a]);
        }
    }
    let scaled = xy.axis_iter(Axis(0)).flat_map(|r| (0..2).map(move |a| (r[a] - lo[a]) / (hi[a] - lo[a]).max(1e-9) * WORLD).collect::<Vec<_>>());
    Ok(Some(scaled.collect()))
}

/// UMAP's layout step: the photos move in 2D, pulled towards the photos they are linked to and pushed away from random others,
/// the step shrinking over `epochs` (umap-learn's optimize_layout_euclidean). `start` is scaled to 0 ... 10 first.
fn optimise(
    manifold: &LearnedManifold,
    mut emb: Array2<f32>,
    epochs: usize,
    progress: &(dyn Fn(&str, usize, usize) + Sync),
    paused: &(dyn Fn() -> bool + Sync),
) -> Option<Array2<f32>> {
    let n = emb.nrows();
    let (a, b) = manifold.curve_params();
    let (gamma, negatives) = (1.0f32, 5usize);
    for d in 0..2 {
        let (lo, hi) = emb.column(d).iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        let scale = if hi > lo { 10.0 / (hi - lo) } else { 0.0 };
        emb.column_mut(d).mapv_inplace(|v| (v - lo) * scale);
    }
    // the edges of the graph worth keeping, and how often each is used: stronger links more often
    let graph = manifold.graph();
    let max_weight = graph.data().iter().copied().fold(0.0f32, f32::max);
    let keep = max_weight / epochs as f32;
    let (mut head, mut tail, mut per_sample) = (Vec::new(), Vec::new(), Vec::new());
    for (row, vector) in graph.outer_iterator().enumerate() {
        for (col, &w) in vector.iter() {
            if w >= keep && w > 0.0 {
                head.push(row);
                tail.push(col);
                per_sample.push(max_weight / w);
            }
        }
    }
    let mut next: Vec<f32> = per_sample.clone();
    let per_negative: Vec<f32> = per_sample.iter().map(|e| e / negatives as f32).collect();
    let mut next_negative = per_negative.clone();
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let clip = |v: f32| v.clamp(-4.0, 4.0);
    let data = emb.as_slice_mut().expect("contiguous");
    for epoch in 0..epochs {
        if paused() {
            return None;
        }
        let alpha = 1.0 - epoch as f32 / epochs as f32;
        for e in 0..head.len() {
            if next[e] > epoch as f32 {
                continue;
            }
            let (j, k) = (head[e] * 2, tail[e] * 2);
            let (dx, dy) = (data[j] - data[k], data[j + 1] - data[k + 1]);
            let d2 = dx * dx + dy * dy;
            let attract = if d2 > 0.0 { -2.0 * a * b * d2.powf(b - 1.0) / (a * d2.powf(b) + 1.0) } else { 0.0 };
            let (gx, gy) = (clip(attract * dx), clip(attract * dy));
            data[j] += gx * alpha;
            data[j + 1] += gy * alpha;
            data[k] -= gx * alpha;
            data[k + 1] -= gy * alpha;
            next[e] += per_sample[e];
            let count = ((epoch as f32 - next_negative[e]) / per_negative[e]) as usize;
            for _ in 0..count {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let other = (rng % n as u64) as usize * 2;
                if other == j {
                    continue;
                }
                let (dx, dy) = (data[j] - data[other], data[j + 1] - data[other + 1]);
                let d2 = dx * dx + dy * dy;
                let (gx, gy) = if d2 > 0.0 {
                    let repel = 2.0 * gamma * b / ((0.001 + d2) * (a * d2.powf(b) + 1.0));
                    (clip(repel * dx), clip(repel * dy))
                } else {
                    (4.0, 4.0)
                };
                data[j] += gx * alpha;
                data[j + 1] += gy * alpha;
            }
            next_negative[e] += count as f32 * per_negative[e];
        }
        progress("arranging the photos", epoch + 1, epochs);
    }
    Some(emb)
}

/// Each photo's `k` nearest photos by cosine (itself first, distance 0): the indices and the Euclidean distances (the vectors are unit length).
/// None if `paused()` said so.
fn neighbours(pcs: &Array2<f32>, k: usize, progress: &(dyn Fn(&str, usize, usize) + Sync), paused: &(dyn Fn() -> bool + Sync)) -> Option<(Array2<u32>, Array2<f32>)> {
    let n = pcs.nrows();
    let mut indices = Array2::<u32>::zeros((n, k));
    let mut dists = Array2::<f32>::zeros((n, k));
    let block = 256;
    let blocks = n.div_ceil(block);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let stopped = std::sync::atomic::AtomicBool::new(false);
    let pcs_t = pcs.t().to_owned();
    indices
        .axis_chunks_iter_mut(Axis(0), block)
        .zip(dists.axis_chunks_iter_mut(Axis(0), block))
        .enumerate()
        .par_bridge()
        .for_each(|(b, (mut idx, mut dist))| {
            if stopped.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            if paused() {
                stopped.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }
            let rows = pcs.slice(ndarray::s![b * block..(b * block + block).min(n), ..]);
            let sims = rows.dot(&pcs_t);
            let mut order: Vec<u32> = Vec::with_capacity(n);
            for (r, sim) in sims.axis_iter(Axis(0)).enumerate() {
                let me = (b * block + r) as u32;
                order.clear();
                order.extend(0..n as u32);
                let by_sim = |a: &u32, c: &u32| sim[*c as usize].total_cmp(&sim[*a as usize]);
                order.select_nth_unstable_by(k - 1, by_sim);
                let mut top = order[..k].to_vec();
                top.sort_by(by_sim);
                // the photo itself comes first (a duplicate of it could tie)
                if let Some(at) = top.iter().position(|&t| t == me) {
                    top.swap(0, at);
                } else {
                    top[0] = me;
                }
                for (j, &t) in top.iter().enumerate() {
                    idx[(r, j)] = t;
                    dist[(r, j)] = if t == me { 0.0 } else { (2.0 - 2.0 * sim[t as usize]).max(0.0).sqrt() };
                }
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress("finding each photo's neighbours", d, blocks);
        });
    (!stopped.load(std::sync::atomic::Ordering::Relaxed)).then_some((indices, dists))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups(groups: usize, per: usize, dim: usize) -> Array2<f32> {
        let mut rng = Rng(7);
        let centres: Vec<Vec<f32>> = (0..groups).map(|_| (0..dim).map(|_| rng.next()).collect()).collect();
        let mut v = Array2::<f32>::zeros((groups * per, dim));
        for g in 0..groups {
            for i in 0..per {
                for d in 0..dim {
                    v[(g * per + i, d)] = centres[g][d] + 0.15 * rng.next();
                }
            }
        }
        normalise_rows(&mut v);
        v
    }

    /// Nearest neighbours of photos in a tight group are in the same group, itself first.
    #[test]
    fn neighbours_are_in_the_group() {
        let (g, per) = (4, 150);
        let v = groups(g, per, 64);
        let (idx, dist) = neighbours(&v, 11, &|_, _, _| {}, &|| false).unwrap();
        let same = (0..g * per).map(|i| idx.row(i).iter().filter(|&&j| j as usize / per == i / per).count()).sum::<usize>();
        assert_eq!(idx[(5, 0)], 5);
        assert_eq!(dist[(5, 0)], 0.0);
        assert_eq!(same, g * per * 11);
    }

    /// Points in four tight groups on a sphere end up in four separate places, each group together.
    #[test]
    fn groups_stay_together() {
        let (groups, per, dim) = (4, 150, 64);
        let v = self::groups(groups, per, dim);
        let xy = layout(v, &|_, _, _| {}, &|| false).unwrap().unwrap();
        assert_eq!(xy.len(), groups * per * 2);
        let at = |i: usize| (xy[i * 2], xy[i * 2 + 1]);
        let dist = |a: (f32, f32), b: (f32, f32)| ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt();
        let centre = |g: usize| {
            let (sx, sy) = (0..per).fold((0.0, 0.0), |(sx, sy), i| (sx + at(g * per + i).0, sy + at(g * per + i).1));
            (sx / per as f32, sy / per as f32)
        };
        let within: f32 = (0..groups).map(|g| (0..per).map(|i| dist(at(g * per + i), centre(g))).sum::<f32>() / per as f32).sum::<f32>() / groups as f32;
        let between = (1..groups).map(|g| dist(centre(0), centre(g))).fold(f32::MAX, f32::min);
        eprintln!("average distance within a group {within:.0}, nearest other group {between:.0} (of {WORLD})");
        assert!(between > 3.0 * within);
    }
}


#[cfg(test)]
mod compare {
    use super::*;

    /// (Run by hand: cargo test --release layout_of_a_real_map -- --ignored --nocapture.) The Rust layout of a finished map's photos
    /// against the layout Python made: time, and how many of a photo's 10 most similar photos land among its 10 nearest on the map.
    #[test]
    #[ignore]
    fn layout_of_a_real_map() {
        let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("maps/flickr30k.photomap");
        let n = 31_014;
        let raw = std::fs::read(bundle.join("embeddings.bin")).unwrap();
        let v: Vec<f32> = raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        let vectors = Array2::from_shape_vec((n, 768), v).unwrap();
        let started = std::time::Instant::now();
        let last = std::sync::Mutex::new(String::new());
        let xy = layout(
            vectors.clone(),
            &|what, done, total| {
                let mut l = last.lock().unwrap();
                if *l != what {
                    eprintln!("[{:5.1}s] {what} ({total})", started.elapsed().as_secs_f32());
                    *l = what.to_string();
                }
                let _ = done;
            },
            &|| false,
        )
        .unwrap()
        .unwrap();
        eprintln!("layout took {:.1}s", started.elapsed().as_secs_f32());
        let python: Vec<f32> = std::fs::read(bundle.join("layout.bin")).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let precision = |pos: &[f32]| {
            let sample: Vec<usize> = (0..n).step_by(n / 600).collect();
            let hits: usize = sample
                .par_iter()
                .map(|&i| {
                    let sims: Vec<f32> = (0..n).map(|j| vectors.row(i).dot(&vectors.row(j))).collect();
                    let d2 = |j: usize| (pos[j * 2] - pos[i * 2]).powi(2) + (pos[j * 2 + 1] - pos[i * 2 + 1]).powi(2);
                    let mut by_sim: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                    by_sim.sort_by(|&a, &b| sims[b].total_cmp(&sims[a]));
                    let mut by_map: Vec<usize> = (0..n).filter(|&j| j != i).collect();
                    by_map.sort_by(|&a, &b| d2(a).total_cmp(&d2(b)));
                    by_sim[..10].iter().filter(|j| by_map[..10].contains(j)).count()
                })
                .sum();
            hits as f32 / (sample.len() * 10) as f32
        };
        eprintln!("10 most similar photos among the 10 nearest on the map: rust {:.3}, python {:.3}", precision(&xy), precision(&python));
    }
}
