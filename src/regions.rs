//! Regions: connected areas of the map at several zoom levels, finer ones inside coarser ones.
//! The photos are first grouped into small cells (k-means on their positions, about 8 photos each); neighbouring cells (Delaunay) are
//! merged, always the pair that keeps the groups most compact first (Ward), into one tree; cutting the tree at several heights gives
//! the levels. Every region is one connected piece of the map.

use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const STEP: usize = 3; // each level has 3x the regions of the one above ...
const LEAST: usize = 10; // ... down to about 10 photos a region

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Each point's nearest centre, by a grid of the centres (2D, so rings of cells are enough).
fn assign(xy: &[f32], centres: &[[f32; 2]], world: f32) -> Vec<u32> {
    let g = ((centres.len() as f32).sqrt().ceil() as usize).max(1);
    let cell = world / g as f32;
    let at = |v: f32| ((v / cell) as usize).min(g - 1);
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); g * g];
    for (i, c) in centres.iter().enumerate() {
        grid[at(c[1]) * g + at(c[0])].push(i as u32);
    }
    xy.par_chunks_exact(2)
        .map(|p| {
            let (cx, cy) = (at(p[0]) as isize, at(p[1]) as isize);
            let (mut best, mut best_d2) = (0u32, f32::MAX);
            for r in 0..=g as isize {
                for y in (cy - r).max(0)..=(cy + r).min(g as isize - 1) {
                    for x in (cx - r).max(0)..=(cx + r).min(g as isize - 1) {
                        if (x - cx).abs().max((y - cy).abs()) != r {
                            continue; // only the ring
                        }
                        for &c in &grid[y as usize * g + x as usize] {
                            let d2 = (centres[c as usize][0] - p[0]).powi(2) + (centres[c as usize][1] - p[1]).powi(2);
                            if d2 < best_d2 {
                                (best, best_d2) = (c, d2);
                            }
                        }
                    }
                }
                if best_d2 <= (r as f32 * cell).powi(2) {
                    break;
                }
            }
            best
        })
        .collect()
}

/// Small cells: k-means on the positions. The centres, and each point's cell (cells nobody is in are dropped).
fn cells(xy: &[f32], world: f32) -> (Vec<[f32; 2]>, Vec<u32>) {
    let n = xy.len() / 2;
    let m = (n / 8).clamp(20, 12_000).min(n);
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut centres: Vec<[f32; 2]> = (0..m)
        .map(|_| {
            let i = (rng.next() % n as u64) as usize;
            [xy[i * 2], xy[i * 2 + 1]]
        })
        .collect();
    let mut labels = assign(xy, &centres, world);
    for _ in 0..12 {
        let mut sums = vec![[0f64; 2]; m];
        let mut counts = vec![0usize; m];
        for (p, &l) in xy.chunks_exact(2).zip(&labels) {
            sums[l as usize][0] += p[0] as f64;
            sums[l as usize][1] += p[1] as f64;
            counts[l as usize] += 1;
        }
        for c in 0..m {
            if counts[c] > 0 {
                centres[c] = [(sums[c][0] / counts[c] as f64) as f32, (sums[c][1] / counts[c] as f64) as f32];
            } else {
                let i = (rng.next() % n as u64) as usize;
                centres[c] = [xy[i * 2], xy[i * 2 + 1]];
            }
        }
        labels = assign(xy, &centres, world);
    }
    let mut used = vec![u32::MAX; m];
    let mut kept = Vec::new();
    for &l in &labels {
        if used[l as usize] == u32::MAX {
            used[l as usize] = 0;
        }
    }
    for c in 0..m {
        if used[c] != u32::MAX {
            used[c] = kept.len() as u32;
            kept.push(centres[c]);
        }
    }
    (kept, labels.iter().map(|&l| used[l as usize]).collect())
}

/// Per level, a region number for every photo (coarse levels first); `xy`: x, y per photo within 0 ... `world`.
/// None if `paused()` said so.
pub fn cluster(xy: &[f32], world: f32, progress: &dyn Fn(&str, usize, usize), paused: &dyn Fn() -> bool) -> Option<Vec<Vec<u32>>> {
    let n = xy.len() / 2;
    progress("grouping photos into small cells", 0, 3);
    let (centres, cell_of) = cells(xy, world);
    let m = centres.len();
    if paused() {
        return None;
    }

    // which cells touch (Delaunay); if the cells are all in a line, neighbours in order
    progress("linking neighbouring cells", 1, 3);
    let mut adjacent: Vec<Vec<u32>> = vec![Vec::new(); m];
    let points: Vec<delaunator::Point> = centres.iter().map(|c| delaunator::Point { x: c[0] as f64, y: c[1] as f64 }).collect();
    let tri = delaunator::triangulate(&points);
    let mut link = |a: usize, b: usize| {
        if a != b && !adjacent[a].contains(&(b as u32)) {
            adjacent[a].push(b as u32);
            adjacent[b].push(a as u32);
        }
    };
    if tri.triangles.is_empty() {
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| (centres[a][0] + centres[a][1]).total_cmp(&(centres[b][0] + centres[b][1])));
        for w in order.windows(2) {
            link(w[0], w[1]);
        }
    }
    for t in tri.triangles.chunks_exact(3) {
        link(t[0], t[1]);
        link(t[1], t[2]);
        link(t[0], t[2]);
    }

    // merge: Ward's cost of joining two groups = size a * size b / (size a + size b) * squared distance of their centres
    progress("merging them into regions", 2, 3);
    let total = 2 * m - 1;
    let mut size = vec![0f64; total];
    let mut centre = vec![[0f64; 2]; total];
    let mut alive = vec![false; total];
    let mut neighbours: Vec<Vec<u32>> = adjacent;
    neighbours.resize(total, Vec::new());
    for (c, p) in centres.iter().enumerate() {
        centre[c] = [p[0] as f64, p[1] as f64];
    }
    for &c in &cell_of {
        size[c as usize] += 1.0;
    }
    alive[..m].fill(true);
    let cost = |size: &[f64], centre: &[[f64; 2]], a: usize, b: usize| {
        let d2 = (centre[a][0] - centre[b][0]).powi(2) + (centre[a][1] - centre[b][1]).powi(2);
        size[a] * size[b] / (size[a] + size[b]) * d2
    };
    let mut heap: BinaryHeap<(Reverse<u64>, u32, u32)> = BinaryHeap::new();
    for a in 0..m {
        for &b in &neighbours[a] {
            if (b as usize) > a {
                heap.push((Reverse(cost(&size, &centre, a, b as usize).to_bits()), a as u32, b));
            }
        }
    }
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(m - 1);
    while children.len() < m - 1 {
        let Some((_, a, b)) = heap.pop() else { break };
        let (a, b) = (a as usize, b as usize);
        if !alive[a] || !alive[b] {
            continue;
        }
        let id = m + children.len();
        children.push((a, b));
        alive[a] = false;
        alive[b] = false;
        alive[id] = true;
        size[id] = size[a] + size[b];
        centre[id] = [(centre[a][0] * size[a] + centre[b][0] * size[b]) / size[id], (centre[a][1] * size[a] + centre[b][1] * size[b]) / size[id]];
        let mut next: Vec<u32> = neighbours[a].iter().chain(&neighbours[b]).copied().filter(|&c| c as usize != a && c as usize != b && alive[c as usize]).collect();
        next.sort_unstable();
        next.dedup();
        for &c in &next {
            let list = &mut neighbours[c as usize];
            list.retain(|&x| x as usize != a && x as usize != b);
            list.push(id as u32);
            heap.push((Reverse(cost(&size, &centre, id, c as usize).to_bits()), id as u32, c));
        }
        neighbours[id] = next;
        neighbours[a] = Vec::new();
        neighbours[b] = Vec::new();
    }
    if paused() {
        return None;
    }

    // the levels: apply the merges until k groups are left, for k from many to few
    let mut sizes = vec![(((n as f64).sqrt() / 25.0).round() as usize).max(4)];
    while sizes.last().unwrap() * STEP * LEAST <= n && sizes.last().unwrap() * STEP < m {
        sizes.push(sizes.last().unwrap() * STEP);
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let mut parent: Vec<usize> = (0..total).collect();
    let mut merged = 0;
    let mut levels = Vec::new();
    for k in sizes {
        while m - merged > k && merged < children.len() {
            let (a, b) = children[merged];
            parent[a] = m + merged;
            parent[b] = m + merged;
            merged += 1;
        }
        let root = |mut c: usize| {
            while parent[c] != c {
                c = parent[c];
            }
            c
        };
        let roots: Vec<usize> = (0..m).map(root).collect();
        let mut number = std::collections::HashMap::new();
        let mut order: Vec<usize> = roots.clone();
        order.sort_unstable();
        order.dedup();
        for (i, r) in order.into_iter().enumerate() {
            number.insert(r, i as u32);
        }
        levels.push(cell_of.iter().map(|&c| number[&roots[c as usize]]).collect::<Vec<u32>>());
    }
    levels.reverse();
    Some(levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four blobs of points: the coarsest level that has four or more regions puts each blob in regions of its own, and
    /// finer levels nest in coarser ones.
    #[test]
    fn blobs_become_regions_that_nest() {
        let mut rng = Rng(99);
        let mut xy = Vec::new();
        for (cx, cy) in [(2000.0, 2000.0), (8000.0, 2500.0), (2500.0, 8000.0), (8000.0, 8000.0)] {
            for _ in 0..1500 {
                xy.push(cx + (rng.next() % 1000) as f32 - 500.0);
                xy.push(cy + (rng.next() % 1000) as f32 - 500.0);
            }
        }
        let levels = cluster(&xy, 10_000.0, &|_, _, _| {}, &|| false).unwrap();
        assert!(levels.len() >= 2, "{} levels", levels.len());
        let counts: Vec<u32> = levels.iter().map(|l| l.iter().max().unwrap() + 1).collect();
        eprintln!("regions per level: {counts:?}");
        assert!(counts.windows(2).all(|w| w[0] < w[1]), "coarse to fine: {counts:?}");
        // nesting: every fine region lies in exactly one coarse region
        for pair in levels.windows(2) {
            let mut parent = std::collections::HashMap::new();
            for (fine, coarse) in pair[1].iter().zip(&pair[0]) {
                assert_eq!(*parent.entry(fine).or_insert(coarse), coarse);
            }
        }
        // a region at the level with 4+ regions never spans two blobs
        let level = levels.iter().find(|l| l.iter().max().unwrap() + 1 >= 4).unwrap();
        for blob in 0..4 {
            let mine: std::collections::HashSet<_> = (blob * 1500..(blob + 1) * 1500).map(|i| level[i]).collect();
            for other in 0..4 {
                if other != blob {
                    assert!((other * 1500..(other + 1) * 1500).all(|i| !mine.contains(&level[i])), "blob {blob} shares a region with {other}");
                }
            }
        }
    }
}
