//! The map's tree (tree.bin, made with the map): all the photos joined by the shortest links that join them all (the minimum
//! spanning tree; a link's length is 1 - cosine of the two photos' vectors). Between any two photos, the way along the tree is
//! the chain whose biggest step is the smallest of all chains between them, and the same is true of every piece of it: a path
//! between two photos (right-click) is that way, worked out at once from the tree.
//!
//! Made from each photo's nearest photos, found among the photos of the PROBE cells nearest to it (crate::cells), so that a photo is compared with a few thousand others, not with
//! all: the time grows with n^1.5, not n². Then Borůvka: each round, every group of joined photos takes its shortest link to
//! another group. A photo's nearest outside its group is the first of its list that is outside. Only a photo whose whole list
//! is inside its group, and could still have a shorter link than the group's best, is looked up again, in its nearest cells
//! that hold photos outside its group; its list becomes its nearest outside the group, which stays right as the group grows
//! (what is outside only shrinks). A link outside a photo's cells is missed, and a little longer one taken instead: on
//! Wikivoyage (88,000 photos) 99.3% of the links are those of comparing all with all, 97% of paths between random photos are
//! the same, the rest's biggest step at most 0.03 longer; 9 s instead of 47.

use crate::cells::{Cells, Rows};
use rayon::prelude::*;
use std::path::Path;

pub const FILE: &str = "tree.bin";
/// How many of its nearest photos each photo keeps (more: fewer photos looked up again later, more memory).
const NEAREST: usize = 32;
/// How many of its nearest cells (crate::cells) each photo is compared with (Wikivoyage: 8, 95% of the links of comparing all with all;
/// 16, 98%; 32, 99.3%).
const PROBE: usize = 32;

/// A link: its length, and the photos it joins (the smaller number first). Ordered by all three, so that links of the same
/// length still have one order (Borůvka needs it).
type Link = (f32, u32, u32);

fn link(d: f32, a: usize, b: usize) -> Link {
    (d, a.min(b) as u32, a.max(b) as u32)
}

fn shorter(a: Link, b: Link) -> bool {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)).is_lt()
}

/// Each photo's parent in the tree (photo 0, the root, has itself), from the photos' vectors (n x dim, unit length).
/// `progress(text, done, total)`; None if `paused()` said so.
pub fn build<R: Rows>(v: &R, progress: &(dyn Fn(&str, usize, usize) + Sync), paused: &(dyn Fn() -> bool + Sync)) -> Option<Vec<u32>> {
    grow(v, PROBE, progress, paused)
}

/// `build`, each photo compared with the photos of its `probe` nearest cells.
fn grow<R: Rows>(v: &R, probe: usize, progress: &(dyn Fn(&str, usize, usize) + Sync), paused: &(dyn Fn() -> bool + Sync)) -> Option<Vec<u32>> {
    let n = v.count();
    if n == 0 {
        return Some(Vec::new());
    }
    let cells = Cells::new(v, progress, paused)?;
    let mut sets = Sets((0..n as u32).collect());
    // each photo's group: the photo that stands for it (flattened at the start of each round)
    let mut group: Vec<u32> = (0..n as u32).collect();
    let all: Vec<usize> = (0..n).collect();
    let mut lists = vec![Vec::new(); n];
    for (i, list) in all.iter().zip(cells.nearest_outside(probe, NEAREST, &all, &group, "finding each photo's nearest photos", progress, paused)?) {
        lists[*i] = list;
    }
    let mut links: Vec<(u32, u32)> = Vec::with_capacity(n - 1);
    while links.len() < n - 1 {
        if paused() {
            return None;
        }
        progress("joining the photos into a tree", links.len(), n - 1);
        for i in 0..n {
            group[i] = sets.find(i as u32);
        }
        let first = |list: &[(f32, u32)], i: usize| list.iter().find(|&&(_, j)| group[j as usize] != group[i]).copied();
        // each group's shortest link out, from the lists
        let mut best: Vec<Option<Link>> = vec![None; n];
        let offer = |best: &mut Vec<Option<Link>>, i: usize, (d, j): (f32, u32)| {
            let (g, l) = (group[i] as usize, link(d, i, j as usize));
            if best[g].is_none_or(|b| shorter(l, b)) {
                best[g] = Some(l);
            }
        };
        let found: Vec<Option<(f32, u32)>> = (0..n).into_par_iter().map(|i| first(&lists[i], i)).collect();
        for (i, f) in found.iter().enumerate() {
            if let Some(f) = *f {
                offer(&mut best, i, f);
            }
        }
        // photos whose lists are all inside their group: anything outside is at least as far as the last of the list
        let again: Vec<usize> = (0..n)
            .filter(|&i| found[i].is_none())
            .filter(|&i| lists[i].last().is_none_or(|&(d, _)| best[group[i] as usize].is_none_or(|b| d < b.0)))
            .collect();
        if !again.is_empty() {
            let fresh = cells.nearest_outside(probe, NEAREST, &again, &group, "joining the photos into a tree", &|_, _, _| {}, paused)?;
            for (&i, list) in again.iter().zip(fresh) {
                if let Some(f) = first(&list, i) {
                    offer(&mut best, i, f);
                }
                lists[i] = list;
            }
        }
        let before = links.len();
        for &(_, a, b) in best.iter().flatten() {
            if sets.join(a, b) {
                links.push((a, b));
            }
        }
        if links.len() == before {
            break; // no group could reach another (cannot happen: every group has a cell with photos outside it)
        }
    }
    // hung from photo 0
    let mut next: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in &links {
        next[a as usize].push(b);
        next[b as usize].push(a);
    }
    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut seen = vec![false; n];
    let mut todo = vec![0u32];
    seen[0] = true;
    while let Some(i) = todo.pop() {
        for &j in &next[i as usize] {
            if !seen[j as usize] {
                seen[j as usize] = true;
                parent[j as usize] = i;
                todo.push(j);
            }
        }
    }
    Some(parent)
}

/// Which photos are joined so far (union-find).
struct Sets(Vec<u32>);

impl Sets {
    fn find(&mut self, mut i: u32) -> u32 {
        while self.0[i as usize] != i {
            self.0[i as usize] = self.0[self.0[i as usize] as usize];
            i = self.0[i as usize];
        }
        i
    }

    /// Join the two photos' groups: false if they were one already.
    fn join(&mut self, a: u32, b: u32) -> bool {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a.max(b) as usize] = a.min(b);
        }
        a != b
    }
}

/// tree.bin: each photo's parent, as u32 (little-endian).
pub fn write(bundle: &Path, parent: &[u32]) -> std::io::Result<()> {
    crate::build::write_atomic(&bundle.join(FILE), &parent.iter().flat_map(|p| p.to_le_bytes()).collect::<Vec<u8>>())
}

/// A map's tree, if it has one for its `n` photos.
pub fn read(bundle: &Path, n: usize) -> Option<Vec<u32>> {
    let raw = std::fs::read(bundle.join(FILE)).ok()?;
    let parent: Vec<u32> = raw.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    (raw.len() == n * 4 && parent.iter().all(|&p| (p as usize) < n)).then_some(parent)
}

/// The way along the tree from photo `a` to `b`, both included.
pub fn route(parent: &[u32], a: usize, b: usize) -> Vec<usize> {
    // each one up to the root (no further than there are photos, whatever the file says), then the shared part dropped
    let up = |mut i: usize| {
        let mut way = vec![i];
        while parent[i] as usize != i && way.len() <= parent.len() {
            i = parent[i] as usize;
            way.push(i);
        }
        way
    };
    let (from_a, from_b) = (up(a), up(b));
    let mut shared = 0;
    while shared < from_a.len().min(from_b.len()) && from_a[from_a.len() - 1 - shared] == from_b[from_b.len() - 1 - shared] {
        shared += 1;
    }
    // a's way up to where they meet (that one included), then b's way down from there
    let mut way = from_a[..=from_a.len() - shared.max(1)].to_vec();
    way.extend(from_b[..from_b.len() - shared.max(1)].iter().rev());
    way
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Axis};

    fn unit(rows: Vec<Vec<f32>>) -> Array2<f32> {
        let dim = rows[0].len();
        let mut v = Array2::from_shape_vec((rows.len(), dim), rows.into_iter().flatten().collect()).unwrap();
        for mut r in v.axis_iter_mut(Axis(0)) {
            let norm = r.dot(&r).sqrt();
            r.mapv_inplace(|x| x / norm);
        }
        v
    }

    /// A small random set: looking in every cell, the tree is the true minimum spanning tree (its total length the same as
    /// Prim's, comparing all with all), with more photos than NEAREST so that the lists run out and photos are looked up
    /// again; looking in the nearest cells only, it is hardly longer.
    #[test]
    fn the_tree_is_the_shortest() {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        // clumps, so that groups fill up their lists before they reach each other
        let centres: Vec<Vec<f32>> = (0..6).map(|_| (0..16).map(|_| rnd()).collect()).collect();
        let v = unit((0..300).map(|i| centres[i % 6].iter().map(|c| c + 0.1 * rnd()).collect()).collect());
        let n = v.nrows();
        let d = |i: usize, j: usize| 1.0 - v.row(i).dot(&v.row(j));
        let length = |probe: usize| {
            let parent = grow(&v, probe, &|_, _, _| {}, &|| false).unwrap();
            // one tree: every photo reaches photo 0
            assert!((0..n).all(|i| *route(&parent, i, 0).last().unwrap() == 0));
            (1..n).map(|i| d(i, parent[i] as usize)).sum::<f32>()
        };
        // Prim
        let mut inside = vec![false; n];
        let mut best = vec![f32::INFINITY; n];
        best[0] = 0.0;
        let mut prim = 0.0;
        for _ in 0..n {
            let i = (0..n).filter(|&i| !inside[i]).min_by(|&a, &b| best[a].total_cmp(&best[b])).unwrap();
            inside[i] = true;
            prim += best[i];
            for j in 0..n {
                best[j] = best[j].min(d(i, j));
            }
        }
        let (all, near) = (length(usize::MAX), length(PROBE));
        assert!((all - prim).abs() < 1e-3, "tree {all}, Prim {prim}");
        assert!(near < prim * 1.01, "nearest cells {near}, Prim {prim}");
    }

    /// The way between two photos goes up to where their branches meet and down again, without the part above.
    #[test]
    fn the_way_goes_through_where_the_branches_meet() {
        //      0
        //     / \
        //    1   4
        //   / \
        //  2   3
        let parent = [0, 0, 1, 1, 0];
        assert_eq!(route(&parent, 2, 3), vec![2, 1, 3]);
        assert_eq!(route(&parent, 2, 4), vec![2, 1, 0, 4]);
        assert_eq!(route(&parent, 4, 2), vec![4, 0, 1, 2]);
        assert_eq!(route(&parent, 1, 2), vec![1, 2]);
        assert_eq!(route(&parent, 3, 3), vec![3]);
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use ndarray::Array2;

    /// (Run by hand: MAP=maps/<name>.photomap cargo test --release tree_of_a_real_map -- --ignored --nocapture; WRITE=1 also
    /// saves the tree in the map.) How long a real map's tree takes, and how long paths between random photos are.
    #[test]
    #[ignore]
    fn tree_of_a_real_map() {
        let bundle = crate::data_root().join(std::env::var("MAP").unwrap_or("maps/rare-species.photomap".into()));
        let raw = std::fs::read(bundle.join("embeddings.bin")).unwrap();
        let v: Vec<f32> = raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        let n = v.len() / 768;
        let v = Array2::from_shape_vec((n, 768), v).unwrap();
        let started = std::time::Instant::now();
        let last = std::sync::Mutex::new(String::new());
        let parent = build(
            &v,
            &|what, _, _| {
                let mut l = last.lock().unwrap();
                if *l != what {
                    eprintln!("[{:5.1}s] {what}", started.elapsed().as_secs_f32());
                    *l = what.to_string();
                }
            },
            &|| false,
        )
        .unwrap();
        eprintln!("{n} photos: tree in {:.1}s", started.elapsed().as_secs_f32());
        let mut lens: Vec<usize> = (0..30).map(|k| route(&parent, (k * 7919) % n, (k * 104729 + 13) % n).len()).collect();
        lens.sort();
        eprintln!("path lengths {lens:?}");
        if std::env::var("WRITE").is_ok() {
            write(&bundle, &parent).unwrap();
            eprintln!("wrote {}", bundle.join(FILE).display());
        }
    }
}
