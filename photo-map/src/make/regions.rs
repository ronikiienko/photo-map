//! Names for the map's regions at 3 levels of detail (coarse / middle / fine), shown by the viewer as you zoom.
//! 1. Regions: the photos' map positions are cut into small cells (k-means, ~8 photos each), which are merged, always
//!    with a neighbouring cell, into one tree (Ward: first the merge that keeps groups most compact). Cutting the tree
//!    at several heights gives the levels, each with ~3x the regions of the one above, down to ~10 photos a region:
//!    every region is one connected piece of the map, and finer ones nest in coarser ones.
//! 2. Names, from one big general vocabulary (Open Images' ~20k labels and Places365's scenes, data/vocab/), embedded
//!    as search queries. How well a word fits a group of photos: how far it points in the direction they differ from
//!    the average photo. A region's name has to fit all of its parts (its sub-regions; for the finest level, its
//!    cells): a word scores its average fit over the parts minus how unevenly it fits them. That's what makes big
//!    regions get general words and small ones specific words, without any hand-picked lists. Best matches are
//!    assigned first; a name is used once among regions that share a parent and never repeats one of a region it
//!    sits in; tiny regions, and ones with exactly the photos of the region around them, get none.
//! Written to regions.json: per level, the named regions with their label position and size.

use super::embed::{DIM, Python, read_f32};
use super::step;
use crate::data::Photo;
use rayon::prelude::*;
use serde::Serialize;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io;
use std::path::Path;

const PROMPT: &str = "task: search result | query: ";
/// A word's score for a region: its average fit over the region's parts minus this many standard deviations.
const EVENNESS: f32 = 1.0;
/// Each level has this many times the regions of the one above, down to about LEAST photos per region.
const STEP: usize = 3;
const LEAST: usize = 10;
/// Coarsest-level regions under this share of the photos, and other regions under MIN_PHOTOS, get no name.
const MIN_SHARE: f32 = 0.01;
const MIN_PHOTOS: usize = 10;
#[derive(Serialize)]
struct Label {
    name: String,
    /// Where the label goes: the region's most central photo.
    x: f32,
    y: f32,
    /// How far the region's photos lie from its centre (root mean square), in world units: the viewer fades the
    /// label in and out by how big this is on screen.
    spread: f32,
    photos: usize,
}

pub fn regions(project: &Path, python: &mut Python, bundle: &Path, photos: &[Photo], vectors: &[f32], xy: &[[f32; 2]]) -> io::Result<()> {
    let n = photos.len();
    let mut sizes = vec![((n as f32).sqrt() / 25.0).round().max(4.0) as usize];
    while sizes.last().unwrap() * STEP * LEAST <= n {
        sizes.push(sizes.last().unwrap() * STEP);
    }
    let (levels, cells) = cluster(xy, &sizes);
    step(format!("regions: levels of {:?}", levels.iter().map(|l| l.iter().max().map_or(0, |m| m + 1)).collect::<Vec<_>>()));

    let words = vocabulary(project);
    if words.is_empty() {
        return Err(io::Error::other("no vocabulary in data/vocab/ (places365.txt, openimages_all.csv)"));
    }
    let wvecs = word_vectors(project, python, bundle, &words)?;
    step(format!("{} words to choose names from", words.len()));

    let names = name(vectors, &levels, &cells, &words, &wvecs);
    let out: Vec<Vec<Label>> = levels
        .iter()
        .zip(&names)
        .map(|(lab, names)| {
            let k = names.len();
            let mut sum = vec![[0f64; 2]; k];
            let mut count = vec![0usize; k];
            for (&r, p) in lab.iter().zip(xy) {
                sum[r as usize][0] += p[0] as f64;
                sum[r as usize][1] += p[1] as f64;
                count[r as usize] += 1;
            }
            let mean: Vec<[f32; 2]> = sum.iter().zip(&count).map(|(s, &c)| [(s[0] / c as f64) as f32, (s[1] / c as f64) as f32]).collect();
            let mut best = vec![(f32::MAX, 0usize); k];
            let mut spread = vec![0f64; k];
            for (i, (&r, p)) in lab.iter().zip(xy).enumerate() {
                let (dx, dy) = (p[0] - mean[r as usize][0], p[1] - mean[r as usize][1]);
                let d = dx * dx + dy * dy;
                spread[r as usize] += d as f64;
                if d < best[r as usize].0 {
                    best[r as usize] = (d, i);
                }
            }
            (0..k)
                .filter_map(|r| {
                    let name = names[r].clone()?;
                    let [x, y] = xy[best[r].1];
                    Some(Label { name, x, y, spread: (spread[r] / count[r] as f64).sqrt() as f32, photos: count[r] })
                })
                .collect()
        })
        .collect();
    std::fs::write(bundle.join("regions.json"), serde_json::to_vec(&serde_json::json!({ "levels": out }))?)?;
    step(format!("regions.json written: {} names", out.iter().map(Vec::len).sum::<usize>()));
    Ok(())
}

/// Per level (coarse first): each photo's region, numbered 0, 1, ...; and each photo's cell (the tree's leaves).
fn cluster(xy: &[[f32; 2]], sizes: &[usize]) -> (Vec<Vec<u32>>, Vec<u32>) {
    let n = xy.len();
    // 1. small cells: k-means on the positions
    let k = (n / 8).clamp(20, 12_000).min(n);
    let mut centers: Vec<[f32; 2]> = (0..k).map(|i| xy[i * n / k]).collect();
    let mut cell = vec![0u32; n];
    for _ in 0..25 {
        cell.par_iter_mut().zip(xy).for_each(|(c, p)| {
            let d = |q: &[f32; 2]| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
            *c = (0..centers.len()).min_by(|&a, &b| d(&centers[a]).total_cmp(&d(&centers[b]))).unwrap() as u32;
        });
        let mut sum = vec![[0f64; 3]; centers.len()];
        for (&c, p) in cell.iter().zip(xy) {
            sum[c as usize][0] += p[0] as f64;
            sum[c as usize][1] += p[1] as f64;
            sum[c as usize][2] += 1.0;
        }
        for (c, s) in centers.iter_mut().zip(&sum) {
            if s[2] > 0.0 {
                *c = [(s[0] / s[2]) as f32, (s[1] / s[2]) as f32];
            }
        }
    }
    // drop empty cells
    let mut count = vec![0usize; centers.len()];
    cell.iter().for_each(|&c| count[c as usize] += 1);
    let mut renumber = vec![u32::MAX; centers.len()];
    let mut cells: Vec<([f64; 2], f64)> = Vec::new(); // (centre, photos)
    for (c, &m) in count.iter().enumerate() {
        if m > 0 {
            renumber[c] = cells.len() as u32;
            cells.push(([centers[c][0] as f64, centers[c][1] as f64], m as f64));
        }
    }
    cell.iter_mut().for_each(|c| *c = renumber[*c as usize]);
    let k = cells.len();

    // 2. neighbours: each cell's 6 nearest cells, both ways
    let mut nbrs: Vec<HashSet<usize>> = vec![HashSet::new(); k];
    for a in 0..k {
        let mut near: Vec<(f64, usize)> = (0..k).filter(|&b| b != a).map(|b| (dist2(cells[a].0, cells[b].0), b)).collect();
        let six = near.len().min(6);
        if six < near.len() {
            near.select_nth_unstable_by(six, |x, y| x.0.total_cmp(&y.0));
        }
        for &(_, b) in &near[..six] {
            nbrs[a].insert(b);
            nbrs[b].insert(a);
        }
    }

    // 3. Ward merging of neighbours; clusters k.. are merges, in order
    let ward = |a: &([f64; 2], f64), b: &([f64; 2], f64)| a.1 * b.1 / (a.1 + b.1) * dist2(a.0, b.0);
    let mut alive = vec![true; k];
    let mut heap = BinaryHeap::new();
    for a in 0..k {
        for &b in &nbrs[a] {
            if a < b {
                heap.push(Reverse((ward(&cells[a], &cells[b]).to_bits(), a, b)));
            }
        }
    }
    let mut merges = Vec::new();
    let mut left = k;
    while left > 1 {
        let Some(Reverse((_, a, b))) = heap.pop() else {
            // parts not connected by neighbours: let all remaining clusters merge with each other
            let rest: Vec<usize> = (0..cells.len()).filter(|&c| alive[c]).collect();
            for (i, &a) in rest.iter().enumerate() {
                for &b in &rest[i + 1..] {
                    nbrs[a].insert(b);
                    nbrs[b].insert(a);
                    heap.push(Reverse((ward(&cells[a], &cells[b]).to_bits(), a, b)));
                }
            }
            continue;
        };
        if !alive[a] || !alive[b] {
            continue;
        }
        let new = cells.len();
        let (ca, cb) = (cells[a], cells[b]);
        let m = ca.1 + cb.1;
        cells.push(([(ca.0[0] * ca.1 + cb.0[0] * cb.1) / m, (ca.0[1] * ca.1 + cb.0[1] * cb.1) / m], m));
        alive[a] = false;
        alive[b] = false;
        alive.push(true);
        let joined: HashSet<usize> = nbrs[a].union(&nbrs[b]).copied().filter(|&c| c != a && c != b && alive[c]).collect();
        for &c in &joined {
            nbrs[c].remove(&a);
            nbrs[c].remove(&b);
            nbrs[c].insert(new);
            heap.push(Reverse((ward(&cells[new], &cells[c]).to_bits(), c.min(new), c.max(new))));
        }
        nbrs.push(joined);
        merges.push((a, b, new));
        left -= 1;
    }

    // 4. cut the tree: replay the merges, noting each cell's cluster when the count reaches each level's size
    let mut parent: Vec<usize> = (0..cells.len()).collect();
    fn find(parent: &mut [usize], mut c: usize) -> usize {
        while parent[c] != c {
            parent[c] = parent[parent[c]];
            c = parent[c];
        }
        c
    }
    let snapshot = |parent: &mut Vec<usize>| -> Vec<u32> {
        let mut ids = HashMap::new();
        let of_cell: Vec<u32> = (0..k).map(|c| {
            let r = find(parent, c);
            let next = ids.len() as u32;
            *ids.entry(r).or_insert(next)
        }).collect();
        cell.iter().map(|&c| of_cell[c as usize]).collect()
    };
    let mut levels: Vec<Option<Vec<u32>>> = vec![None; sizes.len()];
    let mut count = k;
    for (l, &s) in sizes.iter().enumerate() {
        if s >= count {
            levels[l] = Some(snapshot(&mut parent));
        }
    }
    for &(a, b, new) in &merges {
        parent[a] = new;
        parent[b] = new;
        count -= 1;
        for (l, &s) in sizes.iter().enumerate() {
            if s == count && levels[l].is_none() {
                levels[l] = Some(snapshot(&mut parent));
            }
        }
    }
    (levels.into_iter().map(|l| l.unwrap_or_else(|| vec![0; n])).collect(), cell)
}

fn dist2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// The vocabulary: Places365 scenes ('/f/forest/broadleaf' -> 'broadleaf forest') and Open Images labels, lowercased.
fn vocabulary(project: &Path) -> Vec<String> {
    let dir = project.join("data/vocab");
    let mut words: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for line in std::fs::read_to_string(dir.join("places365.txt")).unwrap_or_default().lines() {
        let parts: Vec<&str> = line.split_whitespace().next().unwrap_or("").split('/').skip(2).collect();
        let Some((base, quals)) = parts.split_first() else { continue };
        words.push([quals, &[*base]].concat().join(" ").replace('_', " "));
    }
    for line in std::fs::read_to_string(dir.join("openimages_all.csv")).unwrap_or_default().lines() {
        // '/m/0f9_l,Egg (Food)' or '/m/x,"Name, with comma"'
        let name = line.split_once(',').map_or("", |x| x.1).trim().trim_matches('"').to_lowercase();
        words.push(name);
    }
    words.retain(|w| !w.is_empty() && seen.insert(w.clone()));
    words
}

/// Text vectors of the words, cached for the whole project in data/vocab/ (the model only runs for new words).
fn word_vectors(project: &Path, python: &mut Python, bundle: &Path, words: &[String]) -> io::Result<Vec<f32>> {
    let (cache_words, cache_vecs) = (project.join("data/vocab/text_cache.json"), project.join("data/vocab/text_cache.f32"));
    let mut known: Vec<String> = std::fs::read(&cache_words).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let mut vecs = if known.is_empty() { Vec::new() } else { read_f32(&cache_vecs)? };
    if vecs.len() != known.len() * DIM {
        (known, vecs) = (Vec::new(), Vec::new()); // cache damaged: start over
    }
    let index: HashSet<&String> = known.iter().collect();
    let new: Vec<String> = words.iter().filter(|w| !index.contains(w)).cloned().collect();
    if !new.is_empty() {
        step(format!("embedding {} new words", new.len()));
        let tmp = bundle.join("work/words.tmp");
        vecs.extend(python.ask(serde_json::json!({ "texts": new, "prompt": PROMPT }), &tmp)?);
        known.extend(new);
        std::fs::create_dir_all(cache_words.parent().unwrap())?;
        std::fs::write(&cache_vecs, bytemuck::cast_slice(&vecs))?;
        std::fs::write(&cache_words, serde_json::to_vec(&known)?)?;
    }
    let at: HashMap<&String, usize> = known.iter().enumerate().map(|(i, w)| (w, i)).collect();
    Ok(words.iter().flat_map(|w| vecs[at[w] * DIM..(at[w] + 1) * DIM].to_vec()).collect())
}

/// A word with its plural/singular endings off, so 'lake' and 'lakes' count as the same name.
fn forms(word: &str) -> Vec<String> {
    let mut out = vec![word.to_string(), format!("{word}s"), format!("{word}es")];
    if let Some(w) = word.strip_suffix("es") {
        out.push(w.to_string());
    }
    if let Some(w) = word.strip_suffix('s').filter(|_| !word.ends_with("ss")) {
        out.push(w.to_string());
    }
    out
}

/// Per level, per region: its name, or None.
fn name(vectors: &[f32], levels: &[Vec<u32>], cells: &[u32], words: &[String], wvecs: &[f32]) -> Vec<Vec<Option<String>>> {
    let n = levels[0].len();
    let mut mean = vec![0f64; DIM];
    for v in vectors.chunks_exact(DIM) {
        mean.iter_mut().zip(v).for_each(|(m, x)| *m += *x as f64);
    }
    let mean: Vec<f64> = mean.iter().map(|m| m / n as f64).collect();
    // how each group of photos differs from the average photo
    let diffs = |lab: &[u32]| -> (Vec<f32>, Vec<usize>) {
        let k = lab.iter().max().map_or(0, |m| *m as usize + 1);
        let mut sum = vec![0f64; k * DIM];
        let mut size = vec![0usize; k];
        for (&r, v) in lab.iter().zip(vectors.chunks_exact(DIM)) {
            sum[r as usize * DIM..(r as usize + 1) * DIM].iter_mut().zip(v).for_each(|(s, x)| *s += *x as f64);
            size[r as usize] += 1;
        }
        let d = (0..k * DIM).map(|i| (sum[i] / size[i / DIM].max(1) as f64 - mean[i % DIM]) as f32).collect();
        (d, size)
    };
    let fit = |diff: &[f32]| -> Vec<f32> {
        wvecs.chunks_exact(DIM).map(|w| w.iter().zip(diff).map(|(a, b)| a * b).sum()).collect()
    };

    let mut names: Vec<Vec<Option<String>>> = Vec::new();
    for (level, lab) in levels.iter().enumerate() {
        let (_, size) = diffs(lab);
        let k = size.len();
        // the parts: the next level's regions, or the cells for the finest level
        let parts = if level + 1 < levels.len() { &levels[level + 1] } else { cells };
        let (part_diff, part_size) = diffs(parts);
        let mut part_of: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut owner = vec![usize::MAX; part_size.len()];
        for (&r, &c) in lab.iter().zip(parts) {
            if owner[c as usize] == usize::MAX {
                owner[c as usize] = r as usize;
                part_of[r as usize].push(c as usize);
            }
        }
        // a word's score: average fit over the parts minus how unevenly it fits them (kept: each region's best 60)
        let top: Vec<Vec<(f32, usize)>> = (0..k)
            .into_par_iter()
            .map(|r| {
                let total = size[r] as f32;
                let (mut m1, mut m2) = (vec![0f32; words.len()], vec![0f32; words.len()]);
                for &c in &part_of[r] {
                    let share = part_size[c] as f32 / total;
                    for (w, f) in fit(&part_diff[c * DIM..(c + 1) * DIM]).into_iter().enumerate() {
                        m1[w] += share * f;
                        m2[w] += share * f * f;
                    }
                }
                let mut scored: Vec<(f32, usize)> =
                    m1.iter().zip(&m2).map(|(a, b)| a - EVENNESS * (b - a * a).max(0.0).sqrt()).zip(0..).collect();
                let keep = scored.len().min(60);
                scored.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0));
                scored.truncate(keep);
                scored
            })
            .collect();
        // names of the regions each one sits in, and whether it's just the same photos as one of them
        let mut above: Vec<HashSet<String>> = vec![HashSet::new(); k];
        let mut same = vec![false; k];
        let mut first = vec![usize::MAX; k];
        for (i, &r) in lab.iter().enumerate() {
            if first[r as usize] == usize::MAX {
                first[r as usize] = i;
            }
        }
        for up in 0..level {
            let mut up_size = vec![0usize; names[up].len()];
            levels[up].iter().for_each(|&x| up_size[x as usize] += 1);
            for r in 0..k {
                let parent = levels[up][first[r]] as usize;
                if let Some(nm) = &names[up][parent] {
                    above[r].extend(forms(nm));
                }
                same[r] |= up_size[parent] == size[r];
            }
        }
        let big_enough = |r: usize| if level == 0 { size[r] as f32 >= MIN_SHARE * n as f32 } else { size[r] >= MIN_PHOTOS };
        let mut cands: Vec<(f32, usize, usize)> = Vec::new();
        for r in (0..k).filter(|&r| big_enough(r) && !same[r]) {
            cands.extend(top[r].iter().map(|&(score, w)| (score, r, w)));
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        // names are unique among regions with the same parent (the whole map, for the coarsest level)
        let group: Vec<usize> = (0..k).map(|r| if level == 0 { 0 } else { levels[level - 1][first[r]] as usize }).collect();
        let mut chosen: Vec<Option<String>> = vec![None; k];
        let mut used: HashSet<(usize, String)> = HashSet::new();
        for (_, r, w) in cands {
            let f = forms(&words[w]);
            if chosen[r].is_some() || f.iter().any(|x| used.contains(&(group[r], x.clone())) || above[r].contains(x)) {
                continue;
            }
            chosen[r] = Some(words[w].clone());
            used.extend(f.into_iter().map(|x| (group[r], x)));
        }
        names.push(chosen);
    }
    names
}
