//! Naming regions: a plain-English label for each region of a map (what src/name_regions.py did). The label vocabulary (vocab/*.txt,
//! about 4,500 phrases) is embedded with the same model as the photos (and kept per model: another model's vectors are no use here); a region's average vector, less the whole map's average, is
//! compared with every phrase's vector, and the best match - with a small bonus for common phrases - is the region's name.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// How much a phrase's commonness (log10: 1000 for the most basic ... 0.1 for the rarest) adds to its score.
const COMMON: f32 = 0.01;
/// Regions of fewer photos stay unnamed, and so do levels of regions of about 10 photos or fewer (the finest).
const MIN_PHOTOS: usize = 10;
const FINEST: usize = 6;
const ROUNDS: [&str; 6] = ["1", "1b", "2", "3", "4", "5"];

#[derive(Serialize, Deserialize, Clone)]
pub struct Phrase {
    pub name: String,
    /// 1000 / 100 / 10 / 1 / 0.1 for rounds 1 (and 1b) / 2 / 3 / 4 / 5.
    pub common: f32,
}

/// Every phrase once (its earliest round).
pub fn phrases(root: &Path) -> Result<Vec<Phrase>, String> {
    let (mut out, mut seen) = (Vec::new(), std::collections::HashSet::new());
    for round in ROUNDS {
        let file = root.join("vocab").join(format!("curated2_round{round}.txt"));
        let text = std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let rank: i32 = round.trim_end_matches('b').parse().unwrap_or(1);
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            if seen.insert(line.to_lowercase()) {
                out.push(Phrase { name: line.to_string(), common: 10f32.powi(4 - rank) });
            }
        }
    }
    Ok(out)
}

/// A key that changes when the vocabulary does (so cached vectors are made again).
fn key(phrases: &[Phrase]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in phrases {
        for b in p.name.bytes().chain([0]) {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// The file name part for a model's cache: its id with the slashes made harmless.
pub fn slug(model: &str) -> String {
    model.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect()
}

/// The phrases' vectors (n x dim, row by row), from the cache in .cache/vocab/ or made now with `embed` (told how far it is). The
/// cache is per model: `model` is the id of the one `embed` is.
pub fn vectors(root: &Path, model: &str, dim: usize, phrases: &[Phrase], embed: &mut dyn FnMut(&str) -> Result<Vec<f32>, String>, progress: &dyn Fn(usize, usize), paused: &dyn Fn() -> bool) -> Result<Option<Vec<f32>>, String> {
    let dir = root.join(".cache/vocab");
    let name = slug(model);
    let (bin, stamp) = (dir.join(format!("curated2.{name}.f16")), dir.join(format!("curated2.{name}.key")));
    // the cache from before models could be chosen was Gemma's
    if model == crate::models::GEMMA && !bin.exists() {
        let _ = std::fs::rename(dir.join("curated2.f16"), &bin);
        let _ = std::fs::rename(dir.join("curated2.key"), &stamp);
    }
    let want = key(phrases).to_string();
    if std::fs::read_to_string(&stamp).ok().as_deref() == Some(want.as_str()) {
        if let Ok(raw) = std::fs::read(&bin) {
            if raw.len() == phrases.len() * dim * 2 {
                return Ok(Some(raw.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect()));
            }
        }
    }
    let mut all = Vec::with_capacity(phrases.len() * dim);
    for (i, p) in phrases.iter().enumerate() {
        if paused() {
            return Ok(None);
        }
        let v = embed(&p.name)?;
        if v.len() != dim {
            return Err(format!("the model made {} numbers for a phrase, expected {dim}", v.len()));
        }
        all.extend(v);
        progress(i + 1, phrases.len());
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let bytes: Vec<u8> = all.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect();
    crate::build::write_atomic(&bin, &bytes).map_err(|e| e.to_string())?;
    crate::build::write_atomic(&stamp, want.as_bytes()).map_err(|e| e.to_string())?;
    Ok(Some(all))
}

#[derive(Serialize)]
pub struct Label {
    pub name: String,
    pub x: f32,
    pub y: f32,
    pub spread: f32,
    pub photos: usize,
}

fn unit(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter_mut().for_each(|x| *x /= norm);
}

/// Per level, the named regions (biggest first). `vectors`: n x `dim`; `xy`: x, y per photo; `levels`: a region number per photo per level.
pub fn name_levels(dim: usize, vectors: &[f32], xy: &[f32], levels: &[Vec<u32>], phrases: &[Phrase], phrase_vectors: &[f32]) -> Vec<Vec<Label>> {
    let n = xy.len() / 2;
    // the phrases, centred on their own average so generic ones don't win everywhere, unit length
    let mut v = phrase_vectors.to_vec();
    let mean: Vec<f32> = (0..dim).map(|d| v.chunks_exact(dim).map(|r| r[d]).sum::<f32>() / phrases.len() as f32).collect();
    for row in v.chunks_exact_mut(dim) {
        row.iter_mut().zip(&mean).for_each(|(x, m)| *x -= m);
        unit(row);
    }
    let bonus: Vec<f32> = phrases.iter().map(|p| COMMON * p.common.log10()).collect();
    // the map's average photo
    let mut map_mean = vec![0f32; dim];
    for row in vectors.chunks_exact(dim) {
        map_mean.iter_mut().zip(row).for_each(|(m, x)| *m += x / n as f32);
    }

    levels
        .iter()
        .enumerate()
        .map(|(li, labels)| {
            let regions = labels.iter().max().map_or(0, |m| *m as usize + 1);
            let mut members: Vec<Vec<u32>> = vec![Vec::new(); regions];
            for (i, &l) in labels.iter().enumerate() {
                members[l as usize].push(i as u32);
            }
            let mut out: Vec<Label> = if li >= FINEST {
                Vec::new()
            } else {
                members
                    .par_iter()
                    .filter(|m| m.len() >= MIN_PHOTOS)
                    .map(|m| {
                        let mut region = vec![0f32; dim];
                        for &i in m {
                            region.iter_mut().zip(&vectors[i as usize * dim..(i as usize + 1) * dim]).for_each(|(r, x)| *r += x / m.len() as f32);
                        }
                        region.iter_mut().zip(&map_mean).for_each(|(r, a)| *r -= a);
                        unit(&mut region);
                        let best = (0..phrases.len())
                            .map(|w| (v[w * dim..(w + 1) * dim].iter().zip(&region).map(|(a, b)| a * b).sum::<f32>() + bonus[w], w))
                            .max_by(|a, b| a.0.total_cmp(&b.0))
                            .map_or(0, |b| b.1);
                        // the label sits at the photo nearest the region's centre
                        let c = m.iter().fold([0f64; 2], |c, &i| [c[0] + xy[i as usize * 2] as f64, c[1] + xy[i as usize * 2 + 1] as f64]);
                        let c = [(c[0] / m.len() as f64) as f32, (c[1] / m.len() as f64) as f32];
                        let d2 = |i: u32| (xy[i as usize * 2] - c[0]).powi(2) + (xy[i as usize * 2 + 1] - c[1]).powi(2);
                        let nearest = m.iter().copied().min_by(|&a, &b| d2(a).total_cmp(&d2(b))).unwrap();
                        let spread = (m.iter().map(|&i| d2(i)).sum::<f32>() / m.len() as f32).sqrt();
                        Label { name: phrases[best].name.clone(), x: xy[nearest as usize * 2], y: xy[nearest as usize * 2 + 1], spread, photos: m.len() }
                    })
                    .collect()
            };
            out.sort_by(|a, b| b.photos.cmp(&a.photos));
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary is read, every phrase once, with its commonness.
    #[test]
    fn reads_the_vocabulary() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let list = phrases(root).unwrap();
        assert_eq!(list.len(), 3385, "same as the Python list (data/vocab/curated2.json)");
        assert_eq!(list.iter().map(|p| p.name.to_lowercase()).collect::<std::collections::HashSet<_>>().len(), list.len());
        let common = |name: &str| list.iter().find(|p| p.name == name).map(|p| p.common);
        assert_eq!(common("sky"), Some(1000.0)); // round 1b counts as round 1
        assert!(list.iter().any(|p| p.common < 0.5)); // round 5
    }

    /// The vocabulary's vectors are kept per model: another model gets its own, made with its own `embed`, and the first keeps its.
    #[test]
    fn the_vocabulary_is_kept_per_model() {
        let root = std::env::temp_dir().join(format!("photo-map-vocab-test-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let phrases = vec![Phrase { name: "sky".into(), common: 1000.0 }, Phrase { name: "sea".into(), common: 1000.0 }];
        let make = |model: &str, dim: usize, value: f32| {
            let mut calls = 0;
            let mut embed = |_: &str| {
                calls += 1;
                Ok(vec![value; dim])
            };
            let v = vectors(&root, model, dim, &phrases, &mut embed, &|_, _| {}, &|| false).unwrap().unwrap();
            (v, calls)
        };
        let (a, calls) = make("model/a", 4, 0.5);
        assert_eq!((a.len(), calls), (8, 2), "made now");
        let (a_again, calls) = make("model/a", 4, 9.0);
        assert_eq!((a_again, calls), (a.clone(), 0), "kept: not made again");
        let (b, calls) = make("model/b", 6, 0.25);
        assert_eq!((b.len(), calls), (12, 2), "another model makes its own, whatever size");
        assert_eq!(make("model/a", 4, 9.0).0, a, "and the first one's are still there");
        std::fs::remove_dir_all(&root).unwrap();
    }
}
