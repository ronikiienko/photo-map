//! A second way to lay out a map: two pairs of words, one across and one up and down. A photo's score on an axis is how much
//! more it is like the second word than the first (its cosine with each, from the map's own vectors). Two ways to turn scores
//! into places (`Spread`): by rank, which spreads the photos evenly over the world but shows only their order; or by the score
//! itself, which shows how far apart they are but bunches most of them in the middle, since most photos are neither clearly
//! one nor the other.

use crate::data::{Point, WORLD};

/// The words at the ends of the two axes.
#[derive(Clone, Default)]
pub struct Words {
    pub left: String,
    pub right: String,
    pub down: String,
    pub up: String,
}

impl Words {
    pub fn all(&self) -> [&str; 4] {
        [&self.left, &self.right, &self.down, &self.up]
    }
}

/// How scores become places along an axis.
#[derive(Clone, Copy, PartialEq, Default)]
pub enum Spread {
    /// By rank: every photo an equal share of the width, however close its score is to its neighbours'.
    #[default]
    Even,
    /// By the score itself, from the 1st to the 99th percentile (the few beyond sit at the edge, so that one odd photo
    /// doesn't squeeze all the others together).
    Score,
}

/// Each photo's place, from its cosines with the four words (in the order of `Words::all`).
pub fn place(cos: &[Vec<f32>; 4], spread: Spread) -> Vec<Point> {
    let along = |values: Vec<f32>| match spread {
        Spread::Even => ranks(&values),
        Spread::Score => scaled(&values),
    };
    let x = along(cos[1].iter().zip(&cos[0]).map(|(r, l)| r - l).collect());
    let y = along(cos[3].iter().zip(&cos[2]).map(|(u, d)| u - d).collect());
    // the screen's y grows downwards: the "up" end is at 0
    x.into_iter().zip(y).map(|(x, y)| Point { x: x * WORLD, y: (1.0 - y) * WORLD }).collect()
}

/// Each value's rank among all of them, as 0 ... 1 (the middle of its share, so nothing sits on the world's edge).
fn ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut out = vec![0.0; n];
    for (rank, i) in order.into_iter().enumerate() {
        out[i] = (rank as f32 + 0.5) / n as f32;
    }
    out
}

/// Each value as 0 ... 1 between the 1st and 99th percentile of all of them (clamped beyond).
fn scaled(values: &[f32]) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    let at = |q: f32| sorted[((sorted.len() - 1) as f32 * q).round() as usize];
    let (lo, hi) = (at(0.01), at(0.99));
    let span = (hi - lo).max(1e-6);
    values.iter().map(|v| ((v - lo) / span).clamp(0.0, 1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The photo most like the right word is furthest right, the one most like the up word highest (smallest y); the photos
    /// spread evenly however bunched their scores are.
    #[test]
    fn photos_go_where_their_words_are() {
        // photo 0: night, city; photo 1: day, nature; photo 2: a little of each, bunched close to photo 1's scores
        let left = vec![0.10, 0.30, 0.29]; // day
        let right = vec![0.30, 0.10, 0.11]; // night
        let down = vec![0.10, 0.30, 0.29]; // nature
        let up = vec![0.30, 0.10, 0.11]; // city
        let p = place(&[left, right, down, up], Spread::Even);
        assert!(p[0].x > p[2].x && p[2].x > p[1].x);
        assert!(p[0].y < p[2].y && p[2].y < p[1].y);
        for (x, want) in [(p[1].x, WORLD / 6.0), (p[2].x, WORLD / 2.0), (p[0].x, WORLD * 5.0 / 6.0)] {
            assert!((x - want).abs() < 0.01, "evenly spread: {x} vs {want}");
        }
    }

    /// By score, distances follow the scores: two photos with nearly the same score sit nearly on top of each other.
    #[test]
    fn by_score_keeps_the_distances() {
        let n = 201;
        let zero = vec![0.0; n];
        let mut score: Vec<f32> = (0..n).map(|i| i as f32 / 200.0).collect(); // 0 ... 1, evenly
        score[100] = 0.501; // photo 100 next to photo 101 at 0.505
        let p = place(&[zero.clone(), score, zero.clone(), zero], Spread::Score);
        let close = p[101].x - p[100].x;
        let normal = p[150].x - p[149].x;
        assert!(close < normal, "{close} should be less than {normal}");
        assert_eq!(p[0].x, 0.0, "below the 1st percentile: at the edge");
        assert_eq!(p[200].x, WORLD, "above the 99th: at the other edge");
    }
}
