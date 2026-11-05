//! The matches of a search together, best first: a copy of each in rows beside the world (to the right of it), bigger than the
//! photos on the map, while the photos themselves stay where they are. They are seen and zoomed into like the rest of the map;
//! a click on a copy flies to its photo. A video's moments that match (its best frames, as many as the matches) can sit
//! right of them, the same size; a click on one plays it from there.

use crate::data::{Point, WORLD};

/// The block's left edge, and how wide and high it may be (world units).
pub const LEFT: f32 = WORLD * 1.04;
const WIDTH: f32 = WORLD * 0.3;
const HEIGHT: f32 = WORLD;
/// A cell is never bigger than this, so that a few matches are not enormous.
const MAX_CELL: f32 = WORLD * 0.05;
/// A copy's share of its cell (the rest is the gap between them).
pub const FILL: f32 = 0.9;

/// The copies' centres, best first in rows from the world's top, and a copy's longer side (world units).
pub fn place(count: usize) -> (Vec<Point>, f32) {
    if count == 0 {
        return (Vec::new(), 0.0);
    }
    // as many columns as make the block about WIDTH:HEIGHT
    let cols = ((count as f32 * WIDTH / HEIGHT).sqrt().round() as usize).clamp(1, count);
    let rows = count.div_ceil(cols);
    let cell = (WIDTH / cols as f32).min(HEIGHT / rows as f32).min(MAX_CELL);
    let points = (0..count).map(|k| Point { x: LEFT + ((k % cols) as f32 + 0.5) * cell, y: ((k / cols) as f32 + 0.5) * cell }).collect();
    (points, cell * FILL)
}

/// Copies of `side` (place's) right of `right` (the block before them: the matches), best first in rows from the world's top: as
/// wide as the matches may be, wider when that is not tall enough.
pub fn place_after(count: usize, side: f32, right: f32) -> Vec<Point> {
    if count == 0 {
        return Vec::new();
    }
    let cell = side / FILL;
    let most_rows = ((HEIGHT / cell).floor() as usize).max(1);
    let cols = ((WIDTH / cell).floor() as usize).max(count.div_ceil(most_rows)).clamp(1, count);
    let left = right + GAP;
    (0..count).map(|k| Point { x: left + ((k % cols) as f32 + 0.5) * cell, y: ((k / cols) as f32 + 0.5) * cell }).collect()
}

/// Between the matches and the moments.
const GAP: f32 = WORLD * 0.04;

#[cfg(test)]
mod tests {
    use super::*;

    /// Every copy is right of the world, inside the block, best first along the rows, and no two overlap.
    #[test]
    fn copies_sit_beside_the_world_in_rows() {
        for count in [1, 2, 7, 100, 1000, 5000] {
            let (points, side) = place(count);
            assert_eq!(points.len(), count);
            for p in &points {
                assert!(p.x - side / 2.0 > WORLD && p.x + side / 2.0 <= LEFT + WIDTH && p.y - side / 2.0 >= 0.0 && p.y + side / 2.0 <= HEIGHT, "{count}: {},{}", p.x, p.y);
            }
            for w in points.windows(2) {
                let (a, b) = (w[0], w[1]);
                assert!(b.y > a.y || (b.y == a.y && b.x - a.x >= side), "{count}: rows in order, apart");
            }
        }
    }

    /// The moments sit right of the matches, inside the world's height, best first along the rows, and no two overlap.
    #[test]
    fn moments_sit_right_of_the_matches() {
        for (matches, count) in [(1, 1), (5, 40), (100, 3), (2, 5000)] {
            let (points, side) = place(matches);
            let right = points.iter().map(|p| p.x).fold(LEFT, f32::max) + side / FILL / 2.0;
            let moments = place_after(count, side, right);
            assert_eq!(moments.len(), count);
            for p in &moments {
                assert!(p.x - side / 2.0 > right && p.y - side / 2.0 >= 0.0 && p.y + side / 2.0 <= HEIGHT, "{matches}, {count}: {},{}", p.x, p.y);
            }
            for w in moments.windows(2) {
                let (a, b) = (w[0], w[1]);
                assert!(b.y > a.y || (b.y == a.y && b.x - a.x >= side), "{matches}, {count}: rows in order, apart");
            }
        }
    }
}
