//! A third way to lay out a map: sort the photos into groups by words. Each photo joins the word its picture is most like (its
//! cosine with each, from the map's own vectors); each group is a square island, its area as many photos as it has, the islands
//! packed in rows biggest first. Inside an island a photo's place shows how strongly it matches: the strongest in the middle,
//! the weakest at the rim. Nothing decides what is a good match (no score means the same for every word or model); every photo
//! is in its best group, and the weak ones are plain to see at the edges.

use crate::data::{Point, WORLD};

/// A group's island on the world: its word (by number in the list), how many photos it has, its top left corner and side.
pub struct Island {
    pub word: usize,
    pub photos: usize,
    pub min: Point,
    pub side: f32,
}

/// The words in a box: split at commas and new lines, trimmed, each once (ignoring case).
pub fn words(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in text.split([',', '\n']).map(str::trim).filter(|w| !w.is_empty()) {
        if !out.iter().any(|o| o.eq_ignore_ascii_case(w)) {
            out.push(w.to_string());
        }
    }
    out
}

/// Each photo's place, and the islands (biggest first; words no photo is most like have none). `cos`: per word, every photo's
/// cosine with it.
pub fn place(cos: &[Vec<f32>]) -> (Vec<Point>, Vec<Island>) {
    let n = cos.first().map_or(0, Vec::len);
    // each photo's group: the word it is most like (on the raw scores: less each word's average over all the photos was
    // measured, Wikivoyage captions as the key, and only moved the photos that fit no word from one word's group to another's)
    let group: Vec<usize> = (0..n).map(|i| (0..cos.len()).max_by(|&a, &b| cos[a][i].total_cmp(&cos[b][i])).unwrap_or(0)).collect();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); cos.len()];
    for (i, &g) in group.iter().enumerate() {
        members[g].push(i);
    }
    let mut order: Vec<usize> = (0..cos.len()).filter(|&g| !members[g].is_empty()).collect();
    order.sort_by_key(|&g| std::cmp::Reverse(members[g].len()));

    // islands in rows, in units of one photo's share: side = sqrt(photos); the gaps leave room for the names above them
    let biggest = order.first().map_or(1.0, |&g| (members[g].len() as f32).sqrt());
    let gap = 0.15 * biggest;
    let area: f32 = order.iter().map(|&g| ((members[g].len() as f32).sqrt() + gap).powi(2)).sum();
    let row_width = (area.sqrt() * 1.2).max(biggest);
    let mut islands = Vec::new();
    let (mut x, mut y, mut row_height, mut width) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &g in &order {
        let side = (members[g].len() as f32).sqrt();
        if x > 0.0 && x + side > row_width {
            (x, y, row_height) = (0.0, y + row_height + 1.5 * gap, 0.0);
        }
        islands.push(Island { word: g, photos: members[g].len(), min: Point { x, y }, side });
        x += side + gap;
        width = width.max(x - gap);
        row_height = row_height.max(side);
    }
    let height = y + row_height;

    // all of it scaled into the world, in the middle
    let scale = WORLD / width.max(height).max(1e-6);
    let offset = Point { x: (WORLD - width * scale) / 2.0, y: (WORLD - height * scale) / 2.0 };
    for island in &mut islands {
        island.min = Point { x: offset.x + island.min.x * scale, y: offset.y + island.min.y * scale };
        island.side *= scale;
    }

    // inside each island: the photos best match first, on the cells of a grid nearest the middle first
    let mut points = vec![Point { x: 0.0, y: 0.0 }; n];
    for island in &islands {
        let mut m = members[island.word].clone();
        m.sort_by(|&a, &b| cos[island.word][b].total_cmp(&cos[island.word][a]));
        let g = (m.len() as f32).sqrt().ceil() as usize;
        let middle = (g as f32 - 1.0) / 2.0;
        let mut cells: Vec<(usize, usize)> = (0..g * g).map(|c| (c % g, c / g)).collect();
        let d2 = |&(x, y): &(usize, usize)| (x as f32 - middle).powi(2) + (y as f32 - middle).powi(2);
        cells.sort_by(|a, b| d2(a).total_cmp(&d2(b)));
        let cell = island.side / g as f32;
        for (&i, &(x, y)) in m.iter().zip(&cells) {
            points[i] = Point { x: island.min.x + (x as f32 + 0.5) * cell, y: island.min.y + (y as f32 + 0.5) * cell };
        }
    }
    (points, islands)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_split_at_commas_and_lines_once_each() {
        assert_eq!(words("beach, mountain\ncity,,  Beach \n"), vec!["beach", "mountain", "city"]);
    }

    /// Each photo lands in the island of the word it is most like, inside that island, and the islands don't overlap; the
    /// bigger group gets the bigger island; a word no photo chose gets none; the strongest match is in the middle.
    #[test]
    fn photos_go_to_their_word_s_island() {
        // photos 0..6 like "beach" best (photo 3 most of all), 6..8 "city", none "forest"
        let n = 8;
        let beach: Vec<f32> = (0..n).map(|i| if i == 3 { 0.4 } else if i < 6 { 0.3 } else { 0.1 }).collect();
        let city: Vec<f32> = (0..n).map(|i| if i < 6 { 0.1 } else { 0.3 }).collect();
        let forest = vec![0.0; n];
        let (points, islands) = place(&[beach, city, forest]);
        assert_eq!(islands.iter().map(|i| (i.word, i.photos)).collect::<Vec<_>>(), vec![(0, 6), (1, 2)]);
        assert!(islands[0].side > islands[1].side);
        let inside = |p: Point, isl: &Island| p.x >= isl.min.x && p.x <= isl.min.x + isl.side && p.y >= isl.min.y && p.y <= isl.min.y + isl.side;
        for (i, p) in points.iter().enumerate() {
            assert!(inside(*p, &islands[if i < 6 { 0 } else { 1 }]), "photo {i} at {},{}", p.x, p.y);
        }
        let (a, b) = (&islands[0], &islands[1]);
        assert!(a.min.x + a.side <= b.min.x || b.min.x + b.side <= a.min.x || a.min.y + a.side <= b.min.y || b.min.y + b.side <= a.min.y);
        // the strongest beach photo is nearest the island's middle
        let middle = Point { x: a.min.x + a.side / 2.0, y: a.min.y + a.side / 2.0 };
        let d = |i: usize| (points[i].x - middle.x).powi(2) + (points[i].y - middle.y).powi(2);
        assert!((0..6).all(|i| d(3) <= d(i)));
    }
}
