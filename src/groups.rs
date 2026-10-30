//! A third way to lay out a map: sort the photos into groups by words. Each
//! photo joins the word its picture is most like (its cosine with each, from the map's own vectors); each group is an island
//! (shaped like its typical photo), the islands packed in rows biggest first. Inside an island a photo's place and size show how strongly it matches: the
//! strongest in the middle and biggest, smaller and smaller towards the rim. Nothing decides what is a good match (no score means the same for every word or model); every photo
//! is in its best group, and the weak ones are plain to see at the edges.

use crate::data::{Point, WORLD};

/// A group's island on the world: its word (by number in the list), how many photos it has, its top left corner, width and height.
pub struct Island {
    pub word: usize,
    pub photos: usize,
    pub min: Point,
    pub size: Point,
}

/// Ready-made groups for the Layout menu: a name and its words, as typed in the box. Across subjects, not just places.
pub const PRESETS: [(&str, &str); 14] = [
    ("Scenes", "beach, mountains, city, forest, desert, snow, lake, countryside"),
    ("Time of day", "sunrise, daytime, sunset, night"),
    ("Seasons", "spring, summer, autumn, winter"),
    ("Weather", "sunny, cloudy, rain, fog, snow, storm"),
    ("People", "portrait, group of people, crowd, no people"),
    ("Animals", "dog, cat, bird, horse, insect, fish, wild animal"),
    ("Food and drink", "fruit, vegetables, meat, fish dish, dessert, bread, drink"),
    ("Buildings", "church, castle, bridge, skyscraper, house, ruins, monument"),
    ("Getting around", "car, train, boat, airplane, bicycle, bus, walking"),
    ("Plants and nature", "flower, tree, leaves, grass, rock, water, sky"),
    ("Sport", "football, tennis, running, swimming, skiing, cycling, climbing"),
    ("Colours", "red, orange, yellow, green, blue, purple, black and white"),
    ("Kind of picture", "photograph, painting, drawing, map, diagram, text, screenshot"),
    ("Mood", "happy, calm, sad, dramatic, mysterious, romantic"),
];

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

/// The most typical photo of a group is drawn this many times the size of those at its rim ...
const BIG: f32 = 3.0;
/// ... the size falling off with the share of the group's photos placed before it: 1 + (BIG - 1)(1 - share)^FALL (with 5: 2.5x
/// at 5%, 2.2x at 10%, 1.7x at 20%, 1.3x at 30%, about 1x from half).
const FALL: i32 = 5;
/// A photo is drawn this much of its tile, so neighbours don't quite touch.
const DRAWN: f32 = 0.96;

/// The view: each photo's place and size (times `side`), the islands, and a photo's longer side on the world at size 1.
pub struct Placed {
    pub points: Vec<Point>,
    pub sizes: Vec<f32>,
    pub islands: Vec<Island>,
    pub side: f32,
}

/// Each photo's place and size, and the islands (biggest first; words no photo is most like have none). `cos`: per word, every
/// photo's cosine with it; `forms`: every photo's shape (its longer side 1).
pub fn place(cos: &[Vec<f32>], forms: &[(f32, f32)]) -> Placed {
    let n = cos.first().map_or(0, Vec::len);
    // each photo's group: the word it is most like (on the raw scores: less each word's average over all the photos was
    // measured, Wikivoyage captions as the key, and only moved the photos that fit no word from one word's group to another's)
    let group: Vec<usize> = (0..n).map(|i| (0..cos.len()).max_by(|&a, &b| cos[a][i].total_cmp(&cos[b][i])).unwrap_or(0)).collect();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); cos.len()];
    for (i, &g) in group.iter().enumerate() {
        members[g].push(i);
    }
    // inside each group: the photos best match first, from the middle out (in units of a rim photo's tile)
    let mut tiles: Vec<Vec<(f32, f32, f32)>> = vec![Vec::new(); cos.len()];
    let mut sizes = vec![(0.0f32, 0.0f32); cos.len()];
    for (g, m) in members.iter_mut().enumerate() {
        m.sort_by(|&a, &b| cos[g][b].total_cmp(&cos[g][a]));
        let shapes: Vec<(f32, f32)> = m.iter().map(|&i| forms[i]).collect();
        (tiles[g], sizes[g]) = rings(&shapes);
    }
    let mut order: Vec<usize> = (0..cos.len()).filter(|&g| !members[g].is_empty()).collect();
    order.sort_by_key(|&g| std::cmp::Reverse(members[g].len()));

    // islands in rows; the gaps leave room for the names above them
    let biggest = order.iter().map(|&g| sizes[g].0.max(sizes[g].1)).fold(1.0, f32::max);
    let gap = 0.15 * biggest;
    let area: f32 = order.iter().map(|&g| (sizes[g].0 + gap) * (sizes[g].1 + gap)).sum();
    let row_width = (area.sqrt() * 1.2).max(biggest);
    let mut islands = Vec::new();
    let (mut x, mut y, mut row_height, mut width) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &g in &order {
        let (w, h) = sizes[g];
        if x > 0.0 && x + w > row_width {
            (x, y, row_height) = (0.0, y + row_height + 1.5 * gap, 0.0);
        }
        islands.push(Island { word: g, photos: members[g].len(), min: Point { x, y }, size: Point { x: w, y: h } });
        x += w + gap;
        width = width.max(x - gap);
        row_height = row_height.max(h);
    }
    let height = y + row_height;

    // all of it scaled into the world, in the middle
    let scale = WORLD / width.max(height).max(1e-6);
    let offset = Point { x: (WORLD - width * scale) / 2.0, y: (WORLD - height * scale) / 2.0 };
    for island in &mut islands {
        island.min = Point { x: offset.x + island.min.x * scale, y: offset.y + island.min.y * scale };
        island.size = Point { x: island.size.x * scale, y: island.size.y * scale };
    }

    let mut points = vec![Point { x: 0.0, y: 0.0 }; n];
    let mut photo_sizes = vec![1.0; n];
    for island in &islands {
        let middle = Point { x: island.min.x + island.size.x / 2.0, y: island.min.y + island.size.y / 2.0 };
        for (&i, &(x, y, s)) in members[island.word].iter().zip(&tiles[island.word]) {
            points[i] = Point { x: middle.x + x * scale, y: middle.y + y * scale };
            photo_sizes[i] = s;
        }
    }
    Placed { points, sizes: photo_sizes, islands, side: scale * DRAWN }
}

/// Tiles for photos of these shapes (best first), from the middle out: one in the middle, then rings round it. A tile is shaped
/// like the group's typical photo (the median shape), its size that of the first photo in its ring; a ring's tiles are stretched
/// a little to fit exactly round the ones inside it, and a last ring that isn't full is spread evenly round. Each photo's centre
/// (from the middle) and size (its longer side, to fit its tile), and the width and height they fill.
fn rings(shapes: &[(f32, f32)]) -> (Vec<(f32, f32, f32)>, (f32, f32)) {
    let m = shapes.len();
    let mut out = Vec::with_capacity(m);
    if m == 0 {
        return (out, (0.0, 0.0));
    }
    let mut tall: Vec<f32> = shapes.iter().map(|&(w, h)| h / w.max(1e-6)).collect();
    tall.sort_by(f32::total_cmp);
    let r = tall[m / 2];
    let tile = if r <= 1.0 { (1.0, r) } else { (1.0 / r, 1.0) };
    let size = |placed: usize| 1.0 + (BIG - 1.0) * (1.0 - placed as f32 / m as f32).powi(FALL);
    // the photo's longer side that fits a w x h tile
    let fit = |k: usize, w: f32, h: f32| (w / shapes[k].0.max(1e-6)).min(h / shapes[k].1.max(1e-6));

    let (mut w, mut h) = (tile.0 * size(0), tile.1 * size(0));
    out.push((0.0, 0.0, fit(0, w, h)));
    while out.len() < m {
        let s = size(out.len());
        let (nx, ny) = (((w / (tile.0 * s)).round() as usize).max(1), ((h / (tile.1 * s)).round() as usize).max(1));
        let (tw, th) = (w / nx as f32, h / ny as f32); // exactly round the inside
        let (x0, y0) = (-w / 2.0 - tw / 2.0, -h / 2.0 - th / 2.0); // the top left tile's centre
        let col = |k: usize| x0 + k as f32 * tw;
        let row = |k: usize| y0 + k as f32 * th;
        // round the ring: the top left to right, down the right, the bottom right to left, up the left
        let mut ring: Vec<(f32, f32)> = (0..nx + 2).map(|k| (col(k), row(0))).collect();
        ring.extend((1..=ny).map(|k| (col(nx + 1), row(k))));
        ring.extend((0..nx + 2).rev().map(|k| (col(k), row(ny + 1))));
        ring.extend((1..=ny).rev().map(|k| (col(0), row(k))));
        let take = ring.len().min(m - out.len());
        for j in 0..take {
            let (x, y) = ring[j * ring.len() / take];
            let k = out.len();
            out.push((x, y, fit(k, tw, th)));
        }
        (w, h) = (w + 2.0 * tw, h + 2.0 * th);
    }
    (out, (w, h))
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
        let Placed { points, islands, sizes, .. } = place(&[beach, city, forest], &[(1.0, 0.67); 8]);
        assert_eq!(islands.iter().map(|i| (i.word, i.photos)).collect::<Vec<_>>(), vec![(0, 6), (1, 2)]);
        assert!(islands[0].size.x > islands[1].size.x);
        let inside = |p: Point, isl: &Island| p.x >= isl.min.x && p.x <= isl.min.x + isl.size.x && p.y >= isl.min.y && p.y <= isl.min.y + isl.size.y;
        for (i, p) in points.iter().enumerate() {
            assert!(inside(*p, &islands[if i < 6 { 0 } else { 1 }]), "photo {i} at {},{}", p.x, p.y);
        }
        let (a, b) = (&islands[0], &islands[1]);
        assert!(a.min.x + a.size.x <= b.min.x || b.min.x + b.size.x <= a.min.x || a.min.y + a.size.y <= b.min.y || b.min.y + b.size.y <= a.min.y);
        // the strongest beach photo is nearest the island's middle
        let middle = Point { x: a.min.x + a.size.x / 2.0, y: a.min.y + a.size.y / 2.0 };
        let d = |i: usize| (points[i].x - middle.x).powi(2) + (points[i].y - middle.y).powi(2);
        assert!((0..6).all(|i| d(3) <= d(i)));
        // and the biggest
        assert!((0..6).all(|i| sizes[3] >= sizes[i]) && sizes[3] > sizes[0]);
    }

    /// Photos in their ring tiles never overlap and stay inside the island, whatever their shapes; the ones in the middle are
    /// bigger than those at the rim.
    #[test]
    fn ring_photos_do_not_overlap() {
        for m in [1, 2, 9, 10, 37, 500] {
            let shapes: Vec<(f32, f32)> = (0..m).map(|k| if k % 5 == 0 { (0.67, 1.0) } else if k % 7 == 0 { (1.0, 0.3) } else { (1.0, 0.67) }).collect();
            let (tiles, (w, h)) = rings(&shapes);
            assert_eq!(tiles.len(), m);
            let rect = |k: usize| {
                let (x, y, s) = tiles[k];
                (x, y, shapes[k].0 * s / 2.0, shapes[k].1 * s / 2.0)
            };
            for a in 0..m {
                let (ax, ay, aw, ah) = rect(a);
                assert!(ax.abs() + aw <= w / 2.0 + 1e-3 && ay.abs() + ah <= h / 2.0 + 1e-3, "{m} photos: {a} outside");
                for b in a + 1..m {
                    let (bx, by, bw, bh) = rect(b);
                    assert!((ax - bx).abs() >= aw + bw - 1e-3 || (ay - by).abs() >= ah + bh - 1e-3, "{m} photos: {a} and {b} overlap");
                }
            }
            if m >= 37 {
                assert!(tiles[0].2 > tiles[m - 2].2.max(tiles[m - 1].2));
            }
        }
    }
}
