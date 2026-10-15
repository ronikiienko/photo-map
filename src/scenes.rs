//! Scenes of a video, found from the vectors of frames taken at a steady interval.
//!
//! A scene runs until a frame stops looking like the scene so far (its vector is not close enough to the scene's average), so a slow
//! change inside one scene does not split it and a cut does. Scenes too short to matter are joined to the neighbour they resemble,
//! and a video with too many is cut down by joining the most alike neighbours, until the count is the one wanted.
//!
//! Frames are compared after the video's own average vector is taken away from all of them. Every vector a model makes for pictures
//! shares a large part with every other (two unrelated pictures are often 0.7 alike), and the average of many frames is mostly that
//! shared part, so without this a new frame always looks like the scene so far and a film full of cuts is one scene.

use std::ops::Range;

pub struct Settings {
    /// A frame whose cosine to the scene's average starts a new scene when it is below this: the cosine of what is left of them
    /// once the video's average vector is taken away (1: the same, 0: nothing in common, below 0: opposite).
    pub threshold: f32,
    /// A scene shorter than this many seconds is joined to a neighbour.
    pub min_seconds: f64,
    /// At most this many scenes (the most alike neighbours are joined first); `None`: as many as there are.
    pub at_most: Option<usize>,
}

impl Default for Settings {
    fn default() -> Self {
        Self { threshold: 0.5, min_seconds: 3.0, at_most: None }
    }
}

pub struct Scene {
    pub start: f64,
    pub end: f64,
    /// The frames of the scene: indices into the list given to `split`.
    pub frames: Range<usize>,
    /// The scene's average vector, unit length.
    pub vector: Vec<f32>,
    /// The frame most like the average: the picture that stands for the scene.
    pub best: usize,
}

struct Run {
    frames: Range<usize>,
    /// The frames' vectors added up, with the video's average taken away (what scenes are compared by).
    sum: Vec<f32>,
    /// The frames' vectors added up as they are (what the scene's vector is made of).
    raw: Vec<f32>,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = dot(v, v).sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

/// The cosine of two sums (their lengths do not matter). A sum with nothing left in it (a frame that is the video's average) is
/// like everything.
fn alike(a: &[f32], b: &[f32]) -> f32 {
    let (na, nb) = (dot(a, a).sqrt(), dot(b, b).sqrt());
    if na < 1e-4 || nb < 1e-4 { 1.0 } else { dot(a, b) / (na * nb) }
}

fn join(runs: &mut Vec<Run>, i: usize) {
    let next = runs.remove(i + 1);
    let run = &mut runs[i];
    run.frames.end = next.frames.end;
    run.sum.iter_mut().zip(&next.sum).for_each(|(a, b)| *a += b);
    run.raw.iter_mut().zip(&next.raw).for_each(|(a, b)| *a += b);
}

/// The scenes of a video `seconds` long, from `(time, vector)` of its frames in order (vectors of unit length).
pub fn split(frames: &[(f64, Vec<f32>)], seconds: f64, settings: &Settings) -> Vec<Scene> {
    let Some(first) = frames.first() else { return Vec::new() };
    // the frames with the video's average vector taken away
    let mut mean = vec![0f32; first.1.len()];
    for (_, v) in frames {
        mean.iter_mut().zip(v).for_each(|(m, x)| *m += x / frames.len() as f32);
    }
    let centred: Vec<Vec<f32>> = frames.iter().map(|(_, v)| v.iter().zip(&mean).map(|(x, m)| x - m).collect()).collect();
    let mut runs = vec![Run { frames: 0..1, sum: centred[0].clone(), raw: first.1.clone() }];
    for (i, (_, v)) in frames.iter().enumerate().skip(1) {
        let run = runs.last_mut().unwrap();
        if alike(&centred[i], &run.sum) < settings.threshold {
            runs.push(Run { frames: i..i + 1, sum: centred[i].clone(), raw: v.clone() });
        } else {
            run.frames.end = i + 1;
            run.sum.iter_mut().zip(&centred[i]).for_each(|(a, b)| *a += b);
            run.raw.iter_mut().zip(v).for_each(|(a, b)| *a += b);
        }
    }
    let start = |r: &Run| frames[r.frames.start].0;
    let length = |runs: &[Run], i: usize| (runs.get(i + 1).map_or(seconds, start) - start(&runs[i])).max(0.0);
    // too short: joined to the neighbour it resembles more, the shortest first
    while runs.len() > 1 {
        let Some(i) = (0..runs.len()).filter(|i| length(&runs, *i) < settings.min_seconds).min_by(|a, b| length(&runs, *a).total_cmp(&length(&runs, *b))) else { break };
        // a blink between two runs that are alike joins them all (its own, unlike frame would pull the joined average its way)
        if i > 0 && i + 1 < runs.len() && alike(&runs[i - 1].sum, &runs[i + 1].sum) >= settings.threshold {
            join(&mut runs, i - 1);
            join(&mut runs, i - 1);
            continue;
        }
        let with_previous = i > 0 && (i + 1 == runs.len() || alike(&runs[i - 1].sum, &runs[i].sum) >= alike(&runs[i].sum, &runs[i + 1].sum));
        join(&mut runs, if with_previous { i - 1 } else { i });
    }
    // what a blink in the middle had split is one scene again
    while let Some(i) = (0..runs.len().saturating_sub(1)).find(|i| alike(&runs[*i].sum, &runs[*i + 1].sum) >= settings.threshold) {
        join(&mut runs, i);
    }
    // too many: the most alike neighbours are joined
    while settings.at_most.is_some_and(|m| runs.len() > m.max(1)) {
        let i = (0..runs.len() - 1).max_by(|a, b| alike(&runs[*a].sum, &runs[*a + 1].sum).total_cmp(&alike(&runs[*b].sum, &runs[*b + 1].sum))).unwrap();
        join(&mut runs, i);
    }
    let starts: Vec<f64> = runs.iter().map(start).collect();
    runs.iter()
        .enumerate()
        .map(|(i, r)| {
            let vector = unit(&r.raw);
            let best = r.frames.clone().max_by(|a, b| dot(&frames[*a].1, &vector).total_cmp(&dot(&frames[*b].1, &vector))).unwrap();
            Scene { start: starts[i], end: starts.get(i + 1).copied().unwrap_or(seconds), frames: r.frames.clone(), vector, best }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit vector in 8 dimensions pointing at `axis`, nudged by `wobble` along the next axis.
    fn at(axis: usize, wobble: f32) -> Vec<f32> {
        let mut v = vec![0.0; 8];
        v[axis] = 1.0;
        v[(axis + 1) % 8] = wobble;
        unit(&v)
    }

    fn video(axes: &[usize]) -> Vec<(f64, Vec<f32>)> {
        axes.iter().enumerate().map(|(i, a)| (i as f64 * 2.0, at(*a, 0.05 * (i % 3) as f32))).collect()
    }

    #[test]
    fn a_cut_starts_a_scene_and_a_wobble_does_not() {
        let frames = video(&[0, 0, 0, 0, 3, 3, 3, 3, 0, 0, 0, 0]);
        let scenes = split(&frames, 24.0, &Settings::default());
        let spans: Vec<_> = scenes.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(spans, vec![(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)]);
        assert_eq!(scenes[1].frames, 4..8);
    }

    #[test]
    fn a_blink_is_joined_to_the_scene_it_interrupts() {
        let frames = video(&[0, 0, 0, 0, 5, 0, 0, 0, 0, 0]);
        let scenes = split(&frames, 20.0, &Settings { min_seconds: 3.0, ..Settings::default() });
        assert_eq!(scenes.len(), 1);
        assert_eq!((scenes[0].start, scenes[0].end), (0.0, 20.0));
    }

    #[test]
    fn too_many_scenes_are_cut_down_by_joining_the_most_alike() {
        // 0 and 1 are close (a nudge apart), 4 and 6 far from them and from each other
        let mut frames = video(&[0, 0, 0, 0, 4, 4, 4, 4, 6, 6, 6, 6]);
        frames.iter_mut().skip(4).take(4).for_each(|f| f.1 = at(4, 0.0));
        let scenes = split(&frames, 24.0, &Settings { at_most: Some(2), ..Settings::default() });
        assert_eq!(scenes.len(), 2);
        assert_eq!(scenes[0].frames.end, scenes[1].frames.start);
        assert_eq!(scenes.last().unwrap().end, 24.0);
    }

    #[test]
    fn the_picture_standing_for_a_scene_is_the_frame_nearest_its_average() {
        let frames = vec![(0.0, at(0, 0.6)), (2.0, at(0, 0.0)), (4.0, at(0, -0.6))];
        let scenes = split(&frames, 6.0, &Settings { threshold: -2.0, min_seconds: 0.0, at_most: None }); // never cut
        assert_eq!(scenes.len(), 1);
        assert_eq!(scenes[0].best, 1);
        assert!((dot(&scenes[0].vector, &scenes[0].vector) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn no_frames_no_scenes_and_one_frame_one_scene() {
        assert!(split(&[], 10.0, &Settings::default()).is_empty());
        let scenes = split(&[(0.0, at(1, 0.0))], 1.0, &Settings::default());
        assert_eq!(scenes.len(), 1);
        assert_eq!((scenes[0].start, scenes[0].end), (0.0, 1.0));
    }

    /// The scenes of PHOTO_MAP_VIDEO found with SigLIP 256, from a frame every PHOTO_MAP_EVERY seconds (2), printed; the picture standing for each is
    /// saved in .cache/test/scenes/. PHOTO_MAP_THRESHOLD sets the cosine. cargo test --release real_video_scenes -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_video_scenes() {
        let env = |k: &str| std::env::var(k).ok();
        let path = std::path::PathBuf::from(env("PHOTO_MAP_VIDEO").expect("PHOTO_MAP_VIDEO"));
        let every: f64 = env("PHOTO_MAP_EVERY").and_then(|v| v.parse().ok()).unwrap_or(2.0);
        let settings = Settings { threshold: env("PHOTO_MAP_THRESHOLD").and_then(|v| v.parse().ok()).unwrap_or(Settings::default().threshold), ..Settings::default() };
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/test/scenes");
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(&out).unwrap();

        let id = match env("PHOTO_MAP_MODEL").as_deref() {
            Some("gemma") => crate::models::GEMMA,
            Some("s512") => crate::models::SIGLIP_BASE_512,
            _ => crate::models::SIGLIP_BASE_256,
        };
        let hub = crate::models::test_hub_with(id);
        assert_eq!(hub.wait_for(id, &|| false), Ok(true));
        let seconds = crate::video::info(&path).unwrap().seconds;
        let started = std::time::Instant::now();
        let mut pictures = Vec::new();
        let mut vectors = Vec::new();
        crate::video::frames(&path, every, |f| {
            vectors.push((f.seconds, hub.with(id, |m| m.embed_image(&f.image)).unwrap().unwrap()));
            pictures.push(f.image);
            true
        })
        .unwrap();
        eprintln!("{} frames embedded in {:.1} s", vectors.len(), started.elapsed().as_secs_f32());
        if let Some(name) = env("PHOTO_MAP_DUMP") {
            // times then vectors (f32), to look at the numbers elsewhere
            let bytes: Vec<u8> = vectors.iter().flat_map(|(t, v)| std::iter::once(*t as f32).chain(v.iter().copied()).flat_map(f32::to_le_bytes).collect::<Vec<u8>>()).collect();
            std::fs::write(out.parent().unwrap().join(format!("vectors_{name}.f32")), bytes).unwrap();
        }
        let mut sims: Vec<f32> = vectors.windows(2).map(|w| dot(&w[0].1, &w[1].1)).collect();
        sims.sort_by(f32::total_cmp);
        let q = |f: f32| sims[((sims.len() - 1) as f32 * f) as usize];
        eprintln!("cosine between neighbouring frames: min {:.3}, 5% {:.3}, 25% {:.3}, median {:.3}, 75% {:.3}", sims[0], q(0.05), q(0.25), q(0.5), q(0.75));
        for t in [0.0f32, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8] {
            eprintln!("  threshold {t}: {} scenes", split(&vectors, seconds, &Settings { threshold: t, ..Settings::default() }).len());
        }
        let mut sims: Vec<f32> = vectors.windows(2).map(|w| dot(&w[0].1, &w[1].1)).collect();
        sims.sort_by(f32::total_cmp);
        let q = |f: f32| sims[((sims.len() - 1) as f32 * f) as usize];
        eprintln!("cosine between neighbouring frames: min {:.3}, 5% {:.3}, 25% {:.3}, median {:.3}, 75% {:.3}", sims[0], q(0.05), q(0.25), q(0.5), q(0.75));
        for t in [0.0f32, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8] {
            eprintln!("  threshold {t}: {} scenes", split(&vectors, seconds, &Settings { threshold: t, ..Settings::default() }).len());
        }
        let scenes = split(&vectors, seconds, &settings);
        eprintln!("{} scenes in {seconds:.0} s", scenes.len());
        for (n, s) in scenes.iter().enumerate() {
            eprintln!("  {n:2}: {:6.1} - {:6.1} s, {} frames, picture from {:.1} s", s.start, s.end, s.frames.len(), vectors[s.best].0);
            image::DynamicImage::ImageRgba8(pictures[s.best].clone()).thumbnail(640, 640).to_rgb8().save(out.join(format!("{n:03}.jpg"))).unwrap();
        }
    }
}
