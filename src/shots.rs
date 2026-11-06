//! What the UI tests that draw pictures share: where the pictures go, and a made-up app folder (maps with covers) for the Maps page.
//! The pictures are for looking at, not compared: `cargo test --bin photo-map shots -- --ignored` draws every screen to .local/shots/.

use std::path::{Path, PathBuf};

/// The tests that draw take turns: side by side, the map's pages and the others' textures don't fit a 4 GB card.
pub fn gpu_turn() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// Save a drawn frame as .local/shots/<name>.png.
pub fn save(image: image::RgbaImage, name: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/shots");
    std::fs::create_dir_all(&dir).unwrap();
    image.save(dir.join(format!("{name}.png"))).unwrap();
}

/// A harness the size of a window, the app's look applied, its state `s`, drawn by `draw` over the map's dark.
pub fn harness<S>(size: egui::Vec2, s: S, mut draw: impl FnMut(&egui::Context, &mut S) + 'static) -> egui_kittest::Harness<'static, S> {
    egui_kittest::Harness::builder().with_size(size).wgpu().build_ui_state(
        move |ui, s: &mut S| {
            let ctx = ui.ctx().clone();
            crate::theme::apply(&ctx);
            ui.painter().rect_filled(ctx.content_rect(), 0.0, crate::theme::BACKGROUND);
            draw(&ctx, s);
        },
        s,
    )
}

/// A made-up app folder, .local/ui-fixture/<name> (one a test: tests run side by side), with these maps in it (name, photos, videos among them, created): each a manifest and a cover (a gradient
/// of its own colour). Made again every time, so it is what the test says.
pub fn app_folder(name: &str, maps: &[(&str, usize, usize, &str)]) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/ui-fixture").join(name);
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("maps");
    std::fs::create_dir_all(&dir).unwrap();
    for (k, (name, photos, videos, created)) in maps.iter().enumerate() {
        let map = dir.join(format!("{}.photomap", name.to_lowercase().replace(' ', "-")));
        std::fs::create_dir_all(&map).unwrap();
        let manifest = serde_json::json!({ "name": name, "photos": photos, "videos": videos, "created": created });
        std::fs::write(map.join("manifest.json"), manifest.to_string()).unwrap();
        let tint = [(90u8, 140u8, 200u8), (200, 140, 90), (110, 180, 120), (170, 110, 190)][k % 4];
        let cover = image::RgbaImage::from_fn(192, 192, |x, y| {
            let f = |c: u8, t: u32| (c as u32 * (128 + t) / 320) as u8;
            image::Rgba([f(tint.0, x), f(tint.1, y), f(tint.2, (x + y) / 2), 255])
        });
        cover.save(map.join("cover.png")).unwrap();
        if *videos > 0 {
            // the files a map with videos has, at about their real sizes (empty: sparse files, taking no disk)
            let sized = |file: &str, bytes: u64| {
                let path = map.join(file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                std::fs::File::create(path).unwrap().set_len(bytes).unwrap();
            };
            sized("pages.bin", *photos as u64 * 2_300_000 / 1_000);
            sized("look.bin", *photos as u64 * 1536);
            sized("frames/0/poster.jpg", *videos as u64 * 180_000);
            sized("layout.bin", *photos as u64 * 16);
        }
    }
    root
}

/// A made-up map in `maps`: `n` photos (small pictures of their own colour, in clusters, one cluster a colour), with names for
/// the clusters and cameras and dates for filters, baked as the app would (bake.rs). No vectors: it can't be searched by words.
pub fn map_bundle(maps: &Path, name: &str, n: usize) -> PathBuf {
    let bundle = maps.join(format!("{name}.photomap"));
    std::fs::create_dir_all(bundle.join("photos")).unwrap();
    let clusters = [("beach", (230u8, 200u8, 120u8)), ("forest", (60, 140, 70)), ("city at night", (60, 70, 140)), ("snow", (225, 230, 240)), ("sunset", (230, 110, 60)), ("documents", (200, 200, 190))];
    let centres = [(2800.0, 3000.0), (6500.0, 2600.0), (4600.0, 5200.0), (2600.0, 7000.0), (7200.0, 6600.0), (5000.0, 8200.0)];
    // the same photos every run: a fixed little random walk
    let mut seed = 12345u64;
    let mut rand = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / (1u64 << 31) as f32
    };
    let mut items = Vec::new();
    let mut files = Vec::new();
    let mut points: Vec<f32> = Vec::new();
    for i in 0..n {
        let c = i % clusters.len();
        let (label, (r, g, b)) = clusters[c];
        let (w, h) = if rand() < 0.3 { (72, 96) } else { (96, 72) };
        let shade = 0.6 + 0.5 * rand();
        let img = image::RgbaImage::from_fn(w, h, |x, y| {
            let f = |v: u8, t: f32| ((v as f32 * shade * t).min(255.0)) as u8;
            let t = 0.75 + 0.25 * ((x as f32 / w as f32) + (y as f32 / h as f32)) / 2.0;
            image::Rgba([f(r, t), f(g, t), f(b, t), 255])
        });
        let (a, d) = (rand() * std::f32::consts::TAU, 900.0 * rand().sqrt());
        points.extend([centres[c].0 + d * a.cos(), centres[c].1 + d * a.sin()]);
        let camera = ["Google Pixel 3", "Xiaomi Redmi Note 7", "Canon EOS 550D"][i % 3];
        let taken = format!("{}-{:02}-{:02} 12:00", 2016 + i % 5, 1 + i % 12, 1 + i % 28);
        let facts = Some(Box::new(crate::data::Facts { camera: camera.into(), taken, ..Default::default() }));
        // a video (no frames): its poster in the map, at the size a poster of it has
        let item = if i % 7 == 0 || i == 200 {
            let (pw, ph) = crate::data::poster_size(w * 40, h * 40);
            let poster = crate::data::poster(&bundle, i);
            std::fs::create_dir_all(poster.parent().unwrap()).unwrap();
            image::DynamicImage::ImageRgba8(image::imageops::resize(&img, pw, ph, image::imageops::FilterType::Triangle)).to_rgb8().save(&poster).unwrap();
            files.push(poster);
            let clip = crate::data::Clip { duration: 5.0 + (i * 37 % 300) as f32, ..Default::default() };
            crate::data::Item { source: format!("/videos/{label} {i}.mp4"), width: w * 40, height: h * 40, facts, video: Some(clip) }
        } else {
            let file = bundle.join(format!("photos/{label} {i}.png"));
            img.save(&file).unwrap();
            let item = crate::data::Item { source: file.to_string_lossy().into_owned(), width: w, height: h, facts, video: None };
            files.push(file);
            item
        };
        items.push(item);
    }
    let videos = items.iter().filter(|i| i.video.is_some()).count();
    let look = crate::data::Vectors { model: crate::models::DEFAULT.into(), dim: 768 };
    // spread as a build does (data::spread)
    let side = crate::median_neighbour_distance(bytemuck::cast_slice(&points));
    let points = crate::data::spread(bytemuck::cast_slice(&points), side);
    let manifest = crate::data::Manifest { name: name.into(), created: "2026-11-03 12:00".into(), photos: n, videos, photo_side: side, look };
    std::fs::write(bundle.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
    std::fs::write(bundle.join("items.json"), serde_json::to_vec(&items).unwrap()).unwrap();
    std::fs::write(bundle.join("layout.bin"), bytemuck::cast_slice::<crate::data::Point, u8>(&points)).unwrap();
    // two levels of names: the clusters, and halves of them
    let label = |name: &str, (x, y): (f32, f32), spread: f32, photos: usize| serde_json::json!({ "name": name, "x": x, "y": y, "spread": spread, "photos": photos });
    let coarse: Vec<_> = clusters.iter().zip(centres).map(|((name, _), at)| label(name, at, 900.0, n / 6)).collect();
    let fine: Vec<_> = clusters.iter().zip(centres).flat_map(|((name, _), (x, y))| [label(&format!("{name}, left"), (x - 400.0, y), 350.0, n / 12), label(&format!("{name}, right"), (x + 400.0, y), 350.0, n / 12)]).collect();
    std::fs::write(bundle.join("regions.json"), serde_json::json!({ "levels": [coarse, fine] }).to_string()).unwrap();
    crate::bake::bake(&bundle, &files, &crate::data::Progress::default()).unwrap();
    bundle
}

#[test]
#[ignore = "needs a GPU; writes pictures to look at"]
fn map_shots() {
    let _turn = gpu_turn();
    let root = app_folder("map", &[]);
    map_bundle(&root.join("maps"), "made-up", 600);
    let script = "
        open made-up
        shot world-1-whole
        # zoom in on a photo of the middle cluster (city at night: every 6th from 2), the mouse on it
        photo 200
        scroll 2
        wait 60
        photo 200
        wait 20
        shot world-2-zoomed
        # a step closer: the videos (every 7th photo, and 200) say how long they are; then back
        scroll 1
        wait 60
        shot world-2b-video-marks
        # its mark plays it: the frame over the tile, the controls along its bottom
        click 543 495
        wait 40
        shot world-2c-video-playing
        photo 200
        scroll -1
        wait 60
        photo 200
        click right
        shot world-3-right-click
        appkey Escape
        click 866 27
        shot world-4a-filter-menu
        click 868 154
        type pixel
        shot world-4b-writing-filter
        key Enter
        wait 60
        shot world-4-filter
        appkey KeyN
        wait 60
        shot world-4c-next-match
        appkey KeyH
        shot world-5-help
        appkey KeyH
        # the empty box offers a photo of the map every fourth idea; Enter searches with it
        click 500 27
        click 500 27
        key ArrowDown
        key ArrowDown
        key ArrowDown
        shot world-6-photo-idea
        key Enter
        wait 60
        shot world-7-photo-idea-searched
    ";
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/shots");
    crate::run_headless(crate::library::Library::new(root.join("maps")), (1200.0, 780.0), 1.0, crate::script::Script::new(script, dir));
}
