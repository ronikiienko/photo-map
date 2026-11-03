//! What the UI tests that draw pictures share: where the pictures go, and a made-up app folder (maps with covers) for the Maps page.
//! The pictures are for looking at, not compared: `cargo test --bin photo-map shots -- --ignored` draws every screen to .local/shots/.

use std::path::{Path, PathBuf};

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

/// A made-up app folder, .local/ui-fixture, with these maps in it (name, photos, created): each a manifest and a cover (a gradient
/// of its own colour). Made again every time, so it is what the test says.
pub fn app_folder(maps: &[(&str, usize, &str)]) -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/ui-fixture");
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("maps");
    std::fs::create_dir_all(&dir).unwrap();
    for (k, (name, photos, created)) in maps.iter().enumerate() {
        let map = dir.join(format!("{}.photomap", name.to_lowercase().replace(' ', "-")));
        std::fs::create_dir_all(&map).unwrap();
        let manifest = serde_json::json!({ "name": name, "photos": photos, "created": created });
        std::fs::write(map.join("manifest.json"), manifest.to_string()).unwrap();
        let tint = [(90u8, 140u8, 200u8), (200, 140, 90), (110, 180, 120), (170, 110, 190)][k % 4];
        let cover = image::RgbaImage::from_fn(192, 192, |x, y| {
            let f = |c: u8, t: u32| (c as u32 * (128 + t) / 320) as u8;
            image::Rgba([f(tint.0, x), f(tint.1, y), f(tint.2, (x + y) / 2), 255])
        });
        cover.save(map.join("cover.png")).unwrap();
    }
    root
}
