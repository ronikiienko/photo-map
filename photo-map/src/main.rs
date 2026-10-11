//! Photo map viewer: every Wikivoyage photo as a square at its map position, similar photos close together.
//! Data comes from src/prep_photo_map.py (data/wikivoyage/photo_map/). Run: cargo run --release

mod camera;
mod data;
mod gpu;
mod pages;
mod stream;

use camera::{Camera, Keys};
use data::{MapData, WORLD};
use glam::Vec2;
use gpu::{CameraUniform, Gpu, MapRenderer};
use stream::Streamer;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// intel_tex_2's precompiled ISPC kernels (BC7 encoding in stream.rs) need the C++ runtime
#[link(name = "stdc++")]
unsafe extern "C" {}

/// Side of a photo square in world units: zoom changes it on screen, nothing else does.
const PHOTO_WORLD: f32 = 4.0; // about the median distance between neighbouring photos
/// Squares never shrink below this many pixels, so far-out photos don't vanish.
const MIN_PX: f32 = 1.0;
const HIT: u32 = 1;
const SELECTED: u32 = 2;
const BASE_URL: &str = "https://en.wikivoyage.org/wiki/";

struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

struct Viewer {
    window: Arc<Window>,
    gpu: Gpu,
    map: MapRenderer,
    streamer: Streamer,
    ui: Ui,
    data: MapData,
    camera: Camera,
    keys: Keys,
    cursor: Option<Vec2>,
    press_at: Option<Vec2>,
    hovered: Option<usize>,
    selected: Option<usize>,
    flags: Vec<u32>,
    flags_dirty: bool,
    search: String,
    hits: Vec<usize>,
    /// Full-size photo for the panel, loaded on a thread.
    photo: Option<(usize, egui::TextureHandle)>,
    photo_tx: Sender<(usize, egui::ColorImage)>,
    photo_rx: Receiver<(usize, egui::ColorImage)>,
    /// 64 px tiles from the atlas, for the look-alike grid.
    tiles: HashMap<usize, egui::TextureHandle>,
    last_frame: Instant,
    frame_times: Vec<f32>,
}

impl Viewer {
    fn new(window: Arc<Window>, data: MapData) -> Self {
        let gpu = Gpu::new(window.clone());
        let streamer = Streamer::new(&gpu.device, &gpu.queue, &data);
        let map = MapRenderer::new(&gpu, &data, &streamer);
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());
        let state = egui_winit::State::new(ctx.clone(), ctx.viewport_id(), &window, None, None, None);
        let renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, Default::default());
        let size = window.inner_size();
        let mut camera = Camera::new(Vec2::new(size.width as f32, size.height as f32));
        camera.jump(camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
        let (photo_tx, photo_rx) = channel();
        let n = data.meta.n;
        Self {
            window,
            gpu,
            map,
            streamer,
            ui: Ui { ctx, state, renderer },
            data,
            camera,
            keys: Keys::default(),
            cursor: None,
            press_at: None,
            hovered: None,
            selected: None,
            flags: vec![0; n],
            flags_dirty: true,
            search: String::new(),
            hits: Vec::new(),
            photo: None,
            photo_tx,
            photo_rx,
            tiles: HashMap::new(),
            last_frame: Instant::now(),
            frame_times: Vec::new(),
        }
    }

    fn size_px(&self) -> f32 {
        (self.camera.scale * PHOTO_WORLD).max(MIN_PX)
    }

    /// The photo under the mouse: of the squares containing it, the one drawn last (on top).
    fn pick(&self, screen: Vec2) -> Option<usize> {
        let size = self.size_px();
        let world = self.camera.to_world(screen);
        let reach = Vec2::splat((size * 0.5).max(4.0) / self.camera.scale);
        let mut best = None;
        self.data.grid.within(&self.data.points, world - reach, world + reach, |i| {
            let p = self.data.points[i];
            let half = (self.data.shape(i) * size * 0.5).max(Vec2::splat(4.0));  // real shape; tiny ones stay clickable
            let d = (self.camera.to_screen(Vec2::new(p.x, p.y)) - screen).abs();
            if d.x <= half.x && d.y <= half.y && best.is_none_or(|b| i > b) {
                best = Some(i);
            }
        });
        best
    }

    fn select(&mut self, id: Option<usize>) {
        if let Some(old) = self.selected {
            self.flags[old] &= !SELECTED;
        }
        self.selected = id;
        if let Some(id) = id {
            self.flags[id] |= SELECTED;
            if self.photo.as_ref().is_none_or(|(p, _)| *p != id) {
                self.photo = None;
                let (tx, path) = (self.photo_tx.clone(), self.data.root.join(&self.data.meta.img[id]));
                std::thread::spawn(move || {
                    if let Ok(img) = image::open(&path) {
                        let img = img.thumbnail(800, 800).to_rgba8();
                        let size = [img.width() as usize, img.height() as usize];
                        let _ = tx.send((id, egui::ColorImage::from_rgba_unmultiplied(size, &img)));
                    }
                });
            }
        }
        self.flags_dirty = true;
    }

    fn run_search(&mut self) {
        for &i in &self.hits {
            self.flags[i] &= !HIT;
        }
        let q = self.search.trim().to_lowercase();
        self.hits = if q.chars().count() >= 2 {
            (0..self.data.meta.n).filter(|&i| self.data.search_text[i].contains(&q)).collect()
        } else {
            Vec::new()
        };
        for &i in &self.hits {
            self.flags[i] |= HIT;
        }
        self.flags_dirty = true;
    }

    fn fly_to_hits(&mut self) {
        let (mut min, mut max) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for &i in &self.hits {
            let p = Vec2::new(self.data.points[i].x, self.data.points[i].y);
            min = min.min(p);
            max = max.max(p);
        }
        if !self.hits.is_empty() {
            self.camera.fly_to(self.camera.fit(min, max));
        }
    }

    fn tile(&mut self, id: usize) -> egui::TextureHandle {
        let (ctx, data) = (&self.ui.ctx, &self.data);
        self.tiles
            .entry(id)
            .or_insert_with(|| {
                let (w, h, rgba) = data.tile_rgba(id);
                let image = egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba);
                ctx.load_texture(format!("tile{id}"), image, egui::TextureOptions::LINEAR)
            })
            .clone()
    }

    fn key(&mut self, key: &Key, down: bool) {
        let k = &mut self.keys;
        match key {
            Key::Named(NamedKey::ArrowLeft) => k.left = down,
            Key::Named(NamedKey::ArrowRight) => k.right = down,
            Key::Named(NamedKey::ArrowUp) => k.up = down,
            Key::Named(NamedKey::ArrowDown) => k.down = down,
            Key::Named(NamedKey::Escape) if down => self.select(None),
            Key::Character(c) => match c.to_lowercase().as_str() {
                "a" => k.left = down,
                "d" => k.right = down,
                "w" => k.up = down,
                "s" => k.down = down,
                "e" | "=" | "+" => k.zoom_in = down,
                "q" | "-" => k.zoom_out = down,
                _ => {}
            },
            _ => {}
        }
    }

    fn event(&mut self, event: WindowEvent) {
        let egui_took = self.ui.state.on_window_event(&self.window, &event).consumed;
        let over_ui = self.ui.ctx.is_pointer_over_egui() || self.ui.ctx.egui_is_using_pointer();
        match event {
            WindowEvent::Resized(size) => {
                self.gpu.resize(size.width, size.height);
                self.camera.viewport = Vec2::new(size.width.max(1) as f32, size.height.max(1) as f32);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if let Some(old) = self.cursor {
                    self.camera.drag(pos - old);
                }
                self.cursor = Some(pos);
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::MouseInput { state, button: button @ (MouseButton::Left | MouseButton::Middle), .. } => match state {
                ElementState::Pressed if !over_ui => {
                    self.camera.press();
                    self.press_at = self.cursor;
                }
                ElementState::Released if self.camera.dragging() => {
                    let click = matches!((self.press_at, self.cursor), (Some(a), Some(b)) if a.distance(b) < 4.0);
                    if click {
                        self.camera.stop_drag();
                        if button == MouseButton::Left {
                            let id = self.cursor.and_then(|c| self.pick(c));
                            self.select(id);
                        }
                    } else {
                        self.camera.release();
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, .. } if !over_ui => match delta {
                MouseScrollDelta::LineDelta(_, y) => self.camera.scroll_lines(y),
                MouseScrollDelta::PixelDelta(p) => self.camera.scroll_pixels(p.y as f32),
            },
            WindowEvent::KeyboardInput { event, .. } => {
                let typing = self.ui.ctx.egui_wants_keyboard_input() && event.state == ElementState::Pressed;
                if !typing && !egui_took {
                    self.key(&event.logical_key, event.state == ElementState::Pressed);
                } else if event.state == ElementState::Released {
                    self.key(&event.logical_key, false); // never leave a key stuck down
                }
            }
            WindowEvent::Focused(false) => self.keys = Keys::default(),
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_times.push(dt);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        self.camera.update(dt, self.cursor, self.keys);
        if let Ok((id, image)) = self.photo_rx.try_recv() {
            if self.selected == Some(id) {
                self.photo = Some((id, self.ui.ctx.load_texture("photo", image, egui::TextureOptions::LINEAR)));
            }
        }
        let over_ui = self.ui.ctx.is_pointer_over_egui();
        self.hovered = match self.cursor {
            Some(c) if !self.camera.dragging() && !over_ui => self.pick(c),
            _ => None,
        };

        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) => t,
            wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                drop(t);
                let size = self.window.inner_size();
                self.gpu.resize(size.width, size.height);
                return;
            }
            _ => {
                let size = self.window.inner_size();
                self.gpu.resize(size.width, size.height);
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());

        if self.flags_dirty {
            self.map.upload_flags(&self.gpu.queue, &self.flags);
            self.flags_dirty = false;
        }
        let (view_min, view_max) = (self.camera.to_world(Vec2::ZERO), self.camera.to_world(self.camera.viewport));
        let size_px = self.size_px();
        self.streamer.update(&self.gpu.queue, &self.data, view_min, view_max, self.camera.center, size_px);
        let cam = CameraUniform {
            center: self.camera.center.into(),
            viewport: self.camera.viewport.into(),
            scale: self.camera.scale,
            size_px: self.size_px(),
            searching: (self.search.trim().chars().count() >= 2) as u32,
            _pad: 0,
        };
        self.map.draw(&self.gpu.queue, &mut encoder, &view, cam);
        self.draw_ui(&mut encoder, &view);
        self.gpu.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
        self.window.request_redraw();
    }

    fn draw_ui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let raw = self.ui.state.take_egui_input(&self.window);
        let mut actions = UiActions::default();
        let ctx = self.ui.ctx.clone();
        let mut output = ctx.run_ui(raw, |ui| self.build_ui(ui, &mut actions));
        if actions.search_changed {
            self.run_search();
        }
        if actions.fly_to_hits {
            self.fly_to_hits();
        }
        if let Some(id) = actions.go_to {
            let p = self.data.points[id];
            self.camera.fly_to((Vec2::new(p.x, p.y), self.camera.scale));
            self.select(Some(id));
        }

        self.ui.state.handle_platform_output(&self.window, output.platform_output);
        let tris = self.ui.ctx.tessellate(output.shapes, output.pixels_per_point);
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                self.ui.renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.gpu.config.width, self.gpu.config.height],
            pixels_per_point: output.pixels_per_point,
        };
        self.ui.renderer.update_buffers(&self.gpu.device, &self.gpu.queue, encoder, &tris, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.ui.renderer.render(&mut pass.forget_lifetime(), &tris, &screen);
        }
        for id in &output.textures_delta.free {
            self.ui.renderer.free_texture(id);
        }
        output.textures_delta.clear(); // applied above; egui checks that nothing was forgotten
    }

    fn build_ui(&mut self, ui: &mut egui::Ui, actions: &mut UiActions) {
        let ctx = ui.ctx().clone();

        if let Some(id) = self.selected {
            egui::Panel::right("details").exact_size(340.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.details(ui, id, actions));
            });
        }

        egui::Area::new("search".into()).fixed_pos([10.0, 10.0]).show(&ctx, |ui| {
            ui.horizontal(|ui| {
                let edit = ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("search captions / articles").desired_width(260.0));
                if edit.changed() {
                    actions.search_changed = true;
                }
                if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    actions.fly_to_hits = true;
                }
                if self.search.trim().chars().count() >= 2 {
                    ui.label(format!("{} photos", self.hits.len()));
                }
            });
        });

        let avg = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        egui::Area::new("fps".into()).anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -8.0]).show(&ctx, |ui| {
            ui.label(egui::RichText::new(format!(
                "{:.0} fps · {:.1} ms · streamed {} · loading {}",
                1.0 / avg.max(1e-6),
                avg * 1000.0,
                self.streamer.usage().iter().map(|(s, used, total)| format!("{s}px {used}/{total}")).collect::<Vec<_>>().join(", "),
                self.streamer.in_flight()
            )).monospace().color(egui::Color32::from_rgb(150, 200, 150)));
        });
        egui::Area::new("help".into()).anchor(egui::Align2::LEFT_BOTTOM, [10.0, -8.0]).show(&ctx, |ui| {
            ui.label(egui::RichText::new("scroll / Q E = zoom · drag or WASD = move (flick to throw) · click = details · Esc = close").small().weak());
        });

        if let (Some(id), Some(c)) = (self.hovered, self.cursor) {
            let ppp = ctx.pixels_per_point();
            egui::Area::new("tip".into()).fixed_pos([c.x / ppp + 14.0, c.y / ppp + 14.0]).interactable(false).show(&ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(320.0);
                    let cap = &self.data.meta.cap[id];
                    let arts = &self.data.meta.art[id];
                    let more = if arts.len() > 1 { format!(" +{}", arts.len() - 1) } else { String::new() };
                    ui.label(format!("{} — {}{more}", if cap.is_empty() { "(no caption)" } else { cap }, arts[0]));
                });
            });
        }
    }

    fn details(&mut self, ui: &mut egui::Ui, id: usize, actions: &mut UiActions) {
        let width = ui.available_width();
        match &self.photo {
            Some((p, tex)) if *p == id => {
                let s = tex.size_vec2();
                ui.image((tex.id(), s * (width / s.x)));
            }
            _ => {
                let tile = self.tile(id);
                let s = tile.size_vec2();
                ui.image((tile.id(), s * (width / s.x)));
            }
        }
        let cap = &self.data.meta.cap[id];
        ui.label(if cap.is_empty() { "(no caption)" } else { cap });
        ui.horizontal_wrapped(|ui| {
            ui.label("In:");
            for a in &self.data.meta.art[id] {
                ui.hyperlink_to(a, format!("{BASE_URL}{}", a.replace(' ', "_")));
            }
        });
        let path = self.data.root.join(&self.data.meta.img[id]);
        if ui.link(egui::RichText::new(self.data.meta.img[id].as_str()).small()).clicked() {
            let _ = open::that(&path);
        }
        ui.separator();
        ui.strong("Look-alikes");
        let cell = (width - 12.0) / 3.0;
        let neighbours: Vec<_> = self.data.neighbours(id).collect();
        egui::Grid::new("nb").spacing([6.0, 6.0]).show(ui, |ui| {
            for (j, (n, sim)) in neighbours.into_iter().enumerate() {
                let tile = self.tile(n);
                ui.vertical(|ui| {
                    let s = tile.size_vec2() * (cell / tile.size_vec2().max_elem());  // whole photo, fitted in the cell
                    if ui.add(egui::Button::image(egui::Image::new((tile.id(), s))).min_size(egui::vec2(cell, cell))).clicked() {
                        actions.go_to = Some(n);
                    }
                    ui.small(format!("{sim:.2}"));
                });
                if j % 3 == 2 {
                    ui.end_row();
                }
            }
        });
    }
}

#[derive(Default)]
struct UiActions {
    search_changed: bool,
    fly_to_hits: bool,
    go_to: Option<usize>,
}

struct App {
    data: Option<MapData>,
    viewer: Option<Viewer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("Wikivoyage photo map").with_maximized(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.viewer = Some(Viewer::new(window, self.data.take().unwrap()));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if matches!(event, WindowEvent::CloseRequested) {
            // Drop the viewer (and egui's clipboard thread with it) while the Wayland connection still exists;
            // dropping it after the event loop has torn the connection down segfaults.
            self.viewer = None;
            event_loop.exit();
            return;
        }
        if let Some(v) = &mut self.viewer {
            v.event(event);
        }
    }
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let started = Instant::now();
    let data = MapData::load(&root).expect("map files missing: run python src/prep_photo_map.py first");
    eprintln!("{} photos loaded in {:.1}s", data.meta.n, started.elapsed().as_secs_f32());
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App { data: Some(data), viewer: None }).unwrap();
}
