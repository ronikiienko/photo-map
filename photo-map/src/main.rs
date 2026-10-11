//! Photo map: every photo of a map bundle at its map position, similar photos close together.
//!     photo-map [bundle]                         the app: view a bundle (default: the one opened last in maps/),
//!                                                switch maps, make new ones from a folder
//!     photo-map make <bundle> <folder>...        make one without the app (see make/)
//!     photo-map regions <bundle>                 (re)name an existing bundle's regions

mod camera;
mod data;
mod gpu;
mod library;
mod make;
mod pages;
mod photo;
mod stream;

use camera::{Camera, Keys};
use data::{MapData, WORLD};
use glam::Vec2;
use gpu::{CameraUniform, Gpu, MapRenderer};
use library::{Action, Library};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;
use stream::Streamer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// intel_tex_2's precompiled ISPC kernels (BC7 encoding in stream.rs and make/bake.rs) need the C++ runtime
#[link(name = "stdc++")]
unsafe extern "C" {}

/// Side of a photo square in world units: zoom changes it on screen, nothing else does.
const PHOTO_WORLD: f32 = 4.0; // about the median distance between neighbouring photos
/// Squares never shrink below this many pixels, so far-out photos don't vanish.
const MIN_PX: f32 = 1.0;
const HIT: u32 = 1;
/// Region names show while their region's spread is between these sizes on screen (points), fading in and out.
const LABEL_IN: (f32, f32) = (25.0, 50.0);
const LABEL_OUT: (f32, f32) = (300.0, 500.0);

struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

/// The open map: its data, its GPU side and its search.
struct OpenMap {
    path: PathBuf,
    data: MapData,
    renderer: MapRenderer,
    streamer: Streamer,
    flags: Vec<u32>,
    flags_dirty: bool,
    hits: Vec<usize>,
}

struct Viewer {
    window: Arc<Window>,
    gpu: Gpu,
    ui: Ui,
    map: Option<OpenMap>,
    library: Library,
    /// A bundle being read from disk on a thread; it becomes `map` when ready.
    opening: Option<(PathBuf, JoinHandle<std::io::Result<MapData>>)>,
    camera: Camera,
    keys: Keys,
    cursor: Option<Vec2>,
    hovered: Option<usize>,
    search: String,
    last_frame: Instant,
    frame_times: Vec<f32>,
}

impl Viewer {
    fn new(window: Arc<Window>, library: Library, first: Option<PathBuf>) -> Self {
        let gpu = Gpu::new(window.clone());
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());
        let state = egui_winit::State::new(ctx.clone(), ctx.viewport_id(), &window, None, None, None);
        let renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, Default::default());
        let size = window.inner_size();
        let camera = Camera::new(Vec2::new(size.width as f32, size.height as f32));
        let mut viewer = Self {
            window,
            gpu,
            ui: Ui { ctx, state, renderer },
            map: None,
            library,
            opening: None,
            camera,
            keys: Keys::default(),
            cursor: None,
            hovered: None,
            search: String::new(),
            last_frame: Instant::now(),
            frame_times: Vec::new(),
        };
        if let Some(path) = first {
            viewer.open(path);
        }
        viewer
    }

    /// Start reading a bundle on a thread (photos.json and friends take a moment for big maps).
    fn open(&mut self, path: PathBuf) {
        let p = path.clone();
        self.opening = Some((path, std::thread::spawn(move || MapData::load(&p))));
    }

    /// The bundle has been read: let go of the old map, put the new one on the GPU and show all of it.
    fn opened(&mut self, path: PathBuf, data: MapData) {
        self.map = None; // free the old map's GPU memory first
        let started = Instant::now();
        let streamer = Streamer::new(&self.gpu.device, &self.gpu.queue, &data);
        let renderer = MapRenderer::new(&self.gpu, &data, &streamer);
        eprintln!("{}: {} photos opened in {:.1}s", data.manifest.name, data.n(), started.elapsed().as_secs_f32());
        self.window.set_title(&format!("{} — photo map", data.manifest.name));
        self.library.remember(&path);
        self.camera.jump(self.camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
        let n = data.n();
        self.map = Some(OpenMap { path, data, renderer, streamer, flags: vec![0; n], flags_dirty: true, hits: Vec::new() });
        self.run_search();
    }

    fn size_px(&self) -> f32 {
        (self.camera.scale * PHOTO_WORLD).max(MIN_PX)
    }

    /// The photo under the mouse: of the squares containing it, the one drawn last (on top).
    fn pick(&self, screen: Vec2) -> Option<usize> {
        let data = &self.map.as_ref()?.data;
        let size = self.size_px();
        let world = self.camera.to_world(screen);
        let reach = Vec2::splat((size * 0.5).max(4.0) / self.camera.scale);
        let mut best = None;
        data.grid.within(&data.points, world - reach, world + reach, |i| {
            let p = data.points[i];
            let half = (data.shape(i) * size * 0.5).max(Vec2::splat(4.0));  // real shape; tiny ones stay clickable
            let d = (self.camera.to_screen(Vec2::new(p.x, p.y)) - screen).abs();
            if d.x <= half.x && d.y <= half.y && best.is_none_or(|b| i > b) {
                best = Some(i);
            }
        });
        best
    }

    fn run_search(&mut self) {
        let Some(m) = &mut self.map else { return };
        for &i in &m.hits {
            m.flags[i] &= !HIT;
        }
        let q = self.search.trim().to_lowercase();
        m.hits = if q.chars().count() >= 2 {
            (0..m.data.n()).filter(|&i| m.data.search_text[i].contains(&q)).collect()
        } else {
            Vec::new()
        };
        for &i in &m.hits {
            m.flags[i] |= HIT;
        }
        m.flags_dirty = true;
    }

    fn fly_to_hits(&mut self) {
        let Some(m) = &self.map else { return };
        let (mut min, mut max) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for &i in &m.hits {
            let p = Vec2::new(m.data.points[i].x, m.data.points[i].y);
            min = min.min(p);
            max = max.max(p);
        }
        if !m.hits.is_empty() {
            self.camera.fly_to(self.camera.fit(min, max));
        }
    }

    fn key(&mut self, key: &Key, down: bool) {
        let k = &mut self.keys;
        match key {
            Key::Named(NamedKey::ArrowLeft) => k.left = down,
            Key::Named(NamedKey::ArrowRight) => k.right = down,
            Key::Named(NamedKey::ArrowUp) => k.up = down,
            Key::Named(NamedKey::ArrowDown) => k.down = down,
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
            WindowEvent::MouseInput { state, button: MouseButton::Left | MouseButton::Middle, .. } => match state {
                ElementState::Pressed if !over_ui => self.camera.press(),
                ElementState::Released if self.camera.dragging() => self.camera.release(),
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
        if self.opening.as_ref().is_some_and(|(_, job)| job.is_finished()) {
            let (path, job) = self.opening.take().unwrap();
            match job.join() {
                Ok(Ok(data)) => self.opened(path, data),
                Ok(Err(e)) => eprintln!("can't open {}: {e}", path.display()),
                Err(_) => eprintln!("can't open {}: reading it crashed", path.display()),
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

        let (view_min, view_max) = (self.camera.to_world(Vec2::ZERO), self.camera.to_world(self.camera.viewport));
        let size_px = self.size_px();
        let cam = CameraUniform {
            center: self.camera.center.into(),
            viewport: self.camera.viewport.into(),
            scale: self.camera.scale,
            size_px,
            searching: (self.search.trim().chars().count() >= 2) as u32,
            _pad: 0,
        };
        match &mut self.map {
            Some(m) => {
                if m.flags_dirty {
                    m.renderer.upload_flags(&self.gpu.queue, &m.flags);
                    m.flags_dirty = false;
                }
                m.streamer.update(&self.gpu.queue, &m.data, view_min, view_max, self.camera.center, size_px);
                m.renderer.draw(&self.gpu.queue, &mut encoder, &view, cam);
            }
            None => gpu::clear(&mut encoder, &view),
        }
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
        match actions.library {
            Some(Action::Open(path)) => self.open(path),
            Some(Action::Making) => {
                self.map = None; // its GPU memory goes to embedding
                self.window.set_title("photo map");
            }
            None => {}
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

    /// Region names over the map: each shows while its region is a sensible size on screen (coarse names zoomed out,
    /// finer ones as you zoom in), coarser and bigger regions first, skipping names that would overlap one already shown.
    fn draw_labels(&self, ctx: &egui::Context) {
        let Some(m) = &self.map else { return };
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "region names".into()));
        let ppp = ctx.pixels_per_point();
        let screen = ctx.content_rect();
        let smooth = |(a, b): (f32, f32), x: f32| ((x - a) / (b - a)).clamp(0.0, 1.0);
        let mut taken: Vec<egui::Rect> = Vec::new();
        for (level, labels) in m.data.regions.levels.iter().enumerate() {
            // coarsest in big gold letters, then white, each finer level a little smaller and greyer
            let size = (26.0 - 3.0 * level as f32).max(13.0);
            let colour = match level {
                0 => egui::Color32::from_rgb(255, 220, 120),
                l => {
                    let g = (245 - 15 * (l as i32 - 1)).max(185) as u8;
                    egui::Color32::from_rgb(g, g, g.saturating_add(8))
                }
            };
            for l in labels {
                let on_screen = l.spread * self.camera.scale / ppp;
                let alpha = smooth(LABEL_IN, on_screen) * (1.0 - smooth(LABEL_OUT, on_screen));
                if alpha < 0.02 {
                    continue;
                }
                let at = self.camera.to_screen(Vec2::new(l.x, l.y)) / ppp;
                let galley = painter.layout_no_wrap(l.name.clone(), egui::FontId::proportional(size), colour);
                let rect = egui::Rect::from_center_size(egui::pos2(at.x, at.y), galley.size());
                if !screen.intersects(rect) || taken.iter().any(|t| t.intersects(rect.expand(6.0))) {
                    continue;
                }
                taken.push(rect);
                let outline = egui::Color32::from_black_alpha((220.0 * alpha) as u8);
                for (dx, dy) in [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)] {
                    painter.galley_with_override_text_color(rect.min + egui::vec2(dx, dy), galley.clone(), outline);
                }
                painter.galley_with_override_text_color(rect.min, galley, colour.gamma_multiply(alpha));
            }
        }
    }

    fn build_ui(&mut self, ui: &mut egui::Ui, actions: &mut UiActions) {
        let ctx = ui.ctx().clone();
        if !self.library.making() {
            self.draw_labels(&ctx);
        }

        egui::Area::new("top".into()).fixed_pos([10.0, 10.0]).show(&ctx, |ui| {
            ui.horizontal(|ui| {
                let current = self.opening.as_ref().map(|(p, _)| p.as_path()).or(self.map.as_ref().map(|m| m.path.as_path()));
                actions.library = self.library.bar(ui, current);
                if self.opening.is_some() {
                    ui.spinner();
                }
                ui.separator();
                let edit = ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("search titles / metadata / file names").desired_width(260.0));
                if edit.changed() {
                    actions.search_changed = true;
                }
                if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    actions.fly_to_hits = true;
                }
                if let Some(m) = &self.map
                    && self.search.trim().chars().count() >= 2
                {
                    ui.label(format!("{} photos", m.hits.len()));
                }
            });
        });
        if let Some(a) = self.library.window(&ctx) {
            actions.library = Some(a);
        }
        if self.map.is_none() && self.opening.is_none() && !self.library.making() {
            egui::Area::new("empty".into()).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.weak(if self.library.maps.is_empty() { "No maps yet: make one with \"New map…\"" } else { "Choose a map above" });
            });
        }

        let avg = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        if let Some(m) = &self.map {
            egui::Area::new("fps".into()).anchor(egui::Align2::RIGHT_BOTTOM, [-10.0, -8.0]).show(&ctx, |ui| {
                ui.label(egui::RichText::new(format!(
                    "{:.0} fps · {:.1} ms · streamed {} · loading {}",
                    1.0 / avg.max(1e-6),
                    avg * 1000.0,
                    m.streamer.usage().iter().map(|(s, used, total)| format!("{s}px {used}/{total}")).collect::<Vec<_>>().join(", "),
                    m.streamer.in_flight()
                )).monospace().color(egui::Color32::from_rgb(150, 200, 150)));
            });
        }
        egui::Area::new("help".into()).anchor(egui::Align2::LEFT_BOTTOM, [10.0, -8.0]).show(&ctx, |ui| {
            ui.label(egui::RichText::new("scroll / Q E = zoom · drag or WASD = move (flick to throw)").small().weak());
        });

        if let (Some(id), Some(c), Some(m)) = (self.hovered, self.cursor, &self.map) {
            let ppp = ctx.pixels_per_point();
            egui::Area::new("tip".into()).fixed_pos([c.x / ppp + 14.0, c.y / ppp + 14.0]).interactable(false).show(&ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_max_width(320.0);
                    ui.label(&m.data.photos[id].title);
                });
            });
        }
    }
}

#[derive(Default)]
struct UiActions {
    search_changed: bool,
    fly_to_hits: bool,
    library: Option<Action>,
}

struct App {
    project: PathBuf,
    /// The bundle to open first (from the command line), else the one opened last.
    first: Option<PathBuf>,
    viewer: Option<Viewer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("photo map").with_maximized(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let library = Library::new(&self.project);
        let first = self.first.take().or_else(|| library.last());
        self.viewer = Some(Viewer::new(window, library, first));
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
    let project = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("make") {
        if args.len() < 3 {
            eprintln!("usage: photo-map make <bundle> <folder> [<folder> ...]");
            std::process::exit(2);
        }
        let folders: Vec<PathBuf> = args[2..].iter().map(Into::into).collect();
        if let Err(e) = make::run(&project, Path::new(&args[1]), &folders) {
            eprintln!("make failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.first().map(String::as_str) == Some("regions") {
        let Some(bundle) = args.get(1) else {
            eprintln!("usage: photo-map regions <bundle>");
            std::process::exit(2);
        };
        if let Err(e) = make::add_regions(&project, Path::new(bundle)) {
            eprintln!("regions failed: {e}");
            std::process::exit(1);
        }
        return;
    }
    let first = args.first().map(PathBuf::from);
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App { project, first, viewer: None }).unwrap();
}
