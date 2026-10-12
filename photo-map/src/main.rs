//! Photo map: a fast viewer for map bundles made in Python (see data.rs): every photo at its map position,
//! similar photos close together, region names as you zoom.
//!     photo-map [bundle]      default: the one opened last in maps/

mod bake;
mod camera;
mod data;
mod gpu;
mod library;
mod pages;
mod photo;
mod stream;

use camera::{Camera, Keys};
use data::{MapData, Progress, WORLD};
use glam::Vec2;
use gpu::{CameraUniform, Gpu, MapRenderer};
use library::Library;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime};
use stream::Streamer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// intel_tex_2's precompiled ISPC kernels (BC7 encoding in stream.rs and bake.rs) need the C++ runtime
#[link(name = "stdc++")]
unsafe extern "C" {}

/// Side of a photo square in world units is the median distance between nearest neighbours: a map of 12,000 photos
/// gets bigger squares than one of 88,000 in the same world (zoom changes the size on screen, nothing else does).
const PHOTO_WORLD_DEFAULT: f32 = 4.0; // before a map is open
/// Squares never shrink below this many pixels, so far-out photos don't vanish.
const MIN_PX: f32 = 1.0;
/// Region names show while their region's spread is between these sizes on screen (points), fading in and out.
/// Region names on the map are switched off until they are put back carefully (the names stay in each map's regions.json).
const SHOW_REGION_NAMES: bool = false;
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
    /// A photo's side in world units (see PHOTO_WORLD_DEFAULT).
    photo_world: f32,
    /// The last search's cosines: the image ones, then the caption ones (0 = no caption), as u16.
    sims: Vec<u16>,
    /// When the bundle's search.json (written by src/map_search.py) was last read.
    search_file: Option<SystemTime>,
}

struct Viewer {
    window: Arc<Window>,
    gpu: Gpu,
    ui: Ui,
    map: Option<OpenMap>,
    library: Library,
    /// A bundle being read from disk (and baked, the first time) on a thread; it becomes `map` when ready.
    opening: Option<(PathBuf, JoinHandle<std::io::Result<MapData>>, Arc<Progress>)>,
    camera: Camera,
    keys: Keys,
    cursor: Option<Vec2>,
    hovered: Option<usize>,
    search: String,
    /// The search helper this viewer started: one for the whole session, it follows the open map.
    helper: Option<Helper>,
    /// Enter was pressed while the helper was still starting: search as soon as it is up.
    ask_when_ready: bool,
    /// A photo dropped on the window: searched for together with the text.
    photo: Option<PathBuf>,
    /// What the last search did, next to the box.
    status: String,
    /// Matches start at this cosine similarity (the strongest colour is WIDTH above).
    cutoff: f32,
    /// What the shown search was for (words and/or a photo's name), from the helper's answer.
    shown: String,
    /// The photo in the box, as a small picture.
    photo_tex: Option<egui::TextureHandle>,
    last_frame: Instant,
    frame_times: Vec<f32>,
    frames: u64,
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
            helper: None,
            ask_when_ready: false,
            photo: None,
            status: String::new(),
            cutoff: 0.72,
            shown: String::new(),
            photo_tex: None,
            last_frame: Instant::now(),
            frame_times: Vec::new(),
            frames: 0,
        };
        if let Some(path) = first {
            viewer.open(path);
        }
        viewer
    }

    /// Start reading a bundle on a thread (photos.json and friends take a moment for big maps; a new bundle is baked).
    fn open(&mut self, path: PathBuf) {
        let p = path.clone();
        let progress = Arc::new(Progress::default());
        let shared = progress.clone();
        self.opening = Some((path, std::thread::spawn(move || MapData::load(&p, &shared)), progress));
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
        let photo_world = median_neighbour_distance(&data.points);
        let search_file = search_file_time(&path);  // one from before this opening is old news
        self.status.clear();
        self.map = Some(OpenMap { path, data, renderer, streamer, flags: vec![0; n], flags_dirty: true, photo_world, sims: Vec::new(), search_file });
        self.follow_map();
    }

    fn size_px(&self) -> f32 {
        (self.camera.scale * self.map.as_ref().map_or(PHOTO_WORLD_DEFAULT, |m| m.photo_world)).max(MIN_PX)
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

    /// Take a photo for the search: remember its path and a small picture of it to show in the bar. False if it can't be read.
    fn set_photo(&mut self, path: PathBuf) -> bool {
        let Ok(img) = image::open(&path) else {
            self.status = "can't read that photo".into();
            return false;
        };
        let small = img.thumbnail(96, 96).to_rgba8();
        let color = egui::ColorImage::from_rgba_unmultiplied([small.width() as usize, small.height() as usize], small.as_raw());
        self.photo_tex = Some(self.ui.ctx.load_texture("query photo", color, egui::TextureOptions::LINEAR));
        self.photo = Some(path);
        true
    }

    /// Make sure a search helper runs and works on the open map: it is started once, and told when the map changes
    /// (the project's logs/search.bundle holds the map's path) so it keeps its model loaded and just loads the map's vectors.
    fn follow_map(&mut self) {
        let Some(path) = self.map.as_ref().map(|m| m.path.clone()) else { return };
        match std::fs::canonicalize(&path).map_err(|e| e.to_string()).and_then(|b| project_root(&b).map(|r| (b, r))) {
            Ok((bundle, root)) => {
                let _ = std::fs::create_dir_all(root.join("logs"));
                let _ = std::fs::write(root.join("logs/search.bundle"), bundle.to_string_lossy().as_bytes());
            }
            Err(e) => {
                self.status = format!("search can't start: {e}");
                return;
            }
        }
        let running = self.helper.as_mut().is_some_and(|h| h.0.try_wait().ok().flatten().is_none());
        if !running {
            self.helper = None; // a dead one's leftovers
            if !helper_alive(&path) {  // not one started by hand
                match start_helper(&path) {
                    Ok(h) => self.helper = Some(h),
                    Err(e) => self.status = format!("search can't start: {e}"),
                }
            }
        }
    }

    /// Enter in the search box, or a photo dropped on the window: ask the helper (src/map_search.py) for the best photos
    /// for the text and/or the photo (query.txt, as JSON); its answer comes back in search.json (poll_search_file).
    /// Nothing to search for clears the last search.
    fn ask(&mut self) {
        let text = self.search.trim().to_string();
        // a path to a photo typed or pasted in the box (from a file manager, or dragged on a terminal) is a photo search
        if let Some(path) = photo_path(&text) {
            if self.set_photo(path) {
                self.search.clear();
            }
        }
        let text = self.search.trim().to_string();
        let Some(path) = self.map.as_ref().map(|m| m.path.clone()) else { return };
        if text.is_empty() && self.photo.is_none() {
            if let Some(m) = &mut self.map {
                m.flags.fill(0);
                m.flags_dirty = true;
                m.sims.clear();
            }
            self.shown.clear();
            self.status.clear();
        } else if helper_alive(&path) {
            let query = serde_json::json!({ "text": text, "image": self.photo });
            let _ = std::fs::write(path.join("query.txt"), query.to_string());
            self.status = "searching…".into();
            self.ask_when_ready = false;
        } else {
            // not answering: still starting (it takes about 30 s the first time), or stopped: start it again
            self.follow_map();
            if self.helper.is_none() && !helper_alive(&path) {
                return; // follow_map said why
            }
            self.ask_when_ready = true;
            self.status = "starting the search (a few seconds, 30 s the first time), it will run when ready…".into();
        }
    }

    /// The answer of src/map_search.py ({"query", "file"}: two u16 cosines per photo): shown (apply_search) and the
    /// camera pulls back to show the matches.
    fn poll_search_file(&mut self) {
        let Some(m) = &mut self.map else { return };
        let time = search_file_time(&m.path);
        if time.is_none() || time == m.search_file {
            return;
        }
        m.search_file = time;
        let Ok(text) = std::fs::read_to_string(m.path.join("search.json")) else { return };
        let Ok(found) = serde_json::from_str::<serde_json::Value>(&text) else { return };
        if let Some(error) = found["error"].as_str() {
            self.status = error.to_string();
            return;
        }
        let Some(bytes) = found["file"].as_str().and_then(|f| std::fs::read(m.path.join(f)).ok()).filter(|r| r.len() == 4 * m.flags.len()) else { return };
        self.shown = found["query"].as_str().unwrap_or_default().to_string();
        m.sims = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if let Some((min, max)) = self.apply_search() {
            self.camera.fly_to(self.camera.fit(min, max));
        }
    }

    /// Turn the last search's cosines into the photos' match bytes (0 = no match, 1..=255 from the cutoff up): a photo's
    /// score is the better of its image and caption cosines. The box the matches span, if there are enough of them.
    fn apply_search(&mut self) -> Option<(Vec2, Vec2)> {
        const WIDTH: f32 = 0.06; // the strongest colour is this far above the cutoff
        let cut = self.cutoff;
        let top = cut + WIDTH;
        const MIN_HITS: usize = 8; // fewer matches than this: nothing matches well
        let m = self.map.as_mut()?;
        let n = m.flags.len();
        if m.sims.len() != 2 * n {
            return None;
        }
        let mut hits = 0;
        for i in 0..n {
            let (a, b) = (m.sims[i] as f32 / 65535.0, m.sims[n + i] as f32 / 65535.0);
            let s = if m.sims[n + i] > 0 { a.max(b) } else { a };
            m.flags[i] = if s >= cut { hits += 1; 1 + ((s - cut) / (top - cut)).clamp(0.0, 1.0).mul_add(254.0, 0.5) as u32 } else { 0 };
        }
        if hits < MIN_HITS {
            m.flags.fill(0);
            self.status = format!("{}: nothing matches well", self.shown);
            m.flags_dirty = true;
            return None;
        }
        self.status = format!("{}: {hits} matches", self.shown);
        m.flags_dirty = true;
        let pts = m.flags.iter().enumerate().filter(|&(_, &f)| f > 0).map(|(i, _)| Vec2::new(m.data.points[i].x, m.data.points[i].y));
        Some(pts.fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(lo, hi), p| (lo.min(p), hi.max(p))))
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
            WindowEvent::DroppedFile(path) => {
                if path.extension().and_then(|e| e.to_str()).is_some_and(|e| PHOTO_TYPES.contains(&e.to_lowercase().as_str())) {
                    if self.set_photo(path) {
                        self.ask();
                    }
                } else {
                    self.status = "that isn't a photo I can search with (jpg, png, webp, gif, bmp)".into();
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
        self.frames += 1;
        if self.frames % 30 == 0 {
            self.poll_search_file();
            if self.ask_when_ready && self.map.as_ref().is_some_and(|m| helper_alive(&m.path)) {
                self.ask();
            }
        }
        if self.opening.as_ref().is_some_and(|(_, job, _)| job.is_finished()) {
            let (path, job, _) = self.opening.take().unwrap();
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
            _pad: [0; 2],
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
        if actions.clear_photo {
            self.photo = None;
            self.photo_tex = None;
            self.ask();
        }
        if actions.ask {
            self.ask();
        }
        if actions.reblend {
            self.apply_search();
        }
        if let Some(path) = actions.open {
            self.open(path);
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
        if SHOW_REGION_NAMES {
            self.draw_labels(&ctx);
        }

        egui::Area::new("top".into()).fixed_pos([10.0, 10.0]).show(&ctx, |ui| {
            ui.horizontal(|ui| {
                let current = self.opening.as_ref().map(|(p, _, _)| p.as_path()).or(self.map.as_ref().map(|m| m.path.as_path()));
                actions.open = self.library.bar(ui, current);
                if let Some((path, _, progress)) = &self.opening {
                    let (text, fraction) = progress.now();
                    ui.spinner();
                    let name = path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
                    ui.colored_label(egui::Color32::from_rgb(240, 190, 90), format!("opening {name}: {text}"));
                    if let Some(f) = fraction {
                        ui.add(egui::ProgressBar::new(f).desired_width(160.0).show_percentage());
                    }
                }
                ui.separator();
                // whether a search can run now
                let helper_up = self.helper.as_mut().is_some_and(|h| h.0.try_wait().ok().flatten().is_none());
                match self.map.as_ref().map(|m| (helper_alive(&m.path), helper_up)) {
                    Some((true, _)) => {
                        ui.colored_label(egui::Color32::from_rgb(110, 200, 120), "● search ready");
                    }
                    Some((false, true)) => {
                        ui.spinner();
                        ui.weak("search starting…");
                    }
                    Some((false, false)) => {
                        ui.weak("○ search off (Enter starts it)");
                    }
                    None => {}
                }
                let edit = ui.add(egui::TextEdit::singleline(&mut self.search).hint_text("words, or a photo (drop it, or paste its path); Enter").desired_width(260.0));
                if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    actions.ask = true;
                }
                if let Some(photo) = &self.photo {
                    if let Some(tex) = &self.photo_tex {
                        ui.add(egui::Image::new((tex.id(), egui::vec2(36.0 * tex.aspect_ratio(), 36.0))));
                    }
                    ui.label(photo.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
                    if ui.small_button("x").on_hover_text("remove the photo and search again").clicked() {
                        actions.clear_photo = true;
                    }
                }
                ui.label("cutoff");
                if ui.add(egui::Slider::new(&mut self.cutoff, 0.1..=0.95).fixed_decimals(2)).changed() {
                    actions.reblend = true;
                }
                ui.label(&self.status);
            });
        });
        if self.map.is_none() && self.opening.is_none() {
            egui::Area::new("empty".into()).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.weak(if self.library.maps.is_empty() { "No maps in maps/ yet: make one in Python (src/photomap.py)" } else { "Choose a map above" });
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
    ask: bool,
    reblend: bool,
    clear_photo: bool,
    open: Option<PathBuf>,
}

struct App {
    maps: PathBuf,
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
        let library = Library::new(self.maps.clone());
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

/// The median distance from a photo to its nearest neighbour, over a sample of up to 2,000 photos.
fn median_neighbour_distance(points: &[data::Point]) -> f32 {
    let step = (points.len() / 2000).max(1);
    let mut nearest: Vec<f32> = (0..points.len())
        .step_by(step)
        .map(|i| {
            let p = points[i];
            let d2 = points.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, q)| (q.x - p.x).powi(2) + (q.y - p.y).powi(2)).fold(f32::MAX, f32::min);
            d2.sqrt()
        })
        .collect();
    nearest.sort_by(f32::total_cmp);
    nearest.get(nearest.len() / 2).copied().filter(|d| *d > 0.0).unwrap_or(PHOTO_WORLD_DEFAULT)
}

const PHOTO_TYPES: [&str; 6] = ["jpg", "jpeg", "png", "webp", "gif", "bmp"];

/// The file a text names, if it is a photo: plain, in quotes, with escaped spaces, or as file://.
fn photo_path(text: &str) -> Option<PathBuf> {
    let t = text.trim().trim_matches(|c| c == '\'' || c == '"').replace("\\ ", " ");
    let path = PathBuf::from(t.strip_prefix("file://").unwrap_or(&t));
    let image = path.extension().and_then(|e| e.to_str()).is_some_and(|e| PHOTO_TYPES.contains(&e.to_lowercase().as_str()));
    (image && path.is_file()).then_some(path)
}

/// The project folder: the one that has the bundle's maps/ folder.
fn project_root(bundle: &std::path::Path) -> Result<PathBuf, String> {
    bundle.parent().and_then(|p| p.parent()).map(PathBuf::from).ok_or_else(|| "no project folder around the map".to_string())
}

/// The search helper (src/map_search.py) this viewer started; stopped with it.
struct Helper(std::process::Child);

impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Start the helper for a bundle: the project's .venv python on src/map_search.py, its output in logs/map_search.log.
/// (The project folder is the one with the bundle's maps/ folder.)
fn start_helper(bundle: &std::path::Path) -> Result<Helper, String> {
    let bundle = std::fs::canonicalize(bundle).map_err(|e| e.to_string())?;
    let root = project_root(&bundle)?;
    let (python, script) = (root.join(".venv/bin/python"), root.join("src/map_search.py"));
    if !python.exists() || !script.exists() {
        return Err("search needs .venv/bin/python and src/map_search.py in the project folder".into());
    }
    let _ = std::fs::create_dir_all(root.join("logs"));
    let log = std::fs::File::create(root.join("logs/map_search.log")).map_err(|e| e.to_string())?;
    let child = std::process::Command::new(python)
        .arg(script)
        .arg(&bundle)
        .current_dir(&root)
        .stdin(std::process::Stdio::null())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(Helper(child))
}

/// Is src/map_search.py running for this bundle? It keeps query.ready fresh.
fn helper_alive(bundle: &std::path::Path) -> bool {
    std::fs::metadata(bundle.join("query.ready")).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|age| age.as_secs_f32() < 3.0)
}

/// When a bundle's search.json was last written, if it has one.
fn search_file_time(bundle: &std::path::Path) -> Option<SystemTime> {
    std::fs::metadata(bundle.join("search.json")).and_then(|m| m.modified()).ok()
}

fn main() {
    let maps = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("maps");
    let first = std::env::args().nth(1).map(PathBuf::from);
    // winit doesn't deliver files dropped on a native Wayland window (the photo search takes dropped photos), but does on
    // an X11 one: with an X server around (XWayland), use that, unless PHOTO_MAP_WAYLAND is set
    if std::env::var_os("DISPLAY").is_some() && std::env::var_os("PHOTO_MAP_WAYLAND").is_none() {
        // SAFETY: nothing else is running yet to read the environment
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
    }
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App { maps, first, viewer: None }).unwrap();
}
