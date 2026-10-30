//! Photo map: a fast viewer for map bundles made in Python (see data.rs): every photo at its map position,
//! similar photos close together, region names as you zoom.
//!     photo-map [bundle]      default: the one opened last in maps/

mod axes;
mod beside;
mod bake;
mod backends;
mod bc7;
mod build;
mod camera;
mod cells;
mod data;
mod embed;
mod files;
mod gpu;
mod groups;
mod home;
mod layout;
mod library;
#[cfg(test)]
mod macbench;
mod manager;
mod modelstore;
mod models;
mod packs;
mod patch;
mod runtime;
mod naming;
mod pages;
mod regions;
mod reuse;
mod photo;
mod places;
mod search;
mod siglip;
mod storage;
mod stream;
mod theme;
mod tree;

use camera::{Camera, Keys};
use data::{MapData, Progress, WORLD};
use glam::Vec2;
use gpu::{CameraUniform, Gpu, MapRenderer};
use library::Library;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;
use stream::Streamer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Side of a photo square in world units is the median distance between nearest neighbours: a map of 12,000 photos
/// gets bigger squares than one of 88,000 in the same world (zoom changes the size on screen, nothing else does).
const PHOTO_WORLD_DEFAULT: f32 = 4.0; // before a map is open
/// Squares never shrink below this many pixels, so far-out photos don't vanish.
const MIN_PX: f32 = 1.0;
/// Region names show while their region's spread is between these sizes on screen (points), fading in and out.
/// Region names on the map (region_names.json, else regions.json).
const SHOW_REGION_NAMES: bool = true;
/// A photo shows its own caption above it once it is this wide on screen (points), fully from the second size.
const CAPTION_FROM: (f32, f32) = (130.0, 190.0);
/// Stepping to a match zooms in until its photo is this wide on screen (points), unless you are already closer.
const STEP_TO: f32 = 240.0;
const LABEL_IN: (f32, f32) = (25.0, 50.0);
const LABEL_OUT: (f32, f32) = (300.0, 500.0);
/// Seconds the photos take to slide to their places in another view.
const SLIDE_SECONDS: f32 = 0.8;

/// Where the photos are: the map's own layout (similar photos together), along two pairs of words (axes.rs), or sorted into
/// groups by words (groups.rs).
#[derive(Clone, Copy, PartialEq)]
enum View {
    Map,
    Axes,
    Groups,
}

/// The two screens: your maps (and the ones being made), and a map.
#[derive(Clone, Copy, PartialEq)]
enum Page {
    Maps,
    Viewer,
}

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
    /// The last search's cosine with every photo's picture (empty: no search).
    sims: Vec<f32>,
    /// The photos that match the search, best first, and which of them the last next/previous step went to.
    order: Vec<usize>,
    at: Option<usize>,
    /// Where each match's copy sits beside the world (in the order of `order`), and a copy's side (world units).
    copies: Vec<data::Point>,
    copy_side: f32,
    /// The block beside the world shows a path between these two photos instead of a search's matches.
    trail: Option<(usize, usize)>,
    /// The tree paths follow (tree.rs), if the map has one.
    tree: Option<Vec<u32>>,
    /// The photos' vectors, if the map has them.
    vectors: Option<search::Vectors>,
    /// Which view the photos are in, and the map's own layout to go back to (its places and photo size).
    view: View,
    map_points: Vec<data::Point>,
    map_photo_world: f32,
    /// Where the photos were before the view last changed, and when it changed: they slide from there for SLIDE_SECONDS.
    from: Vec<data::Point>,
    moved: Option<Instant>,
    /// The photos' cosines with the axes' four words, kept while the Axes view shows (changing how they spread needs no model).
    axes_cos: Option<[Vec<f32>; 4]>,
    /// The Groups view's islands and their words (empty in the other views).
    islands: Vec<groups::Island>,
    island_words: Vec<String>,
}

impl OpenMap {
    /// How far the photos are on their way to their new places: 0 ... 1, eased.
    fn slide(&self) -> f32 {
        let t = self.moved.map_or(1.0, |at| (at.elapsed().as_secs_f32() / SLIDE_SECONDS).min(1.0));
        t * t * (3.0 - 2.0 * t)
    }

    /// Put copies of these photos beside the world, in this order (the search's matches, or a path).
    fn show_beside(&mut self, queue: &wgpu::Queue, photos: Vec<usize>) {
        self.order = photos;
        self.at = None;
        (self.copies, self.copy_side) = beside::place(self.order.len());
        self.renderer.set_copies(queue, &self.order, &self.copies);
    }

    /// The box the photos beside the world and their copies span.
    fn beside_box(&self) -> (Vec2, Vec2) {
        let pts = self.order.iter().map(|&i| self.data.points[i]).chain(self.copies.iter().copied()).map(|p| Vec2::new(p.x, p.y));
        pts.fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(lo, hi), p| (lo.min(p), hi.max(p)))
    }

    /// Put the photos at new places, sliding there from wherever they are drawn now (even halfway through another slide).
    fn move_to(&mut self, queue: &wgpu::Queue, view: View, points: Vec<data::Point>, photo_world: f32) {
        let t = self.slide();
        let shown: Vec<data::Point> = self.from.iter().zip(&self.data.points).map(|(a, b)| data::Point { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }).collect();
        self.data.set_points(points);
        self.renderer.move_points(queue, &self.data, &shown);
        self.from = shown;
        self.moved = Some(Instant::now());
        self.photo_world = photo_world;
        self.view = view;
    }
}

struct Viewer {
    window: Arc<Window>,
    gpu: Gpu,
    ui: Ui,
    page: Page,
    home: home::Home,
    /// Maps being made that have a thread (running, or stopped with an error to show).
    runs: Vec<build::Run>,
    /// Maps waiting for the one being made to finish or pause.
    queue: Vec<PathBuf>,
    /// A line for the Maps page: what happened (text, is it good news).
    notice: Option<(String, bool)>,
    last_refresh: Instant,
    map: Option<OpenMap>,
    library: Library,
    /// A bundle being read from disk (and baked, the first time) on a thread; it becomes `map` when ready.
    opening: Option<(PathBuf, JoinHandle<std::io::Result<MapData>>, Arc<Progress>)>,
    camera: Camera,
    keys: Keys,
    alt: bool,
    /// Fingers are on the trackpad, scrolling (or the system's momentum after them): the scroll pans instead of zooming.
    scroll_gesture: bool,
    /// The window is minimized or covered: no frames are asked for until it is shown again.
    hidden: bool,
    help_open: bool,
    /// Builds stopped because a map was opened (a build and a map together would not fit on the card): they go on when the Maps page is back.
    paused_for_viewing: Vec<PathBuf>,
    /// Files and folders dropped on the Maps page this frame.
    dropped: Vec<PathBuf>,
    cursor: Option<Vec2>,
    hovered: Option<usize>,
    /// The match whose copy beside the world is under the mouse (its position among the matches; `hovered` is its photo).
    hovered_copy: Option<usize>,
    /// The right-click menu: for which photo, and where (points).
    menu: Option<(usize, egui::Pos2)>,
    /// "Details" was chosen in the menu: the photo, where the card goes, and its lines (files::details).
    details: Option<(usize, egui::Pos2, files::Details)>,
    /// "Path from here" was chosen on this photo: the path's end is picked next.
    path_from: Option<usize>,
    /// The region names drawn last frame, where on screen (points) and which (level, number in the level): a click on one zooms to its region.
    label_rects: Vec<(egui::Rect, usize, usize)>,
    search: String,
    /// The text model, on a thread of its own.
    engine: search::Engine,
    /// The query it is working on.
    asked: Option<u64>,
    /// A photo dropped on the window: searched for together with the text.
    photo: Option<PathBuf>,
    /// What the last search did, next to the box.
    status: String,
    /// The words of the Axes view and of the Groups view (as typed), and the view the Layout menu has chosen.
    axes: axes::Words,
    axes_spread: axes::Spread,
    groups_text: String,
    view_choice: View,
    /// A layout's words on their way from the model: which view they are for, the query number each was asked as (None once
    /// answered), and the answers.
    layout_for: View,
    layout_words: Vec<String>,
    layout_asked: Vec<Option<u64>>,
    layout_vectors: Vec<Option<Vec<f32>>>,
    /// How many photos a search should light up (the slider): the closest this many match.
    wanted: f32,
    /// Photos drawn with their pixels blended (smooth) or each pixel as it is (nearest).
    smooth: bool,
    /// What the shown search was for (words and/or a photo's name).
    shown: String,
    /// The photo in the box, as a small picture.
    photo_tex: Option<egui::TextureHandle>,
    /// The runtime libraries: what is installed and the install under way.
    runtime: manager::Manager,
    store: modelstore::ModelStore,
    /// The Restart button was pressed: the app closes and opens again.
    restart: bool,
    last_frame: Instant,
    frame_times: Vec<f32>,
    frames: u64,
}

impl Viewer {
    fn new(window: Arc<Window>, library: Library, first: Option<PathBuf>) -> Self {
        let gpu = Gpu::new(window.clone());
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        let engine = {
            let window = window.clone();
            search::Engine::start(library.root(), move || window.request_redraw())
        };
        let runtime = {
            let window = window.clone();
            manager::Manager::new(library.root(), move || window.request_redraw())
        };
        let store = {
            let window = window.clone();
            modelstore::ModelStore::new(library.root(), move || window.request_redraw())
        };
        let state = egui_winit::State::new(ctx.clone(), ctx.viewport_id(), &window, None, None, None);
        let renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, Default::default());
        let size = window.inner_size();
        let camera = Camera::new(Vec2::new(size.width as f32, size.height as f32));
        let mut viewer = Self {
            window,
            gpu,
            ui: Ui { ctx, state, renderer },
            page: if first.is_some() { Page::Viewer } else { Page::Maps },
            home: home::Home::new(),
            runs: Vec::new(),
            queue: Vec::new(),
            notice: None,
            last_refresh: Instant::now(),
            map: None,
            library,
            runtime,
            store,
            restart: false,
            opening: None,
            camera,
            keys: Keys::default(),
            alt: false,
            scroll_gesture: false,
            hidden: false,
            help_open: false,
            paused_for_viewing: Vec::new(),
            dropped: Vec::new(),
            cursor: None,
            hovered: None,
            hovered_copy: None,
            menu: None,
            details: None,
            path_from: None,
            label_rects: Vec::new(),
            search: String::new(),
            engine,
            asked: None,
            photo: None,
            status: String::new(),
            axes: axes::Words::default(),
            axes_spread: axes::Spread::default(),
            groups_text: String::new(),
            view_choice: View::Map,
            layout_for: View::Map,
            layout_words: Vec::new(),
            layout_asked: Vec::new(),
            layout_vectors: Vec::new(),
            wanted: 100.0,
            smooth: true,
            shown: String::new(),
            photo_tex: None,
            last_frame: Instant::now(),
            frame_times: Vec::new(),
            frames: 0,
        };
        if let Some(path) = first {
            viewer.open(path);
        }
        if viewer.library.resume_on_start() {
            viewer.queue = viewer.library.builds.iter().map(|b| b.dir.clone()).collect();
        }
        viewer
    }

    /// Start reading a bundle on a thread (photos.json and friends take a moment for big maps; a new bundle is baked).
    fn open(&mut self, path: PathBuf) {
        for run in self.runs.iter().filter(|r| r.running()) {
            run.pause();
            if !self.paused_for_viewing.contains(&run.dir) {
                self.paused_for_viewing.push(run.dir.clone());
            }
        }
        let p = path.clone();
        let progress = Arc::new(Progress::default());
        let shared = progress.clone();
        self.opening = Some((path, std::thread::spawn(move || MapData::load(&p, &shared)), progress));
    }

    /// The bundle has been read: let go of the old map, put the new one on the GPU and show all of it.
    fn opened(&mut self, path: PathBuf, data: MapData) {
        self.page = Page::Viewer;
        self.map = None; // free the old map's GPU memory first
        let started = Instant::now();
        let streamer = Streamer::new(&self.gpu.device, &self.gpu.queue, &data);
        let renderer = MapRenderer::new(&self.gpu, &data, &streamer);
        eprintln!("{}: {} photos opened in {:.1}s", data.manifest.name, data.n(), started.elapsed().as_secs_f32());
        self.window.set_title(&format!("{} — photo map", data.manifest.name));
        self.camera.jump(self.camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
        let n = data.n();
        let photo_world = data.side;
        self.status.clear();
        // the last map's search is over with it: its words, its photo, and what the box said it was
        self.search.clear();
        self.photo = None;
        self.photo_tex = None;
        self.shown.clear();
        let vectors = search::Vectors::open(&path, n, data.manifest.dim());
        let tree = tree::read(&path, n);
        let in_process = vectors.is_some();
        self.asked = None;
        self.view_choice = View::Map; // a map opens as itself; the words are kept for the next one
        self.layout_asked.clear();
        self.engine.want(data.manifest.model()); // load the map's model now, so that the first search is quick
        let (map_points, from) = (data.points.clone(), data.points.clone());
        self.map = Some(OpenMap {
            path,
            data,
            renderer,
            streamer,
            flags: vec![0; n],
            flags_dirty: true,
            photo_world,
            sims: Vec::new(),
            order: Vec::new(),
            at: None,
            copies: Vec::new(),
            copy_side: 0.0,
            trail: None,
            tree,
            vectors,
            view: View::Map,
            map_points,
            map_photo_world: photo_world,
            from,
            moved: None,
            axes_cos: None,
            islands: Vec::new(),
            island_words: Vec::new(),
        });
        if !in_process {
            self.status = "this map has no search vectors (embeddings.bin)".into();
        }
    }

    fn wake(&self) -> impl Fn() + Send + Sync + 'static {
        let window = self.window.clone();
        move || window.request_redraw()
    }

    /// Maps being made, as the Maps page shows them.
    fn build_views(&self) -> Vec<home::BuildView> {
        let backend = match self.engine.hub.state() {
            hub if hub.loaded.is_none() => String::new(),
            hub if hub.on_gpu && cfg!(target_os = "macos") => "on the Apple chip".to_string(),
            hub if hub.on_gpu => "on the graphics card".to_string(),
            _ => "on the processor".to_string(),
        };
        self.library
            .builds
            .iter()
            .map(|b| {
                let run = self.runs.iter().find(|r| r.dir == b.dir);
                let state = run.map(|r| r.state());
                home::BuildView {
                    dir: b.dir.clone(),
                    name: b.job.name.clone(),
                    source: build::describe(&b.job.sources),
                    stage: state.as_ref().map_or(b.job.stage, |s| s.stage),
                    running: run.is_some_and(|r| r.running()),
                    backend: backend.clone(),
                    queued: self.queue.contains(&b.dir),
                    elsewhere: run.is_none_or(|r| !r.running()) && build::locked(&b.dir),
                    state,
                }
            })
            .collect()
    }

    /// One map is made at a time (they share the model and the card); the next waits its turn. Finished and stopped ones are tidied away.
    fn pump_builds(&mut self) {
        let mut refresh = false;
        let mut finished = Vec::new();
        self.runs.retain(|r| {
            if r.running() {
                return true;
            }
            let s = r.state();
            if s.finished {
                finished.push(r.dir.clone());
                refresh = true;
                return false;
            }
            s.error.is_some() // a stopped build with an error stays, to show it; a paused one just goes
        });
        for dir in finished {
            let name = dir.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            self.notice = Some((format!("{name} is ready."), true));
        }
        // builds only run on the Maps page; the ones stopped for a map go back in the queue when it is shown again
        let on_maps = self.page == Page::Maps && self.opening.is_none();
        if on_maps && !self.paused_for_viewing.is_empty() {
            let back: Vec<PathBuf> = self.paused_for_viewing.drain(..).filter(|d| !self.queue.contains(d)).collect();
            self.queue.splice(0..0, back);
        }
        if on_maps && !self.runs.iter().any(|r| r.running()) {
            while !self.queue.is_empty() {
                let dir = self.queue.remove(0);
                if !dir.exists() {
                    continue;
                }
                self.runs.retain(|r| r.dir != dir);
                if let Some(run) = build::Run::start(dir, self.library.root(), self.engine.hub.clone(), self.wake()) {
                    self.runs.push(run);
                    refresh = true;
                    break;
                }
            }
        }
        // nothing is being made or searched: the model goes (it holds gigabytes on the Apple chip)
        if on_maps && self.queue.is_empty() && !self.runs.iter().any(|r| r.running()) {
            self.engine.hub.unload();
        }
        if refresh || (self.page == Page::Maps && self.runs.iter().any(|r| r.running()) && self.last_refresh.elapsed().as_secs_f32() > 1.5) {
            self.library.refresh();
            self.last_refresh = Instant::now();
        }
    }

    /// Stop a build and wait until it has (a few seconds at most).
    fn stop_build(&mut self, dir: &std::path::Path) -> bool {
        self.queue.retain(|d| d != dir);
        if let Some(run) = self.runs.iter().find(|r| r.dir == dir) {
            run.pause();
            let waited = Instant::now();
            while run.running() && waited.elapsed().as_secs() < 20 {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if run.running() {
                return false;
            }
        }
        self.runs.retain(|r| r.dir != dir);
        true
    }

    fn handle_home(&mut self, action: home::Action) {
        use home::Action;
        match action {
            Action::Open(path) => self.open(path),
            Action::Create { name, sources, every, model, copy_photos } => match build::create(self.library.dir(), &name, &sources, every, &model, copy_photos) {
                Ok(dir) => {
                    self.library.refresh();
                    self.queue.push(dir);
                    self.notice = None;
                }
                Err(e) => self.notice = Some((e, false)),
            },
            Action::Pause(dir) => {
                if let Some(run) = self.runs.iter().find(|r| r.dir == dir) {
                    run.pause();
                }
                self.queue.retain(|d| *d != dir);
            }
            Action::Continue(dir) => {
                self.runs.retain(|r| r.dir != dir || r.running());
                if !self.queue.contains(&dir) && !self.runs.iter().any(|r| r.dir == dir) {
                    self.queue.push(dir);
                }
            }
            Action::Discard(dir) => {
                if !self.runs.iter().any(|r| r.dir == dir && r.running()) && build::locked(&dir) {
                    self.notice = Some(("another window is making this map: stop it there first".into(), false));
                } else if self.stop_build(&dir) {
                    let _ = std::fs::remove_dir_all(&dir);
                    self.library.refresh();
                } else {
                    self.notice = Some(("it is still stopping, try again in a moment".into(), false));
                }
            }
            Action::Delete(path) => {
                if self.map.as_ref().is_some_and(|m| m.path == path) {
                    self.map = None;
                }
                let _ = std::fs::remove_dir_all(&path);
                self.home.forget_cover(&path);
                self.library.refresh();
            }
            Action::Resume(on) => self.library.set_resume_on_start(on),
            Action::InstallRuntime => {
                if let manager::Advice::Offer(offer) = self.runtime.advice() {
                    self.runtime.start(&offer);
                }
            }
            Action::StopInstall => self.runtime.stop(),
            Action::SetBackend(b) => self.runtime.set_backend(b),
            Action::RemoveRuntime => self.runtime.remove_all(),
            Action::GetModel(id) => self.store.get(&id),
            Action::StopModelFetch => self.store.stop(),
            Action::CountModels => self.store.count(),
            Action::RemoveModel(id) => {
                if self.runs.iter().any(|r| r.running()) {
                    self.notice = Some(("stop the maps being made first: they may be using the model".into(), false));
                } else {
                    if self.engine.hub.state().loaded.as_deref() == Some(id.as_str()) {
                        self.engine.hub.unload();
                    }
                    self.store.remove(&id);
                }
            }
            Action::DismissOffer => self.runtime.dismiss(),
            Action::Restart => self.restart = true,
        }
    }

    fn go_to_maps(&mut self) {
        self.page = Page::Maps;
        self.map = None; // the Maps page needs none of it: give back its GPU memory
        self.library.refresh();
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

    /// The match whose copy beside the world is under the mouse: its position among the matches.
    fn pick_copy(&self, screen: Vec2) -> Option<usize> {
        let m = self.map.as_ref()?;
        // the whole cell, not just the picture (a wide photo leaves most of its square empty)
        let half = (m.copy_side / beside::FILL * self.camera.scale * 0.5).max(4.0);
        m.copies.iter().position(|p| {
            let d = (self.camera.to_screen(Vec2::new(p.x, p.y)) - screen).abs();
            d.x <= half && d.y <= half
        })
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
        self.search.clear(); // a search is words or a photo, not both: the photo replaces the words
        true
    }

    /// Enter in the search box, or a photo dropped on the window: the words or the photo (one of them, never both) go to the model,
    /// whose answer comes back through poll_engine. Nothing to search for clears the last search.
    fn ask(&mut self) {
        let text = self.search.trim().to_string();
        // a path to a photo typed or pasted in the box (from a file manager, or dragged on a terminal) is a photo search
        if let Some(path) = photo_path(&text) {
            if self.set_photo(path) {
                self.search.clear();
            }
        }
        let text = self.search.trim().to_string();
        let Some(m) = &mut self.map else { return };
        if text.is_empty() && self.photo.is_none() {
            m.flags.fill(0);
            m.flags_dirty = true;
            m.sims.clear();
            m.trail = None;
            m.show_beside(&self.gpu.queue, Vec::new());
            self.shown.clear();
            self.status.clear();
        } else if m.vectors.is_none() {
            self.status = "this map has no search vectors (embeddings.bin)".into();
        } else if models::spec(m.data.manifest.model()).is_none() {
            self.status = format!("this map was made with {}, which this version doesn't know", m.data.manifest.model());
        } else if !models::spec(m.data.manifest.model()).is_some_and(|s| s.installed(&self.library.root())) {
            let spec = models::spec(m.data.manifest.model()).unwrap();
            self.status = format!("this map was made with {}, whose files are missing (.cache/onnx/{})", spec.label, spec.folder);
        } else if let Some(why) = self.engine.failed() {
            self.status = format!("search can't start: {why}");
        } else {
            let model = m.data.manifest.model().to_string();
            let (query, shown) = match &self.photo {
                Some(path) => (search::Query::Photo(path.clone()), path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
                None => (search::Query::Words(text.clone()), text),
            };
            self.asked = Some(self.engine.ask(&model, query));
            self.shown = shown;
            self.status = if self.engine.ready(&model).is_some() { "searching…".into() } else { "loading the search model, it will run when ready…".into() };
        }
    }

    /// The Layout menu's choice: back to the map's own layout, or the chosen view's words to the model (the photos move once
    /// all their vectors are back, in poll_engine).
    fn arrange(&mut self) {
        let Some(m) = &mut self.map else { return };
        if self.view_choice == View::Map {
            self.layout_asked.clear();
            m.axes_cos = None;
            m.islands.clear();
            if m.view != View::Map {
                let (points, size) = (m.map_points.clone(), m.map_photo_world);
                m.move_to(&self.gpu.queue, View::Map, points, size);
                self.camera.fly_to(self.camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
            }
            return;
        }
        let words: Vec<String> = match self.view_choice {
            View::Axes => {
                if self.axes.all().iter().any(|w| w.trim().is_empty()) {
                    self.status = "the axes need a word at each of the four ends".into();
                    return;
                }
                self.axes.all().iter().map(|w| w.trim().to_string()).collect()
            }
            _ => {
                let words = groups::words(&self.groups_text);
                if words.len() < 2 {
                    self.status = "type at least two words to sort the photos into".into();
                    return;
                }
                words
            }
        };
        if m.vectors.is_none() {
            self.status = "this map has no vectors (embeddings.bin) to arrange the photos by".into();
            return;
        }
        if let Some(why) = self.engine.failed() {
            self.status = format!("the model can't start: {why}");
            return;
        }
        let model = m.data.manifest.model().to_string();
        self.layout_asked = words.iter().map(|w| Some(self.engine.ask(&model, search::Query::Words(w.clone())))).collect();
        self.layout_vectors = vec![None; words.len()];
        self.layout_words = words;
        self.layout_for = self.view_choice;
        self.status = "arranging the photos…".into();
    }

    /// All the words' vectors are back: each photo's cosines with them place it (axes.rs, groups.rs), and the photos slide there.
    fn arranged(&mut self) {
        let Some(m) = &mut self.map else { return };
        let Some(vectors) = &m.vectors else { return };
        let words = std::mem::take(&mut self.layout_vectors);
        let Some(cos) = words.iter().map(|v| v.as_ref().map(|v| vectors.cosines(v))).collect::<Option<Vec<_>>>() else { return };
        if self.layout_for == View::Axes {
            m.axes_cos = Some(cos.try_into().unwrap());
            m.islands.clear();
            self.spread_axes();
            let a = &self.axes;
            self.status = format!("left to right: {} to {}, bottom to top: {} to {}", a.left.trim(), a.right.trim(), a.down.trim(), a.up.trim());
        } else {
            m.axes_cos = None;
            let (points, islands) = groups::place(&cos);
            let size = median_neighbour_distance(&points);
            m.move_to(&self.gpu.queue, View::Groups, points, size);
            self.status = format!("{} groups: each photo is in the one whose word it is most like", islands.len());
            m.islands = islands;
            m.island_words = self.layout_words.clone();
        }
        self.camera.fly_to(self.camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
    }

    /// Place the photos by the axes' cosines (kept from the last Arrange), spread the way the Layout menu says.
    fn spread_axes(&mut self) {
        let Some(m) = &mut self.map else { return };
        let Some(cos) = &m.axes_cos else { return };
        let points = axes::place(cos, self.axes_spread);
        let size = median_neighbour_distance(&points);
        m.move_to(&self.gpu.queue, View::Axes, points, size);
    }

    /// What the text model has said: a query's vector becomes the photos' cosines, shown at once.
    fn poll_engine(&mut self) {
        for event in self.engine.poll() {
            let search::Event::Vector { id, vector, note } = event;
            if let Some(k) = self.layout_asked.iter().position(|&a| a == Some(id)) {
                self.layout_asked[k] = None;
                match vector {
                    Ok(v) => self.layout_vectors[k] = Some(v),
                    Err(e) => {
                        self.layout_asked.clear(); // the other words are no use without this one
                        self.status = format!("couldn't arrange the photos: {e}");
                    }
                }
                if !self.layout_vectors.is_empty() && self.layout_vectors.iter().all(Option::is_some) {
                    self.arranged();
                }
                continue;
            }
            if self.asked != Some(id) {
                continue; // an older query, or the map changed
            }
            self.asked = None;
            let Some(m) = &mut self.map else { continue };
            match (vector, &m.vectors) {
                (Ok(v), Some(vectors)) => {
                    m.sims = vectors.cosines(&v);
                    if let Some((min, max)) = self.apply_search() {
                        self.camera.fly_to(self.camera.fit(min, max));
                    }
                    if let Some(note) = note {
                        self.status = format!("{} - {note}", self.status);
                    }
                }
                (Err(e), _) => self.status = format!("couldn't search: {e}"),
                _ => {}
            }
        }
    }

    /// Turn the last search's cosines into the photos' match bytes (0 = no match, 1..=255 from the cutoff up): the closest
    /// `wanted` photos match. The box the matches and their copies span.
    fn apply_search(&mut self) -> Option<(Vec2, Vec2)> {
        let m = self.map.as_mut()?;
        let n = m.flags.len();
        if n == 0 || m.sims.len() != n {
            return None;
        }
        let score = &m.sims;
        // the closest `wanted` photos match: the cutoff is the similarity of the last of them (ties may add a few)
        let k = (self.wanted.round() as usize).clamp(1, n);
        let mut sorted = score.clone();
        let (_, &mut cut, _) = sorted.select_nth_unstable_by(n - k, |a, b| a.total_cmp(b));
        let best = score.iter().copied().fold(f32::MIN, f32::max);
        let span = (best - cut).max(1e-4); // red at the cutoff, yellow at the best photo
        for i in 0..n {
            m.flags[i] = if score[i] >= cut { 1 + ((score[i] - cut) / span).clamp(0.0, 1.0).mul_add(254.0, 0.5) as u32 } else { 0 };
        }
        let mut order: Vec<usize> = (0..n).filter(|&i| m.flags[i] > 0).collect();
        order.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
        m.trail = None;
        m.show_beside(&self.gpu.queue, order);
        // how many is already in "6 of 100" beside it
        self.status = format!("{}: similarity {cut:.2} to {best:.2}", self.shown);
        m.flags_dirty = true;
        Some(m.beside_box())
    }

    /// The path from photo `a` to `b` along the map's tree (tree.rs) beside the world, in place of a search: its photos are
    /// marked on the map.
    fn show_path(&mut self, a: usize, b: usize) {
        self.path_from = None;
        let Some(m) = &mut self.map else { return };
        let Some(tree) = &m.tree else {
            self.status = "this map was made before paths: make it again to find paths in it".into();
            return;
        };
        let path = tree::route(tree, a, b);
        // the search's words and matches go
        self.search.clear();
        self.photo = None;
        self.photo_tex = None;
        self.shown.clear();
        self.asked = None;
        m.sims.clear();
        m.flags.fill(0);
        for &i in &path {
            m.flags[i] = 255;
        }
        m.flags_dirty = true;
        // its biggest change from photo to photo: the least of all the ways between the two
        let worst = m.vectors.as_ref().map(|v| path.windows(2).map(|w| v.row(w[0]).iter().zip(v.row(w[1])).map(|(x, y)| x * y).sum::<f32>()).fold(1.0f32, f32::min));
        self.status = match worst {
            Some(w) => format!("a path of {} steps; the least alike neighbours on it: similarity {w:.2}", path.len() - 1),
            None => format!("a path of {} steps", path.len() - 1),
        };
        m.trail = Some((a, b));
        m.show_beside(&self.gpu.queue, path);
        self.camera.fly_to(self.camera.fit(m.beside_box().0, m.beside_box().1));
    }

    /// F over a photo: the same as dropping that photo's file on the window (it goes in the search box and is searched).
    fn ask_like(&mut self, photo: usize) {
        let Some(file) = self.map.as_ref().map(|m| m.data.path(photo)) else { return };
        if self.set_photo(file) {
            self.ask();
        }
    }

    /// The next (dir 1) or previous (-1) match, best first, wrapping round: the camera flies to it (zooming in if it is small on screen).
    fn step_match(&mut self, dir: i32) {
        let Some(m) = &mut self.map else { return };
        let found = m.order.len();
        if found == 0 {
            return;
        }
        let k = match m.at {
            None if dir > 0 => 0,
            None => found - 1,
            Some(k) => (k as i32 + dir).rem_euclid(found as i32) as usize,
        };
        self.go_to_match(k);
    }

    /// Fly to the `k`th match (zooming in if its photo is small on screen) and make it the current one.
    fn go_to_match(&mut self, k: usize) {
        let Some(m) = &mut self.map else { return };
        let Some(&photo) = m.order.get(k) else { return };
        m.at = Some(k);
        let p = m.data.points[photo];
        let wanted_scale = STEP_TO * self.window.scale_factor() as f32 / m.photo_world;
        self.camera.fly_to((Vec2::new(p.x, p.y), self.camera.scale.max(wanted_scale)));
    }

    /// Borderless, like mote-control: exclusive forces a modeset and Wayland ignores it.
    fn toggle_fullscreen(&self) {
        let on = self.window.fullscreen().is_none();
        self.window.set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
    }

    fn key(&mut self, key: &Key, down: bool) {
        if let (true, Key::Named(NamedKey::F11)) = (down, key) {
            return self.toggle_fullscreen();
        }
        if self.page != Page::Viewer {
            return;
        }
        if let (true, Key::Named(NamedKey::Escape)) = (down, key) {
            if self.menu.is_some() {
                self.menu = None;
            } else if self.details.is_some() {
                self.details = None;
            } else if self.path_from.is_some() {
                self.path_from = None;
                self.status.clear();
            } else if self.help_open {
                self.help_open = false;
            } else {
                self.go_to_maps();
            }
            return;
        }
        if let (true, Key::Character(c)) = (down, key) {
            match c.as_str() {
                "n" | "N" | "]" => return self.step_match(1),
                "p" | "P" | "[" => return self.step_match(-1),
                "h" | "H" => {
                    self.help_open = !self.help_open;
                    return;
                }
                _ => {}
            }
        }
        if let Key::Character(c) = key {
            if c.eq_ignore_ascii_case("f") {
                if let (true, Some(i)) = (down, self.hovered) {
                    self.ask_like(i);
                }
                return;
            }
        }
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
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Right, .. } if !over_ui && self.page == Page::Viewer => {
                let ppp = self.window.scale_factor() as f32;
                self.menu = self.hovered.zip(self.cursor).map(|(i, c)| (i, egui::pos2(c.x / ppp, c.y / ppp)));
            }
            WindowEvent::MouseInput { state, button: button @ (MouseButton::Left | MouseButton::Middle), .. } => match state {
                ElementState::Pressed if !over_ui && self.page == Page::Viewer => {
                    self.menu = None;
                    self.details = None;
                    self.camera.press();
                }
                ElementState::Released if self.camera.dragging() => {
                    let click = self.camera.drag_distance() < 4.0;
                    self.camera.release();
                    if click && button == MouseButton::Left {  // the middle button only pans
                        self.click_at();
                    }
                }
                _ => {}
            },
            WindowEvent::MouseWheel { delta, phase, .. } => {
                // a trackpad's two fingers pan (from the first touch to the end of the system's own momentum); a wheel zooms
                match phase {
                    winit::event::TouchPhase::Started => self.scroll_gesture = true,
                    winit::event::TouchPhase::Ended | winit::event::TouchPhase::Cancelled => self.scroll_gesture = false,
                    winit::event::TouchPhase::Moved => {}
                }
                if !over_ui && self.page == Page::Viewer {
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => self.camera.scroll_lines(y),
                        MouseScrollDelta::PixelDelta(p) if self.scroll_gesture => self.camera.pan_pixels(Vec2::new(p.x as f32, p.y as f32)),
                        MouseScrollDelta::PixelDelta(p) => self.camera.scroll_pixels(p.y as f32),
                    }
                }
            }
            WindowEvent::PinchGesture { delta, .. } if !over_ui && self.page == Page::Viewer => self.camera.pinch(delta as f32),
            WindowEvent::ModifiersChanged(m) => self.alt = m.state().alt_key(),
            WindowEvent::KeyboardInput { event, .. } => {
                if self.alt && event.state == ElementState::Pressed && !event.repeat && event.logical_key == Key::Named(NamedKey::Enter) {
                    return self.toggle_fullscreen();
                }
                let typing = self.ui.ctx.egui_wants_keyboard_input() && event.state == ElementState::Pressed;
                if !typing && !egui_took {
                    self.key(&event.logical_key, event.state == ElementState::Pressed);
                } else if event.state == ElementState::Released {
                    self.key(&event.logical_key, false); // never leave a key stuck down
                }
            }
            WindowEvent::HoveredFile(_) if self.page == Page::Maps => {
                self.home.set_drag_over(true);
                self.window.request_redraw();
            }
            WindowEvent::HoveredFileCancelled => {
                self.home.set_drag_over(false);
                self.window.request_redraw();
            }
            // one event per dropped file: kept until the next frame, then handed over together
            WindowEvent::DroppedFile(path) if self.page == Page::Maps => {
                self.dropped.push(path);
                self.window.request_redraw();
            }
            WindowEvent::DroppedFile(path) if self.page == Page::Viewer => {
                if path.extension().and_then(|e| e.to_str()).is_some_and(|e| PHOTO_TYPES.contains(&e.to_lowercase().as_str())) {
                    if self.set_photo(path) {
                        self.ask();
                    }
                } else {
                    self.status = "that isn't a photo I can search with (jpg, png, webp, gif, bmp)".into();
                }
            }
            WindowEvent::Focused(false) => self.keys = Keys::default(),
            WindowEvent::Focused(true) => self.window.request_redraw(),
            WindowEvent::Occluded(hidden) => {
                self.hidden = hidden;
                self.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn frame(&mut self) {
        if !self.hidden {
            self.window.request_redraw(); // first, so that the frames go on when the surface can't be drawn to
        }
        let size = self.window.inner_size();
        if (size.width, size.height) != (self.gpu.config.width, self.gpu.config.height) && size.width > 0 && size.height > 0 {
            self.gpu.resize(size.width, size.height);
            self.camera.viewport = Vec2::new(size.width as f32, size.height as f32);
        }
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_times.push(dt);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        self.camera.update(dt, self.cursor, self.keys);
        self.frames += 1;
        self.poll_engine();
        self.runtime.tick();
        self.store.tick();
        self.pump_builds();
        if self.opening.as_ref().is_some_and(|(_, job, _)| job.is_finished()) {
            let (path, job, _) = self.opening.take().unwrap();
            match job.join() {
                Ok(Ok(data)) => self.opened(path, data),
                Ok(Err(e)) => eprintln!("can't open {}: {e}", path.display()),
                Err(_) => eprintln!("can't open {}: reading it crashed", path.display()),
            }
        }
        let over_ui = self.ui.ctx.is_pointer_over_egui();
        self.hovered_copy = match self.cursor {
            Some(c) if self.page == Page::Viewer && !self.camera.dragging() && !over_ui => self.pick_copy(c),
            _ => None,
        };
        self.hovered = match (self.cursor, self.hovered_copy, &self.map) {
            (_, Some(k), Some(m)) => Some(m.order[k]),
            (Some(c), _, _) if self.page == Page::Viewer && !self.camera.dragging() && !over_ui => self.pick(c),
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
            slide: self.map.as_ref().map_or(1.0, OpenMap::slide),
            copy_px: self.map.as_ref().map_or(0.0, |m| m.copy_side * self.camera.scale),
        };
        match &mut self.map {
            Some(m) if self.page == Page::Viewer => {
                if m.flags_dirty {
                    m.renderer.upload_flags(&self.gpu.queue, &m.flags);
                    m.flags_dirty = false;
                }
                let copies = (&m.order[..], &m.copies[..], cam.copy_px);
                m.streamer.update(&self.gpu.queue, &m.data, view_min, view_max, self.camera.center, size_px, copies);
                m.renderer.draw(&self.gpu.queue, &mut encoder, &view, cam, self.smooth);
            }
            _ => gpu::clear(&mut encoder, &view),
        }
        self.draw_ui(&mut encoder, &view);
        self.gpu.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
        self.window.request_redraw();
    }

    /// A map being opened: one card in the middle.
    fn opening_card(&self, ctx: &egui::Context) {
        let Some((path, _, progress)) = &self.opening else { return };
        let (text, fraction) = progress.now();
        let name = path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
        egui::Area::new("opening".into()).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            theme::card().inner_margin(egui::Margin::same(20)).show(ui, |ui| {
                ui.set_width(340.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.heading(format!("Opening {name}"));
                });
                ui.add_space(4.0);
                ui.label(theme::muted(text));
                if let Some(f) = fraction {
                    ui.add_space(4.0);
                    ui.add(egui::ProgressBar::new(f).desired_width(ui.available_width()).show_percentage().fill(theme::ACCENT));
                }
            });
        });
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
            self.ui.ctx.memory_mut(|m| m.request_focus(egui::Id::new("search-box"))); // ready to type
        }
        if actions.clear_search {
            self.search.clear();
            self.ask(); // nothing to search for: the matches go
        }
        if let Some(i) = actions.path_from {
            self.path_from = Some(i);
            self.status = "now right-click the photo the path goes to (Esc: cancel)".into();
        }
        if let (Some(a), Some(b)) = (self.path_from, actions.path_to) {
            self.show_path(a, b);
        }
        if let Some(i) = actions.like {
            self.ask_like(i);
        }
        if actions.ask {
            self.ask();
        }
        if actions.arrange {
            self.arrange();
        }
        if actions.respread {
            self.spread_axes();
        }
        if actions.reblend {
            self.apply_search();
        }
        if actions.step != 0 {
            self.step_match(actions.step);
        }
        if actions.back {
            self.go_to_maps();
        }
        for a in std::mem::take(&mut actions.home) {
            self.handle_home(a);
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
    fn draw_labels(&mut self, ctx: &egui::Context) {
        self.label_rects.clear();
        let Some(m) = &self.map else { return };
        match m.view {
            View::Axes => return self.draw_axis_words(ctx),
            View::Groups => return self.draw_island_words(ctx),
            View::Map => {}
        }
        let mut shown: Vec<(egui::Rect, usize, usize)> = Vec::new();
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
            for (index, l) in labels.iter().enumerate() {
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
                if alpha > 0.3 {
                    shown.push((rect, level, index));
                }
                // the name under the mouse is a button: it goes white and underlined
                let under_mouse = alpha > 0.3 && self.cursor.is_some_and(|c| rect.contains(egui::pos2(c.x / ppp, c.y / ppp)));
                let colour = if under_mouse { egui::Color32::WHITE } else { colour };
                let outline = egui::Color32::from_black_alpha((220.0 * alpha) as u8);
                for (dx, dy) in [(-1.5, 0.0), (1.5, 0.0), (0.0, -1.5), (0.0, 1.5), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (1.0, -1.0)] {
                    painter.galley_with_override_text_color(rect.min + egui::vec2(dx, dy), galley.clone(), outline);
                }
                painter.galley_with_override_text_color(rect.min, galley, colour.gamma_multiply(alpha));
                if under_mouse {
                    painter.line_segment([rect.left_bottom(), rect.right_bottom()], egui::Stroke::new(2.0, colour));
                }
            }
        }
        self.label_rects = shown;
    }

    /// The Axes view's four words, at the middle of each edge of the photos (kept on the screen when that edge is not).
    fn draw_axis_words(&self, ctx: &egui::Context) {
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "axis words".into()));
        let ppp = ctx.pixels_per_point();
        let screen = ctx.content_rect().shrink(16.0);
        let a = &self.axes;
        let ends = [
            (a.left.trim().to_string(), Vec2::new(0.0, WORLD / 2.0), egui::Align2::RIGHT_CENTER),
            (a.right.trim().to_string(), Vec2::new(WORLD, WORLD / 2.0), egui::Align2::LEFT_CENTER),
            (a.down.trim().to_string(), Vec2::new(WORLD / 2.0, WORLD), egui::Align2::CENTER_TOP),
            (a.up.trim().to_string(), Vec2::new(WORLD / 2.0, 0.0), egui::Align2::CENTER_BOTTOM),
        ];
        for (text, world, align) in ends {
            let galley = painter.layout_no_wrap(text, egui::FontId::proportional(22.0), egui::Color32::from_rgb(255, 220, 120));
            let at = self.camera.to_screen(world) / ppp;
            // just outside the photos' edge, then moved in as far as needed to stay on the screen
            let rect = align.anchor_size(egui::pos2(at.x, at.y) + align.to_sign() * -10.0, galley.size());
            let shift = egui::vec2(
                (screen.left() - rect.left()).max(0.0) + (screen.right() - rect.right()).min(0.0),
                (screen.top() - rect.top()).max(0.0) + (screen.bottom() - rect.bottom()).min(0.0),
            );
            let rect = rect.translate(shift);
            painter.rect_filled(rect.expand2(egui::vec2(8.0, 3.0)), 6.0, egui::Color32::from_black_alpha(170));
            painter.galley(rect.min, galley, egui::Color32::WHITE);
        }
    }

    /// The Groups view's words, each above its island with how many photos it has.
    fn draw_island_words(&self, ctx: &egui::Context) {
        let Some(m) = &self.map else { return };
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "group words".into()));
        let ppp = ctx.pixels_per_point();
        for island in &m.islands {
            let Some(word) = m.island_words.get(island.word) else { continue };
            // no wider than the island on screen (cut with "…"), so neighbours' names never run into each other
            let width = island.side * self.camera.scale / ppp;
            if width < 24.0 {
                continue;
            }
            let mut job = egui::text::LayoutJob::default();
            job.append(word, 0.0, egui::TextFormat::simple(egui::FontId::proportional(20.0), egui::Color32::from_rgb(255, 220, 120)));
            job.append(&format!("  {}", island.photos), 0.0, egui::TextFormat::simple(egui::FontId::proportional(15.0), theme::MUTED));
            job.wrap.max_width = width;
            job.wrap.max_rows = 1;
            job.wrap.break_anywhere = true;
            job.wrap.overflow_character = Some('…');
            let galley = painter.layout_job(job);
            let at = self.camera.to_screen(Vec2::new(island.min.x + island.side / 2.0, island.min.y)) / ppp;
            let rect = egui::Align2::CENTER_BOTTOM.anchor_size(egui::pos2(at.x, at.y - 8.0), galley.size());
            painter.rect_filled(rect.expand2(egui::vec2(8.0, 3.0)), 6.0, egui::Color32::from_black_alpha(170));
            painter.galley(rect.min, galley, egui::Color32::WHITE);
        }
    }

    /// Fly to a region so that it fills the screen.
    fn go_to_region(&mut self, level: usize, index: usize) {
        let Some(l) = self.map.as_ref().and_then(|m| m.data.regions.levels.get(level)?.get(index)) else { return };
        let scale = self.camera.viewport.min_element() / (l.spread * 1.8).max(1.0);
        self.camera.fly_to((Vec2::new(l.x, l.y), scale));
    }

    /// A click on the map (a press that hardly moved): on a match's copy beside the world, it flies to the photo; on a region
    /// name, it zooms to the region.
    fn click_at(&mut self) {
        let Some(c) = self.cursor else { return };
        if let Some(k) = self.pick_copy(c) {
            return self.go_to_match(k);
        }
        let ppp = self.window.scale_factor() as f32;
        let at = egui::pos2(c.x / ppp, c.y / ppp);
        if let Some(&(_, level, index)) = self.label_rects.iter().find(|(rect, _, _)| rect.contains(at)) {
            self.go_to_region(level, index);
        }
    }

    /// Over the matches' copies beside the world (or a path's photos): what they are, above them; the copy under the mouse
    /// outlines its photo on the map and a match under the mouse outlines its copy; the current match's copy (Next, Prev) is
    /// outlined too. A path is a line from photo to photo on the map; a path's chosen start is outlined until its end is.
    fn draw_beside(&self, ctx: &egui::Context) {
        let Some(m) = &self.map else { return };
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "beside".into()));
        let ppp = ctx.pixels_per_point();
        let to_screen = |p: data::Point| {
            let s = self.camera.to_screen(Vec2::new(p.x, p.y)) / ppp;
            egui::pos2(s.x, s.y)
        };
        // a photo's outline: `side` its longer side on screen (points), never smaller than a few points so it can be found
        let outline = |i: usize, at: egui::Pos2, side: f32, width: f32| {
            let shape = m.data.shape(i);
            let size = (egui::vec2(shape.x, shape.y) * side).max(egui::vec2(8.0, 8.0)) + egui::vec2(6.0, 6.0);
            painter.rect_stroke(egui::Rect::from_center_size(at, size), 3.0, egui::Stroke::new(width, egui::Color32::WHITE), egui::StrokeKind::Outside);
        };
        let (photo_side, copy_side) = (self.size_px() / ppp, m.copy_side * self.camera.scale / ppp);
        if let Some(i) = self.path_from {
            outline(i, to_screen(m.data.points[i]), photo_side, 3.0);
        }
        let Some(first) = m.copies.first() else { return };
        if m.trail.is_some() {
            let line: Vec<egui::Pos2> = m.order.iter().map(|&i| to_screen(m.data.points[i])).collect();
            painter.add(egui::Shape::line(line, egui::Stroke::new(2.0, egui::Color32::from_white_alpha(170))));
        }
        let top = self.camera.to_screen(Vec2::new(beside::LEFT, first.y - m.copy_side / 2.0)) / ppp;
        let heading = match m.trail {
            Some(_) => format!("A path of {} photos, from the first to the last", m.order.len()),
            None => format!("{} matches, best first", m.order.len()),
        };
        let galley = painter.layout_no_wrap(heading, egui::FontId::proportional(16.0), theme::MUTED);
        painter.galley(egui::pos2(top.x, top.y - galley.size().y - 6.0), galley, theme::MUTED);
        if let Some(k) = m.at {
            outline(m.order[k], to_screen(m.copies[k]), copy_side, 2.0);
        }
        match (self.hovered_copy, self.hovered) {
            (Some(k), _) => outline(m.order[k], to_screen(m.data.points[m.order[k]]), photo_side, 2.0),
            (None, Some(i)) => {
                if let Some(k) = m.order.iter().position(|&o| o == i) {
                    outline(i, to_screen(m.copies[k]), copy_side, 2.0);
                }
            }
            _ => {}
        }
    }

    /// A photo whose file can't be found (moved, deleted, or on a disk not plugged in) is darkened once it is big enough to want it,
    /// with "not found" on it: what shows is its small picture made when the map was, blown up.
    fn draw_missing(&self, ctx: &egui::Context) {
        let Some(m) = &self.map else { return };
        let ppp = ctx.pixels_per_point();
        let long = self.size_px() / ppp;
        if long < 48.0 {
            return;
        }
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "missing photos".into()));
        let (view_min, view_max) = (self.camera.to_world(Vec2::ZERO), self.camera.to_world(self.camera.viewport));
        m.data.grid.within(&m.data.points, view_min, view_max, |i| {
            if !m.streamer.missing(i) {
                return;
            }
            let at = self.camera.to_screen(Vec2::new(m.data.points[i].x, m.data.points[i].y)) / ppp;
            let size = m.data.shape(i) * long;
            let rect = egui::Rect::from_center_size(egui::pos2(at.x, at.y), egui::vec2(size.x, size.y));
            painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(190));
            if size.x >= 70.0 {
                painter.text(rect.center(), egui::Align2::CENTER_CENTER, "not found", egui::FontId::proportional((size.x / 8.0).clamp(11.0, 18.0)), egui::Color32::from_gray(200));
            }
        });
    }

    /// Photos zoomed in far enough show their own caption (else their title) neatly above them: only the photos in view
    /// and only when they are big, so it costs nothing when zoomed out. A caption that would overlap one already drawn is skipped.
    fn draw_photo_captions(&self, ctx: &egui::Context) {
        let Some(m) = &self.map else { return };
        let ppp = ctx.pixels_per_point();
        let long = self.size_px() / ppp; // a photo's longer side, points
        let alpha = ((long - CAPTION_FROM.0) / (CAPTION_FROM.1 - CAPTION_FROM.0)).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return;
        }
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "photo captions".into()));
        let screen = ctx.content_rect();
        let (view_min, view_max) = (self.camera.to_world(Vec2::ZERO), self.camera.to_world(self.camera.viewport));
        let mut shown: Vec<(egui::Pos2, egui::Rect, std::sync::Arc<egui::Galley>)> = Vec::new();
        let mut taken: Vec<egui::Rect> = Vec::new();
        let centre = Vec2::new(0.5 * (view_min.x + view_max.x), 0.5 * (view_min.y + view_max.y));
        let mut visible = Vec::new();
        m.data.grid.within(&m.data.points, view_min, view_max, |i| visible.push(i));
        visible.sort_by(|&a, &b| {  // nearest the middle first: those win when captions would collide
            let d = |i: usize| (Vec2::new(m.data.points[i].x, m.data.points[i].y) - centre).length_squared();
            d(a).total_cmp(&d(b))
        });
        for i in visible.into_iter().take(200) {
            let photo = &m.data.photos[i];
            let text = (if photo.caption.trim().is_empty() { &photo.title } else { &photo.caption }).trim().to_string();
            // while a search is showing: the photo's similarity to it first, in the colour of its match (grey if it is no match)
            let score = m.sims.get(i).map(|&sim| (sim, m.flags[i]));
            if text.is_empty() && score.is_none() {
                continue;
            }
            let shape = m.data.shape(i);
            let at = self.camera.to_screen(Vec2::new(m.data.points[i].x, m.data.points[i].y)) / ppp;
            let (w, h) = (shape.x * long, shape.y * long);
            let font = egui::FontId::proportional(16.0);
            let mut job = egui::text::LayoutJob::default();
            if let Some((sim, flag)) = score {
                let colour = if flag == 0 {
                    egui::Color32::from_gray(150)
                } else {
                    let t = (flag - 1) as f32 / 254.0; // as in map.wgsl: COLD to HOT
                    egui::Color32::from_rgb((255.0 * (0.9 + 0.1 * t)) as u8, (255.0 * (0.15 + 0.77 * t)) as u8, (255.0 * (0.15 + 0.10 * t)) as u8)
                };
                job.append(&format!("{sim:.2}"), 0.0, egui::TextFormat::simple(font.clone(), colour));
                if !text.is_empty() {
                    job.append("  ", 0.0, egui::TextFormat::simple(font.clone(), egui::Color32::WHITE));
                }
            }
            job.append(&text, 0.0, egui::TextFormat::simple(font, egui::Color32::WHITE));
            job.wrap.max_width = w.max(140.0);
            job.wrap.max_rows = 3;
            job.wrap.overflow_character = Some('…');
            job.halign = egui::Align::Center;
            let galley = painter.layout_job(job);
            // the text itself (the galley's own rect is as wide as the wrap width), centred over the photo, its bottom just above it
            let text_box = galley.rows.iter().fold(egui::Rect::NOTHING, |a, r| a.union(r.rect()));
            let origin = egui::pos2(at.x - text_box.center().x, at.y - h * 0.5 - 6.0 - text_box.max.y);
            let rect = text_box.translate(origin.to_vec2());
            if !screen.intersects(rect) || taken.iter().any(|t| t.intersects(rect.expand(3.0))) {
                continue;
            }
            taken.push(rect);
            shown.push((origin, rect, galley));
        }
        for (origin, rect, galley) in shown {
            painter.rect_filled(rect.expand2(egui::vec2(5.0, 2.0)), 4.0, egui::Color32::from_black_alpha((170.0 * alpha) as u8));
            painter.galley(origin, galley, egui::Color32::WHITE.gamma_multiply(alpha));
        }
    }

    fn build_ui(&mut self, ui: &mut egui::Ui, actions: &mut UiActions) {
        let ctx = ui.ctx().clone();
        if self.page == Page::Maps {
            let dropped = std::mem::take(&mut self.dropped);
            if !dropped.is_empty() && !self.home.dropped(dropped) {
                self.notice = Some(("that isn't a folder or a photo I can use (jpg, png, webp, gif, bmp)".into(), false));
            }
            let views = self.build_views();
            let notice = self.notice.clone();
            let ready_msg = notice.as_ref().filter(|(_, good)| *good).map(|(m, _)| m.as_str());
            actions.home = self.home.show(&ctx, &self.library, &views, self.library.resume_on_start(), ready_msg, &self.runtime, &self.store);
            if let Some((text, false)) = &notice {
                egui::Area::new("notice".into()).anchor(egui::Align2::CENTER_BOTTOM, [0.0, -24.0]).show(&ctx, |ui| {
                    theme::card().inner_margin(egui::Margin::symmetric(14, 10)).show(ui, |ui| ui.label(egui::RichText::new(text).color(egui::Color32::from_rgb(235, 110, 100))));
                });
            }
            self.opening_card(&ctx);
            return;
        }
        if SHOW_REGION_NAMES {
            self.draw_labels(&ctx);
        }
        self.draw_missing(&ctx);
        self.draw_photo_captions(&ctx);
        self.draw_beside(&ctx);

        let width = (ctx.content_rect().width() - 24.0).clamp(320.0, 760.0);
        egui::Area::new("top".into()).anchor(egui::Align2::CENTER_TOP, [0.0, 12.0]).show(&ctx, |ui| {
            theme::card().show(ui, |ui| {
                ui.set_width(width - 24.0);
                // row 1: the map, the search (takes what is left), whether search is up, options
                ui.horizontal(|ui| {
                    // the way out, set apart from the search: a back arrow, a divider, then which map this is
                    let (rect, back) = ui.allocate_exact_size(egui::vec2(28.0, 28.0), egui::Sense::click());
                    let colour = if back.hovered() { egui::Color32::WHITE } else { theme::MUTED };
                    if back.hovered() {
                        ui.painter().rect_filled(rect, theme::RADIUS_SMALL, egui::Color32::from_white_alpha(14));
                    }
                    let c = rect.center();
                    let stroke = egui::Stroke::new(1.8, colour);
                    ui.painter().line_segment([c + egui::vec2(6.0, 0.0), c + egui::vec2(-6.0, 0.0)], stroke);
                    ui.painter().line_segment([c + egui::vec2(-6.0, 0.0), c + egui::vec2(-1.0, -5.0)], stroke);
                    ui.painter().line_segment([c + egui::vec2(-6.0, 0.0), c + egui::vec2(-1.0, 5.0)], stroke);
                    if back.on_hover_text("back to your maps (Esc)").clicked() {
                        actions.back = true;
                    }
                    let (line, _) = ui.allocate_exact_size(egui::vec2(1.0, 20.0), egui::Sense::hover());
                    ui.painter().rect_filled(line, 0.0, theme::CARD_STROKE);
                    if let Some(m) = &self.map {
                        ui.label(theme::muted(&m.data.manifest.name));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // stays open while you change things: it closes on a click outside it
                        let stay_open = egui::containers::menu::MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
                        egui::containers::menu::MenuButton::new("Options").config(stay_open).ui(ui, |ui| {
                            ui.set_width(280.0);
                            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                                ui.label(theme::muted("BEST MATCHES TO MARK"));
                                let most = self.map.as_ref().map_or(1000.0, |m| m.flags.len() as f32).max(11.0);
                                if ui.add(egui::Slider::new(&mut self.wanted, 10.0..=most).logarithmic(true).integer()).changed() {
                                    actions.reblend = true;
                                }
                                ui.label(theme::muted("Always marks exactly this many photos, best first, from yellow (strong) to red (weak). It does not judge whether they are good matches."));
                                ui.label(theme::muted("Too few: good matches beyond the count are missed. Too many: poor matches get marked too."));
                                ui.add_space(8.0);
                                ui.label(theme::muted("PHOTO PIXELS"));
                                theme::segmented(ui, &mut self.smooth, &[(true, "Smooth", "pixels blended with their neighbours"), (false, "Nearest", "each pixel as it is: sharp, blocky up close")]);
                            });
                        });
                        egui::containers::menu::MenuButton::new("Layout").config(egui::containers::menu::MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)).ui(ui, |ui| {
                            ui.set_width(300.0);
                            ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                                ui.label(theme::muted("PLACE THE PHOTOS"));
                                let options = [
                                    (View::Map, "Map", "similar photos together, as the map was made"),
                                    (View::Axes, "Axes", "along two pairs of words you choose"),
                                    (View::Groups, "Groups", "sorted into groups by words you choose"),
                                ];
                                if theme::segmented(ui, &mut self.view_choice, &options) && self.view_choice == View::Map {
                                    actions.arrange = true;
                                }
                                if self.view_choice == View::Map {
                                    ui.label(theme::muted("Photos that look alike sit close together, in regions with names."));
                                } else if self.view_choice == View::Axes {
                                    ui.label(theme::muted("Left to right: from the first word to the second. Bottom to top: the same with the other pair. Each photo goes by how much more it is like one end than the other."));
                                    ui.add_space(6.0);
                                    let mut enter = false;
                                    let a = &mut self.axes;
                                    for (label, first, second, hints) in [("across", &mut a.left, &mut a.right, ("left: day", "right: night")), ("up", &mut a.down, &mut a.up, ("bottom: nature", "top: city"))] {
                                        ui.horizontal(|ui| {
                                            ui.add_sized([46.0, 20.0], egui::Label::new(theme::muted(label)));
                                            for (text, hint) in [(first, hints.0), (second, hints.1)] {
                                                let edit = ui.add(egui::TextEdit::singleline(text).hint_text(hint).margin(egui::Margin::symmetric(8, 5)).desired_width(112.0));
                                                enter |= edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                                            }
                                        });
                                    }
                                    ui.add_space(6.0);
                                    let spreads = [
                                        (axes::Spread::Even, "Even", "every photo an equal share of the width: shows their order, not how far apart they are"),
                                        (axes::Spread::Score, "By score", "places follow the scores: shows how far apart photos are, most bunch in the middle"),
                                    ];
                                    if theme::segmented(ui, &mut self.axes_spread, &spreads) {
                                        actions.respread = true;
                                    }
                                    ui.label(theme::muted(match self.axes_spread {
                                        axes::Spread::Even => "Even: photos fill the square, in order. Close scores can still end up far apart.",
                                        axes::Spread::Score => "By score: distance means difference. Most photos crowd the middle.",
                                    }));
                                    ui.add_space(6.0);
                                    if theme::button(ui, egui::Button::new("Arrange")).on_hover_text("place the photos along these words (Enter in a box does it too)").clicked() || enter {
                                        actions.arrange = true;
                                    }
                                } else {
                                    ui.label(theme::muted("Each photo joins the word it is most like. In each group the strongest matches are in the middle, the weakest at the edge."));
                                    ui.add_space(6.0);
                                    ui.add(egui::TextEdit::multiline(&mut self.groups_text).hint_text("beach, mountain, city, forest").desired_rows(3).desired_width(f32::INFINITY));
                                    ui.label(theme::muted("Separate the words with commas or new lines."));
                                    ui.add_space(6.0);
                                    if theme::button(ui, egui::Button::new("Arrange")).on_hover_text("sort the photos into these groups").clicked() {
                                        actions.arrange = true;
                                    }
                                }
                            });
                        });
                        // whether a search can run now
                        let searchable = self.map.as_ref().is_some_and(|m| m.vectors.is_some());
                        let map_model = self.map.as_ref().map(|m| m.data.manifest.model().to_string());
                        let hint = map_model.as_deref().and_then(models::spec).map_or("Search the map: describe what you look for, or drop in a photo", |s| s.query_hint);
                        if let Some(why) = self.engine.failed() {
                            ui.label(theme::muted("search off")).on_hover_text(why);
                            theme::dot(ui, theme::MUTED);
                        } else if self.map.is_some() && !searchable {
                            ui.label(theme::muted("map not searchable")).on_hover_text("it has no embeddings.bin of the right size");
                            theme::dot(ui, theme::MUTED);
                        } else if let Some(gpu) = map_model.as_deref().and_then(|m| self.engine.ready(m)) {
                            ui.label(theme::muted(if gpu { "search ready" } else { "search ready (CPU)" })).on_hover_text("The AI model that turns your words or photo into numbers is loaded. It runs on this computer, nothing is sent anywhere.");
                            theme::dot(ui, theme::OK);
                        } else if self.map.is_some() {
                            ui.label(theme::muted("search starting"));
                            ui.spinner();
                        }
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                            if let Some(photo) = &self.photo {
                                // a search is a photo or words, never both: while a photo is the search it takes the place of the box
                                let name = photo.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                                egui::Frame::new()
                                    .fill(egui::Color32::from_rgb(30, 33, 42))
                                    .stroke(egui::Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.6)))
                                    .corner_radius(6.0)
                                    .inner_margin(egui::Margin::symmetric(8, 4))
                                    .show(ui, |ui| {
                                        ui.set_width(ui.available_width() - 2.0);
                                        ui.horizontal(|ui| {
                                            if let Some(tex) = &self.photo_tex {
                                                ui.add(egui::Image::new((tex.id(), egui::vec2(28.0 * tex.aspect_ratio(), 28.0))).corner_radius(4));
                                            }
                                            ui.label(format!("Photos like {name}"));
                                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                if ui.button("× Search with words instead").on_hover_text("stop searching by this photo, clear the matches and type words").clicked() {
                                                    actions.clear_photo = true;
                                                }
                                            });
                                        });
                                    });
                            } else {
                                // a cross after the box while there is something to clear: the words and the matches
                                let clearable = !self.search.is_empty() || self.map.as_ref().is_some_and(|m| !m.order.is_empty());
                                // the cross, the gap before it and a gap after it (the status beside it would touch it)
                                let cross = if clearable { 24.0 + ui.spacing().item_spacing.x + 8.0 } else { 0.0 };
                                let edit = ui.add(
                                    egui::TextEdit::singleline(&mut self.search)
                                        .id(egui::Id::new("search-box"))
                                        .hint_text(hint)
                                        .margin(egui::Margin::symmetric(10, 6))
                                        .desired_width(ui.available_width() - cross),
                                );
                                if edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    actions.ask = true;
                                }
                                if clearable && ui.add_sized([24.0, edit.rect.height()], egui::Button::new("×")).on_hover_text("clear the search").clicked() {
                                    actions.clear_search = true;
                                }
                                if clearable {
                                    ui.add_space(8.0);
                                }
                            }
                        });
                    });
                });
                // row 2, once there is something to say: the photo searched with, the matches and how to go through them, the status
                let (found, at, on_path) = self.map.as_ref().map_or((0, None, false), |m| (m.order.len(), m.at, m.trail.is_some()));
                if found > 0 || !self.status.is_empty() {
                    ui.add_space(2.0);
                    ui.horizontal(|ui| {
                        if found > 0 {
                            if ui.button("Prev").on_hover_text("the previous match (P or [)").clicked() {
                                actions.step = -1;
                            }
                            if ui.button("Next").on_hover_text("the next match, best first (N or ])").clicked() {
                                actions.step = 1;
                            }
                            ui.label(at.map_or(format!("{found} {}", if on_path { "photos" } else { "matches" }), |k| format!("{} of {found}", k + 1)));
                        }
                        if !self.status.is_empty() {
                            ui.label(theme::muted(&self.status));
                        }
                    });
                }
            });
        });

        self.opening_card(&ctx);

        // bottom left: a "?" that opens how the map works and the controls
        egui::Area::new("help".into()).anchor(egui::Align2::LEFT_BOTTOM, [12.0, -12.0]).show(&ctx, |ui| {
            let button = egui::Button::new(egui::RichText::new("?").strong()).min_size(egui::vec2(28.0, 28.0)).selected(self.help_open);
            if ui.add(button).on_hover_text("how this works (H)").clicked() {
                self.help_open = !self.help_open;
            }
        });
        if self.help_open {
            egui::Area::new("help-card".into()).anchor(egui::Align2::LEFT_BOTTOM, [12.0, -52.0]).show(&ctx, |ui| {
                {
                    theme::card().show(ui, |ui| {
                        ui.set_max_width(400.0);
                        ui.label(egui::RichText::new("How the map works").size(17.0));
                        ui.add_space(4.0);
                        ui.label(theme::muted(
                            "Every photo is turned into numbers that describe what it shows, and photos that look alike are placed close together. \
                             Zoom in and similar photos sit side by side; the names mark regions of the map. \
                             Search compares your words, or a photo you drop in (one or the other), with every photo and highlights the best matches.",
                        ));
                        ui.add_space(8.0);
                        egui::Grid::new("keys").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                            for (keys, what) in [
                                (&["scroll", "Q", "E"][..], "zoom"),
                                (&["drag", "W", "A", "S", "D"][..], "move"),
                                (&["F"][..], "photos like the one under the mouse"),
                                (&["right-click"][..], "a path of similar photos from one photo to another"),
                                (&["N", "P"][..], "next / previous match"),
                                (&["F11", "Alt", "Enter"][..], "fullscreen"),
                                (&["Esc"][..], "back to your maps"),
                                (&["H"][..], "show / hide this help"),
                            ] {
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 4.0;
                                    for k in keys {
                                        theme::key(ui, k);
                                    }
                                });
                                ui.label(theme::muted(what));
                                ui.end_row();
                            }
                        });
                    });
                }
            });
        }
        let avg = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        if let Some(m) = &self.map {
            egui::Area::new("fps".into()).anchor(egui::Align2::RIGHT_BOTTOM, [-14.0, -14.0]).show(&ctx, |ui| {
                let detail = format!(
                    "{:.1} ms a frame\nstreamed: {}\nloading: {}",
                    avg * 1000.0,
                    m.streamer.usage().iter().map(|(s, used, total)| format!("{s} px {used}/{total}")).collect::<Vec<_>>().join(", "),
                    m.streamer.in_flight()
                );
                let speed = egui::RichText::new(format!("{:.0} fps", 1.0 / avg.max(1e-6))).monospace().color(theme::MUTED);
                ui.add(egui::Label::new(speed).sense(egui::Sense::hover())).on_hover_text(detail);
            });
        }

        if let (Some((photo, at)), Some(m)) = (self.menu, &self.map) {
            egui::Area::new("photo menu".into()).fixed_pos(at + egui::vec2(4.0, 4.0)).order(egui::Order::Foreground).show(&ctx, |ui| {
                theme::card().inner_margin(egui::Margin::symmetric(8, 8)).show(ui, |ui| {
                    ui.set_max_width(260.0);
                    ui.label(&m.data.photos[photo].title);
                    ui.add_space(2.0);
                    match self.path_from {
                        Some(from) if from != photo => {
                            if theme::button(ui, egui::Button::new("Path to here")).on_hover_text("a chain of photos from the start you picked to this one, each like the one before it").clicked() {
                                actions.path_to = Some(photo);
                            }
                        }
                        _ => {
                            if theme::button(ui, egui::Button::new("Path from here")).on_hover_text("then right-click the photo it goes to").clicked() {
                                actions.path_from = Some(photo);
                            }
                        }
                    }
                    if theme::button(ui, egui::Button::new("Photos like this")).on_hover_text("search with this photo (F)").clicked() {
                        actions.like = Some(photo);
                    }
                    ui.separator();
                    let file = files::original(&m.data, photo);
                    let source = &m.data.photos[photo].source;
                    let tip = if file.as_os_str() == source.as_str() || source.is_empty() { file.display().to_string() } else { format!("{source} is not there any more: the map's own copy, {}", file.display()) };
                    if theme::button(ui, egui::Button::new("Show in folder")).on_hover_text(tip).clicked() {
                        files::show_in_folder(&file);
                        actions.done = true;
                    }
                    if theme::button(ui, egui::Button::new("Open")).on_hover_text("in the app your system opens it with").clicked() {
                        files::open(&file);
                        actions.done = true;
                    }
                    if theme::button(ui, egui::Button::new("Details")).on_hover_text("file, date taken, camera, place").clicked() {
                        actions.details = Some((photo, at));
                    }
                });
            });
        }
        if actions.path_from.is_some() || actions.path_to.is_some() || actions.like.is_some() || actions.details.is_some() || actions.done {
            self.menu = None;
        }
        if let (Some((photo, at)), Some(m)) = (actions.details, &self.map) {
            self.details = Some((photo, at, files::details(&m.data, photo)));
        }
        if let (Some((photo, at, details)), Some(m)) = (&self.details, &self.map) {
            let mut close = false;
            egui::Area::new("photo details".into()).fixed_pos(*at + egui::vec2(4.0, 4.0)).order(egui::Order::Foreground).constrain(true).show(&ctx, |ui| {
                theme::card().inner_margin(egui::Margin::symmetric(12, 10)).show(ui, |ui| {
                    ui.set_max_width(420.0);
                    close = theme::window_header(ui, &m.data.photos[*photo].title);
                    egui::Grid::new("details grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                        for (name, value) in &details.lines {
                            ui.label(theme::muted(name));
                            match details.place {
                                Some((lat, lon)) if name == "Place" => {
                                    let (place, waiting) = places::describe(lat, lon);
                                    if waiting {
                                        ctx.request_repaint_after(std::time::Duration::from_millis(250));
                                    }
                                    ui.vertical(|ui| {
                                        ui.add(egui::Label::new(place).wrap()).on_hover_text("the nearest town; place names from GeoNames (geonames.org)");
                                        ui.label(theme::muted(format!("{lat:.5}, {lon:.5}")));
                                    });
                                }
                                _ => {
                                    ui.add(egui::Label::new(value).wrap());
                                }
                            }
                            ui.end_row();
                        }
                    });
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        let file = files::original(&m.data, *photo);
                        if theme::button(ui, egui::Button::new("Show in folder")).clicked() {
                            files::show_in_folder(&file);
                        }
                        if theme::button(ui, egui::Button::new("Open")).clicked() {
                            files::open(&file);
                        }
                    });
                });
            });
            if close {
                self.details = None;
            }
        }

        if let (Some(id), Some(c), Some(m), None) = (self.hovered, self.cursor, &self.map, self.menu) {
            let ppp = ctx.pixels_per_point();
            let score = m.sims.get(id).copied();
            egui::Area::new("tip".into()).fixed_pos([c.x / ppp + 14.0, c.y / ppp + 14.0]).interactable(false).show(&ctx, |ui| {
                theme::card().inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                    ui.set_max_width(320.0);
                    ui.label(&m.data.photos[id].title);
                    if let Some(score) = score {
                        ui.label(theme::muted(format!("similarity {score:.2}")));
                    }
                    if self.hovered_copy.is_some() {
                        ui.label(theme::muted("click: go to it on the map"));
                    }
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        theme::key(ui, "F");
                        ui.label(theme::muted("photos like this"));
                    });
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        theme::key(ui, "right-click");
                        ui.label(theme::muted(if self.path_from.is_some() { "end the path here" } else { "a path from here" }));
                    });
                });
            });
        }
    }
}

#[derive(Default)]
struct UiActions {
    ask: bool,
    /// The Layout menu: show the photos the way it now says.
    arrange: bool,
    /// The Layout menu: the same axes, spread the other way.
    respread: bool,
    reblend: bool,
    /// A prev / next button was pressed: -1 / 1.
    step: i32,
    clear_photo: bool,
    /// The cross after the search box: no words, no matches.
    clear_search: bool,
    /// The right-click menu: a path from or to this photo, photos like it.
    path_from: Option<usize>,
    path_to: Option<usize>,
    like: Option<usize>,
    /// "Details": the photo and where the menu was.
    details: Option<(usize, egui::Pos2)>,
    /// A menu item that needs nothing more was chosen: the menu closes.
    done: bool,
    /// The back arrow.
    back: bool,
    /// What the Maps page asked for.
    home: Vec<home::Action>,
}

impl Drop for Viewer {
    /// Closing the app stops the maps being made where they are (everything done so far is on disk).
    fn drop(&mut self) {
        for run in &self.runs {
            run.pause();
        }
        let waited = Instant::now();
        while self.runs.iter().any(|r| r.running()) && waited.elapsed().as_secs() < 20 {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

struct App {
    maps: PathBuf,
    /// The bundle to open first, if one was named on the command line; otherwise the app starts on the Maps page.
    first: Option<PathBuf>,
    viewer: Option<Viewer>,
    /// The Restart button was pressed.
    restart: bool,
}

/// The window's icon (assets/icon.png, drawn by tools/make_icon.py). Wayland ignores it: there the .desktop file gives the icon.
fn icon() -> Option<winit::window::Icon> {
    let img = image::load_from_memory(include_bytes!("../assets/icon.png")).ok()?.into_rgba8();
    let (w, h) = img.dimensions();
    winit::window::Icon::from_rgba(img.into_raw(), w, h).ok()
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.viewer.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("photo map").with_inner_size(winit::dpi::LogicalSize::new(1200.0, 780.0)).with_window_icon(icon());
        // the app id: on Wayland the taskbar finds the icon through the .desktop file of this name (assets/photo-map.desktop)
        #[cfg(target_os = "linux")]
        let attrs = {
            use winit::platform::{wayland::WindowAttributesExtWayland, x11::WindowAttributesExtX11};
            let attrs = WindowAttributesExtWayland::with_name(attrs, "photo-map", "photo-map");
            WindowAttributesExtX11::with_name(attrs, "photo-map", "photo-map")
        };
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        window.request_redraw(); // a window that isn't maximized gets no resize event to start the first frame
        let library = Library::new(self.maps.clone());
        let first = self.first.take();
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
            if v.restart {
                // closed like any other close; main() starts the app again once this process has ended
                self.restart = true;
                self.viewer = None;
                event_loop.exit();
            }
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
    // a map of one photo has no neighbour (the distance is the largest float), and one of a few has them very far apart: either way
    // the square would cover the screen, so it is at most an eighth of the world
    match points.len() {
        0 => PHOTO_WORLD_DEFAULT,
        1 => MAX_PHOTO_WORLD,
        _ => nearest.get(nearest.len() / 2).copied().filter(|d| *d > 0.0).map_or(PHOTO_WORLD_DEFAULT, |d| d.min(MAX_PHOTO_WORLD)),
    }
}

/// The biggest side of a photo's square, in world units (a map of one or two photos would otherwise draw them as big as the world).
const MAX_PHOTO_WORLD: f32 = data::WORLD / 8.0;

const PHOTO_TYPES: [&str; 6] = ["jpg", "jpeg", "png", "webp", "gif", "bmp"];

/// The file a text names, if it is a photo: plain, in quotes, with escaped spaces, or as file://.
fn photo_path(text: &str) -> Option<PathBuf> {
    let t = text.trim().trim_matches(|c| c == '\'' || c == '"').replace("\\ ", " ");
    let path = PathBuf::from(t.strip_prefix("file://").unwrap_or(&t));
    let image = path.extension().and_then(|e| e.to_str()).is_some_and(|e| PHOTO_TYPES.contains(&e.to_lowercase().as_str()));
    (image && path.is_file()).then_some(path)
}

/// Where the app keeps everything it makes and downloads (maps/, runtime/, .cache/): the folder named by PHOTO_MAP_HOME if set, else
/// the system's place for an app's data: ~/.local/share/photo-map on Linux, ~/Library/Application Support/photo-map on macOS,
/// %APPDATA%\photo-map\data on Windows. Tests use it too (the models and maps they need are there).
pub fn data_root() -> PathBuf {
    if let Some(home) = std::env::var_os("PHOTO_MAP_HOME") {
        return PathBuf::from(home);
    }
    match directories::ProjectDirs::from("", "", "photo-map") {
        Some(dirs) => dirs.data_dir().to_path_buf(),
        // no home folder known: beside the program
        None => std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from)).unwrap_or_default(),
    }
}

fn main() {
    let maps = data_root().join("maps");
    let _ = std::fs::create_dir_all(&maps);
    let first = std::env::args().nth(1).map(PathBuf::from);
    // winit doesn't deliver files dropped on a native Wayland window (the photo search takes dropped photos), but does on
    // an X11 one: with an X server around (XWayland), use that, unless PHOTO_MAP_WAYLAND is set
    if std::env::var_os("DISPLAY").is_some() && std::env::var_os("PHOTO_MAP_WAYLAND").is_none() {
        // SAFETY: nothing else is running yet to read the environment
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
    }
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { maps, first, viewer: None, restart: false };
    event_loop.run_app(&mut app).unwrap();
    // The window is closed and the builds are stopped (what they did is on disk). Ending the process the usual way would run the
    // exit code of the CUDA and TensorRT libraries, which after a build takes tens of seconds and then crashes (SIGSEGV); the
    // system frees the video memory by itself, so this ends the process at once.
    if app.restart {
        // a moment's wait so that this process is gone (and the video memory and the locks of the maps it was making are free)
        // before the new one starts
        if let Ok(exe) = std::env::current_exe() {
            let _ = std::process::Command::new("sh").args(["-c", "sleep 1; exec \"$0\"", &exe.to_string_lossy()]).stdin(std::process::Stdio::null()).spawn();
        }
    }
    // SAFETY: `_exit` only ends the process.
    unsafe { libc::_exit(0) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f32, y: f32) -> data::Point {
        data::Point { x, y }
    }

    /// A map of one photo (a single video that made one scene, say) is drawn at a sensible size, not as big as the universe.
    #[test]
    fn a_photo_on_its_own_is_not_drawn_enormous() {
        assert_eq!(median_neighbour_distance(&[at(5000.0, 5000.0)]), MAX_PHOTO_WORLD);
        assert_eq!(median_neighbour_distance(&[]), PHOTO_WORLD_DEFAULT);
    }

    #[test]
    fn photos_far_apart_are_capped_and_close_ones_are_not() {
        assert_eq!(median_neighbour_distance(&[at(1000.0, 5000.0), at(9000.0, 5000.0)]), MAX_PHOTO_WORLD);
        let close: Vec<_> = (0..10).map(|i| at(5000.0 + i as f32 * 30.0, 5000.0)).collect();
        assert_eq!(median_neighbour_distance(&close), 30.0);
    }
}
