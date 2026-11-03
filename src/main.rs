//! Photo map: a fast viewer for map bundles made in Python (see data.rs): every photo at its map position,
//! similar photos close together, region names as you zoom.
//!     photo-map [bundle]      default: the one opened last in maps/

mod beside;
mod bake;
mod backends;
mod bc7;
mod build;
mod camera;
mod cells;
mod data;
mod embed;
mod examples;
mod files;
mod gpu;
mod groups;
mod home;
mod layout;
mod lens;
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
mod query;
mod search;
mod siglip;
mod storage;
mod suggest;
mod stream;
mod theme;
mod tree;

use camera::{Camera, Keys};
use data::{MapData, Progress, WORLD};
use glam::Vec2;
use gpu::{CameraUniform, Gpu, MapRenderer, NO_LIFT};
use library::Library;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;
use stream::Streamer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

/// Side of a photo square in world units is the median distance between nearest neighbours: a map of 12,000 photos
/// gets bigger squares than one of 88,000 in the same world (zoom changes the size on screen, nothing else does).
const PHOTO_WORLD_DEFAULT: f32 = 4.0; // before a map is open
/// Squares never shrink below this many pixels, so far-out photos don't vanish.
const MIN_PX: f32 = 1.0;
/// Region names show while their region's spread is between these sizes on screen (points), fading in and out.
/// Region names on the map (region_names.json, else regions.json).
const SHOW_REGION_NAMES: bool = true;
/// A photo's own caption above it once it is big on screen.
const SHOW_PHOTO_CAPTIONS: bool = false;
/// A photo shows its own caption above it once it is this wide on screen (points), fully from the second size.
const CAPTION_FROM: (f32, f32) = (130.0, 190.0);
/// Stepping to a match zooms in until its photo is this wide on screen (points), unless you are already closer.
const STEP_TO: f32 = 240.0;
const LABEL_IN: (f32, f32) = (25.0, 50.0);
const LABEL_OUT: (f32, f32) = (300.0, 500.0);
/// Seconds the photos take to slide to their places in another view.
const SLIDE_SECONDS: f32 = 0.8;
/// Seconds the photo under the mouse takes to lift (and to settle back).
const LIFT_SECONDS: f32 = 0.12;

/// What is lifted under the mouse: a photo, or a match's copy beside the world (its position among the matches).
#[derive(Clone, Copy, PartialEq)]
enum Lifted {
    Photo(usize),
    Copy(usize),
}

/// Where the photos are: the map's own layout (similar photos together), or sorted into groups by words (groups.rs).
#[derive(Clone, Copy, PartialEq)]
enum View {
    Map,
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

/// A search hidden with "Hide these": its name on the chip, the photos it marked then, and the search itself (to take it back
/// into the box): its words or photos and how many it marked.
struct Hidden {
    what: String,
    photos: Vec<usize>,
    words: String,
    searched: Vec<(PathBuf, egui::TextureHandle)>,
    filters: Vec<query::Filter>,
    wanted: f32,
}

/// A filter chip being written or changed: its place among the filters (`filters.len()`: a new one), what it looks at, whether
/// it leaves out (NOT), and its values as written ("pixel, redmi").
#[derive(Clone)]
struct FilterEdit {
    at: usize,
    key: query::Key,
    not: bool,
    value: String,
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
    /// Where the photos were before the view last changed (x, y, size, unused), and when it changed: they slide from there for
    /// SLIDE_SECONDS.
    from: Vec<[f32; 4]>,
    moved: Option<Instant>,
    /// The Groups view's islands and their words (empty in the other views).
    islands: Vec<groups::Island>,
    island_words: Vec<String>,
    /// Searches hidden with "Hide these": what each was for and the photos it marked then. Hidden photos are drawn faint where
    /// they are (the map doesn't move) and no search marks them. `hidden_mask`: hidden by any of them.
    hidden: Vec<Hidden>,
    hidden_mask: Vec<bool>,
}

impl OpenMap {
    /// Which photos are hidden, after a hidden search was added or taken away.
    fn rehide(&mut self) {
        self.hidden_mask.fill(false);
        for &i in self.hidden.iter().flat_map(|h| &h.photos) {
            self.hidden_mask[i] = true;
        }
        self.flags_dirty = true;
    }

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

    /// Put the photos at new places and sizes (empty: all 1), sliding there from wherever they are drawn now (even halfway through
    /// another slide).
    fn move_to(&mut self, queue: &wgpu::Queue, view: View, points: Vec<data::Point>, sizes: Vec<f32>, photo_world: f32) {
        let t = self.slide();
        let shown: Vec<[f32; 4]> = (0..self.data.n())
            .map(|i| {
                let (a, b, size) = (self.from[i], self.data.points[i], self.data.size(i));
                [a[0] + (b.x - a[0]) * t, a[1] + (b.y - a[1]) * t, a[2] + (size - a[2]) * t, 0.0]
            })
            .collect();
        self.data.set_points(points, sizes);
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
    /// The magnifier while Space is held (lens.rs).
    lens: lens::Lens,
    /// Alt held, as the modifiers say, and as the Alt keys themselves say (in some layouts the modifiers never show Alt).
    alt: bool,
    alt_key: bool,
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
    /// What is lifted, and how far (0 ... 1): it settles back before the next one lifts.
    lifted: (Option<Lifted>, f32),
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
    /// The search being worked on: its terms' query numbers (None once answered), their cosines, and how those combine
    /// (NOT: query.rs).
    asked: Vec<Option<u64>>,
    answers: Vec<Option<Vec<f32>>>,
    logic: query::Logic,
    /// The photos searched for (dropped on the window, or from the photo menu), each with a small picture for the bar. Several are
    /// averaged: what they share counts.
    photos: Vec<(PathBuf, egui::TextureHandle)>,
    /// What the last search did, next to the box.
    status: String,
    /// The words of the Groups view (as typed), and the view the Layout menu has chosen.
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
    /// What the shown search was for (its words, or "this photo").
    shown: String,
    /// Ideas for the search box (suggest.rs).
    ideas: suggest::Suggest,
    /// The Hidden line is open: a chip for each hidden search.
    hidden_open: bool,
    /// How to search shows: the i at the box's end is pointed at.
    tips_open: bool,
    /// Where photos searched for are on the map (None: not on it), found when first pointed at; emptied when a map opens.
    spots: std::collections::HashMap<PathBuf, Option<usize>>,
    /// The search's filters (name: camera: date: place:), chips under the box apart from its words or photos; the one being
    /// written or changed (its place, `filters.len()` for a new one, and its text).
    filters: Vec<query::Filter>,
    filter_edit: Option<FilterEdit>,
    /// Which photos the filters pass, and whether the search is only them.
    names: Option<Vec<bool>>,
    names_only: bool,
    /// How much of the photos' paths every one of them shares (cut off before matching names), found at the first name: filter.
    name_root: Option<usize>,
    /// The runtime libraries: what is installed and the install under way.
    runtime: manager::Manager,
    store: modelstore::ModelStore,
    /// Example photos to make a map from: what is downloaded and the download under way.
    examples: examples::Examples,
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
        let examples = {
            let window = window.clone();
            examples::Examples::new(library.root(), move || window.request_redraw())
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
            examples,
            restart: false,
            opening: None,
            camera,
            keys: Keys::default(),
            lens: lens::Lens::default(),
            alt: false,
            alt_key: false,
            scroll_gesture: false,
            hidden: false,
            help_open: false,
            paused_for_viewing: Vec::new(),
            dropped: Vec::new(),
            cursor: None,
            hovered: None,
            hovered_copy: None,
            lifted: (None, 0.0),
            menu: None,
            details: None,
            path_from: None,
            label_rects: Vec::new(),
            search: String::new(),
            ideas: suggest::Suggest::new(),
            hidden_open: false,
            tips_open: false,
            spots: std::collections::HashMap::new(),
            filters: Vec::new(),
            filter_edit: None,
            names: None,
            names_only: false,
            name_root: None,
            engine,
            asked: Vec::new(),
            answers: Vec::new(),
            logic: query::Logic::one(),
            photos: Vec::new(),
            status: String::new(),
            groups_text: String::new(),
            view_choice: View::Map,
            layout_for: View::Map,
            layout_words: Vec::new(),
            layout_asked: Vec::new(),
            layout_vectors: Vec::new(),
            wanted: 100.0,
            smooth: true,
            shown: String::new(),
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
        self.photos.clear();
        self.filters.clear();
        self.filter_edit = None;
        self.names = None;
        self.shown.clear();
        let vectors = search::Vectors::open(&path, n, data.manifest.dim());
        let tree = tree::read(&path, n);
        let in_process = vectors.is_some();
        self.asked.clear();
        self.view_choice = View::Map; // a map opens as itself; the words are kept for the next one
        self.layout_asked.clear();
        self.engine.want(data.manifest.model()); // load the map's model now, so that the first search is quick
        let map_points = data.points.clone();
        let from = data.points.iter().map(|p| [p.x, p.y, 1.0, 0.0]).collect();
        self.spots.clear();
        self.name_root = None;
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
            islands: Vec::new(),
            island_words: Vec::new(),
            hidden: Vec::new(),
            hidden_mask: vec![false; n],
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
        } else if !self.runs.iter().any(|r| r.running()) {
            // a map is open and nothing is being made (paused for it, or done): the build's sessions leave the card, search stays
            self.engine.hub.release_build();
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
            Action::GetExample(id) => self.examples.get(&id),
            Action::StopExample => self.examples.stop(),
            Action::RemoveExample(id) => self.examples.remove(&id),
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

    /// The lens, if it shows (only on a map).
    fn lens_view(&self) -> Option<lens::LensView> {
        if self.page != Page::Viewer || self.map.is_none() {
            return None;
        }
        self.lens.view(self.camera.center, self.camera.scale, self.camera.viewport, self.window.scale_factor() as f32)
    }

    /// The camera that shows this point of the screen: the lens's inside the lens, else the map's. (World point at the screen's
    /// middle, screen px per world unit.)
    fn camera_at(&self, screen: Vec2) -> (Vec2, f32) {
        match self.lens_view().filter(|l| l.contains(screen)) {
            Some(l) => (l.center, l.scale),
            None => (self.camera.center, self.camera.scale),
        }
    }

    /// The photo under the mouse: of the squares containing it, the one drawn last (on top). Inside the lens, the one it shows there.
    fn pick(&self, screen: Vec2) -> Option<usize> {
        let m = self.map.as_ref()?;
        let data = &m.data;
        let (center, scale) = self.camera_at(screen);
        let viewport = self.camera.viewport;
        let to_screen = |w: Vec2| (w - center) * scale + viewport * 0.5;
        let size = (scale * m.photo_world).max(MIN_PX);
        let world = center + (screen - viewport * 0.5) / scale;
        let reach = Vec2::splat((size * data.max_size() * 0.5).max(4.0) / scale);
        let mut best = None;
        data.grid.within(&data.points, world - reach, world + reach, |i| {
            let p = data.points[i];
            let half = (data.shape(i) * size * 0.5).max(Vec2::splat(4.0));  // real shape; tiny ones stay clickable
            let d = (to_screen(Vec2::new(p.x, p.y)) - screen).abs();
            if d.x <= half.x && d.y <= half.y && best.is_none_or(|b| i > b) {
                best = Some(i);
            }
        });
        best
    }

    /// The match whose copy beside the world is under the mouse: its position among the matches.
    fn pick_copy(&self, screen: Vec2) -> Option<usize> {
        let m = self.map.as_ref()?;
        let (center, scale) = self.camera_at(screen);
        let viewport = self.camera.viewport;
        // the whole cell, not just the picture (a wide photo leaves most of its square empty)
        let half = (m.copy_side / beside::FILL * scale * 0.5).max(4.0);
        m.copies.iter().position(|p| {
            let d = ((Vec2::new(p.x, p.y) - center) * scale + viewport * 0.5 - screen).abs();
            d.x <= half && d.y <= half
        })
    }

    /// Take a photo for the search, instead of the photos before it or (`add`) besides them: remember its path and a small picture
    /// of it to show in the bar. False if it can't be read.
    fn set_photo(&mut self, path: PathBuf, add: bool) -> bool {
        let Ok(img) = image::open(&path) else {
            self.status = "can't read that photo".into();
            return false;
        };
        let small = img.thumbnail(96, 96).to_rgba8();
        let color = egui::ColorImage::from_rgba_unmultiplied([small.width() as usize, small.height() as usize], small.as_raw());
        let tex = self.ui.ctx.load_texture("query photo", color, egui::TextureOptions::LINEAR);
        if !add {
            self.photos.clear();
        }
        if !self.photos.iter().any(|(p, _)| *p == path) {
            self.photos.push((path, tex));
        }
        self.search.clear(); // a search is words or a photo, not both: the photo replaces the words
        true
    }

    /// A picture on the clipboard becomes the search (joining the photos already there, as a dropped one does): it is saved as a
    /// png in .cache/pasted/ (the search reads photos from files). False if the clipboard holds no picture.
    fn paste_photo(&mut self) -> bool {
        let Ok(mut clipboard) = arboard::Clipboard::new() else { return false };
        let Ok(pic) = clipboard.get_image() else { return false };
        let Some(rgba) = image::RgbaImage::from_raw(pic.width as u32, pic.height as u32, pic.bytes.into_owned()) else { return false };
        let dir = self.library.root().join(".cache").join("pasted");
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
        let path = dir.join(format!("{stamp}.png"));
        if std::fs::create_dir_all(&dir).is_err() || rgba.save(&path).is_err() {
            self.status = "can't keep the pasted picture".into();
            return true;
        }
        if self.set_photo(path, !self.photos.is_empty()) {
            self.ask();
        }
        true
    }

    /// Enter in the search box, or a photo dropped on the window: the words or the photo (one of them, never both) go to the model,
    /// whose answer comes back through poll_engine. Nothing to search for clears the last search.
    fn ask(&mut self) {
        let text = self.search.trim().to_string();
        // a path to a photo typed or pasted in the box (from a file manager, or dragged on a terminal) is a photo search
        if let Some(path) = photo_path(&text) {
            if self.set_photo(path, false) {
                self.search.clear();
            }
        }
        // filters typed in the box (camera:pixel ...) leave it for chips; they pick which photos can match, the words or photos rank them
        if self.photos.is_empty() {
            let (rest, found) = query::filters(&self.search);
            if !found.is_empty() {
                self.search = rest;
                self.add_filters(found);
            }
        }
        let text = self.search.trim().to_string();
        let rest = text.clone();
        let filters = self.filters.clone();
        let filters_text = filters.iter().map(query::Filter::text).collect::<Vec<_>>().join(" ");
        self.names = None;
        self.names_only = false;
        if !filters.is_empty() && let Some(m) = &self.map {
            let photos = &m.data.photos;
            let root = *self.name_root.get_or_insert_with(|| common_root(photos.iter().map(|p| p.source.as_str())));
            let pass: Vec<bool> = photos
                .iter()
                .map(|p| {
                    let name = p.source.get(root..).unwrap_or(&p.source);
                    filters.iter().all(|f| f.passes(name, p.facts.as_deref()))
                })
                .collect();
            let hits = pass.iter().zip(&m.hidden_mask).filter(|&(&p, &h)| p && !h).count();
            self.names = Some(pass);
            if rest.is_empty() && self.photos.is_empty() {
                // filters only: every photo they pass marks, no model needed
                self.names_only = true;
                self.asked.clear();
                self.shown = filters_text.clone();
                self.wanted = hits.max(1) as f32;
                if hits == 0 {
                    let knows = |key| filters.iter().any(|f: &query::Filter| f.key == key && !f.not) && photos.iter().all(|p| p.facts.is_none());
                    let unknown = knows(query::Key::Camera) || knows(query::Key::Date) || knows(query::Key::Place);
                    let m = self.map.as_mut().unwrap();
                    m.flags.fill(0);
                    m.flags_dirty = true;
                    m.sims.clear();
                    m.trail = None;
                    m.show_beside(&self.gpu.queue, Vec::new());
                    self.status = if unknown {
                        "this map knows no cameras, dates or places: make it again to read them".into()
                    } else {
                        format!("no photo fits {filters_text}")
                    };
                    return;
                }
                let n = photos.len();
                self.map.as_mut().unwrap().sims = vec![0.0; n];
                if let Some((min, max)) = self.apply_search() {
                    self.camera.fly_to(self.camera.fit(min, max));
                }
                return;
            }
        }
        let Some(m) = &mut self.map else { return };
        if text.is_empty() && self.photos.is_empty() {
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
            let (queries, logic, shown) = match self.photos.len() {
                0 => match query::Logic::parse(&rest) {
                    Ok(logic) => (logic.terms.iter().map(|t| search::Query::Words(t.clone())).collect(), logic, text),
                    Err(e) => {
                        self.status = e;
                        return;
                    }
                },
                1 => (vec![search::Query::Photo(self.photos[0].0.clone())], query::Logic::one(), "this photo".into()),
                n => (self.photos.iter().map(|(p, _)| search::Query::Photo(p.clone())).collect(), query::Logic::photos(n), format!("these {n} photos")),
            };
            self.asked = queries.into_iter().map(|q| Some(self.engine.ask(&model, q))).collect();
            self.answers = vec![None; self.asked.len()];
            self.logic = logic;
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
            m.islands.clear();
            if m.view != View::Map {
                let (points, size) = (m.map_points.clone(), m.map_photo_world);
                m.move_to(&self.gpu.queue, View::Map, points, Vec::new(), size);
                self.camera.fly_to(self.camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
            }
            return;
        }
        let words = groups::words(&self.groups_text);
        if words.len() < 2 {
            self.status = "type at least two words to sort the photos into".into();
            return;
        }
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

    /// All the words' vectors are back: each photo's cosines with them place it (groups.rs), and the photos slide there.
    fn arranged(&mut self) {
        let Some(m) = &mut self.map else { return };
        let Some(vectors) = &m.vectors else { return };
        let words = std::mem::take(&mut self.layout_vectors);
        let Some(cos) = words.iter().map(|v| v.as_ref().map(|v| vectors.cosines(v))).collect::<Option<Vec<_>>>() else { return };
        let forms: Vec<(f32, f32)> = (0..m.data.n()).map(|i| m.data.form(i).into()).collect();
        let groups::Placed { points, sizes, islands, side } = groups::place(&cos, &forms);
        m.move_to(&self.gpu.queue, View::Groups, points, sizes, side);
        self.status = format!("{} groups: each photo is in the one whose word it is most like", islands.len());
        m.islands = islands;
        m.island_words = self.layout_words.clone();
        self.camera.fly_to(self.camera.fit(Vec2::ZERO, Vec2::splat(WORLD)));
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
            let Some(k) = self.asked.iter().position(|&a| a == Some(id)) else {
                continue; // an older query, or the map changed
            };
            self.asked[k] = None;
            let Some(m) = &mut self.map else { continue };
            match (vector, &m.vectors) {
                (Ok(v), Some(vectors)) => {
                    self.answers[k] = Some(vectors.cosines(&v));
                    if self.asked.iter().any(Option::is_some) {
                        continue; // the other terms are still on their way
                    }
                    let cos: Vec<Vec<f32>> = std::mem::take(&mut self.answers).into_iter().flatten().collect();
                    m.sims = self.logic.combine(&cos);
                    if let Some((min, max)) = self.apply_search() {
                        self.camera.fly_to(self.camera.fit(min, max));
                    }
                    if let Some(note) = note {
                        self.status = format!("{} - {note}", self.status);
                    }
                }
                (Err(e), _) => {
                    self.asked.clear(); // the other terms are no use without this one
                    self.status = format!("couldn't search: {e}");
                }
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
        // hidden photos never match, nor ones the filters leave out
        let out = |i: usize| m.hidden_mask[i] || self.names.as_ref().is_some_and(|p| !p[i]);
        let score: Vec<f32> = (0..n).map(|i| if out(i) { f32::NEG_INFINITY } else { m.sims[i] }).collect();
        let shown = (0..n).filter(|&i| !out(i)).count();
        if shown == 0 && self.names.is_some() {
            m.flags.fill(0);
            m.flags_dirty = true;
            m.show_beside(&self.gpu.queue, Vec::new());
            self.status = "no photo fits the filters".into();
            return None;
        }
        if shown == 0 {
            m.flags.fill(0);
            m.flags_dirty = true;
            m.show_beside(&self.gpu.queue, Vec::new());
            self.status = "every photo is hidden".into();
            return None;
        }
        let score = &score;
        // the closest `wanted` photos match: the cutoff is the similarity of the last of them (ties may add a few)
        let k = (self.wanted.round() as usize).clamp(1, shown);
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
        self.status = if self.names_only { format!("{}: {shown} photos", self.shown) } else { format!("{}: similarity {cut:.2} to {best:.2}", self.shown) };
        m.flags_dirty = true;
        Some(m.beside_box())
    }

    /// Filters join the search's chips (one already there isn't doubled).
    fn add_filters(&mut self, found: Vec<query::Filter>) {
        for f in found {
            if !self.filters.contains(&f) {
                self.filters.push(f);
            }
        }
    }

    /// The chip being written or changed is done: its values become the filter (none: the chip goes), and the search runs again
    /// with it.
    fn finish_filter(&mut self) {
        let Some(e) = self.filter_edit.take() else { return };
        let before = self.filters.clone();
        if e.at < self.filters.len() {
            self.filters.remove(e.at);
        }
        if let Some(f) = query::Filter::new(e.key, &e.value, e.not).filter(|f| !self.filters.contains(f)) {
            self.filters.insert(e.at.min(self.filters.len()), f);
        }
        if self.filters != before {
            self.ask();
        }
    }

    /// "Hide these": the search's matches are hidden (drawn faint, never matched again) until their chip is clicked, and the
    /// search is cleared.
    fn hide_matches(&mut self) {
        let Some(m) = &mut self.map else { return };
        if m.order.is_empty() || m.trail.is_some() {
            return;
        }
        // a photo search is named by its photo's file (several: the first, and how many more)
        let file = |p: &PathBuf| p.file_stem().map_or("photo".into(), |s| s.to_string_lossy().into_owned());
        let what = match self.photos.as_slice() {
            [] if self.shown.is_empty() => "search".to_string(),
            [] => self.shown.clone(),
            [(p, _)] => format!("like {}", file(p)),
            [(p, _), rest @ ..] => format!("like {} +{}", file(p), rest.len()),
        };
        // with its filters, unless they are the whole search (and so its name already)
        let what = if self.names_only || self.filters.is_empty() { what } else { format!("{what} {}", self.filters.iter().map(query::Filter::text).collect::<Vec<_>>().join(" ")) };
        let words = if self.photos.is_empty() && !self.names_only { self.shown.clone() } else { String::new() };
        m.hidden.push(Hidden { what, photos: m.order.clone(), words, searched: self.photos.clone(), filters: std::mem::take(&mut self.filters), wanted: self.wanted });
        m.rehide();
        self.search.clear();
        self.photos.clear();
        self.ask(); // nothing to search for: the matches go (the Hidden line says what was hidden)
    }

    /// A hidden search's × was clicked: its photos come back (and can match the search showing now). `edit`: its chip was clicked
    /// instead: the search itself comes back too (its words or photos, and how many it marked), to change and hide again.
    fn unhide(&mut self, k: usize, edit: bool) {
        let Some(m) = &mut self.map else { return };
        if k >= m.hidden.len() {
            return;
        }
        let h = m.hidden.remove(k);
        m.rehide();
        if edit {
            self.search = h.words;
            self.photos = h.searched;
            self.filters = h.filters;
            self.wanted = h.wanted;
            self.ask();
        } else if !m.sims.is_empty() {
            self.apply_search();
        }
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
        self.photos.clear();
        self.shown.clear();
        self.asked.clear();
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

    /// F over a photo: the same as dropping that photo's file on the window (it goes in the search box and is searched), instead of
    /// the photos searched for or (`add`) besides them.
    fn ask_like(&mut self, photo: usize, add: bool) {
        let Some(file) = self.map.as_ref().map(|m| m.data.path(photo)) else { return };
        if self.set_photo(file, add) {
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

    fn key(&mut self, key: KeyCode, down: bool) {
        if let (true, KeyCode::F11) = (down, key) {
            return self.toggle_fullscreen();
        }
        if self.page != Page::Viewer {
            return;
        }
        if let (true, KeyCode::Escape) = (down, key) {
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
        if down {
            match key {
                KeyCode::KeyN | KeyCode::BracketRight => return self.step_match(1),
                KeyCode::KeyP | KeyCode::BracketLeft => return self.step_match(-1),
                KeyCode::KeyH => {
                    self.help_open = !self.help_open;
                    return;
                }
                _ => {}
            }
        }
        if key == KeyCode::KeyF {
            if let (true, Some(i)) = (down, self.hovered) {
                self.ask_like(i, false);
            }
            return;
        }
        if key == KeyCode::Space {
            self.lens.held = down && self.page == Page::Viewer;
            return;
        }
        let k = &mut self.keys;
        match key {
            KeyCode::ArrowLeft | KeyCode::KeyA => k.left = down,
            KeyCode::ArrowRight | KeyCode::KeyD => k.right = down,
            KeyCode::ArrowUp | KeyCode::KeyW => k.up = down,
            KeyCode::ArrowDown | KeyCode::KeyS => k.down = down,
            KeyCode::KeyE | KeyCode::Equal | KeyCode::NumpadAdd => k.zoom_in = down,
            KeyCode::KeyQ | KeyCode::Minus | KeyCode::NumpadSubtract => k.zoom_out = down,
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
                if self.lens.held && self.page == Page::Viewer {
                    // while the lens is held the wheel sets how much it magnifies (a trackpad's two fingers still pan)
                    match delta {
                        MouseScrollDelta::LineDelta(_, y) => self.lens.scroll(y),
                        MouseScrollDelta::PixelDelta(p) if self.scroll_gesture => self.camera.pan_pixels(Vec2::new(p.x as f32, p.y as f32)),
                        MouseScrollDelta::PixelDelta(p) => self.lens.scroll(p.y as f32 / 50.0),
                    }
                } else if !over_ui && self.page == Page::Viewer {
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
                if let PhysicalKey::Code(KeyCode::AltLeft | KeyCode::AltRight) = event.physical_key {
                    self.alt_key = event.state == ElementState::Pressed;
                }
                if (self.alt || self.alt_key) && event.state == ElementState::Pressed && !event.repeat && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Enter | KeyCode::NumpadEnter)) {
                    return self.toggle_fullscreen();
                }
                // Ctrl+V with a picture on the clipboard (text pastes go to the box as usual): it is searched like a dropped photo
                if self.page == Page::Viewer
                    && event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::KeyV)
                    && self.ui.ctx.input(|i| i.modifiers.command)
                    && self.paste_photo()
                {
                    return;
                }
                let typing = self.ui.ctx.egui_wants_keyboard_input() && event.state == ElementState::Pressed;
                // by where the key is, not what it types: the shortcuts work in any keyboard layout
                let PhysicalKey::Code(code) = event.physical_key else { return };
                if !typing && !egui_took {
                    self.key(code, event.state == ElementState::Pressed);
                } else if event.state == ElementState::Released {
                    self.key(code, false); // never leave a key stuck down
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
            // a photo dropped while photos are the search joins them (one event per file: several dropped at once all join)
            WindowEvent::DroppedFile(path) if self.page == Page::Viewer => {
                if path.extension().and_then(|e| e.to_str()).is_some_and(|e| PHOTO_TYPES.contains(&e.to_lowercase().as_str())) {
                    if self.set_photo(path, !self.photos.is_empty()) {
                        self.ask();
                    }
                } else {
                    self.status = "that isn't a photo I can search with (jpg, png, webp, gif, bmp)".into();
                }
            }
            WindowEvent::Focused(false) => {
                self.keys = Keys::default();
                self.lens.held = false;
                self.alt_key = false; // Alt+Tab away: its release goes to another window
            }
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
        self.lens.update(dt, self.cursor);
        self.frames += 1;
        self.poll_engine();
        self.runtime.tick();
        self.store.tick();
        self.examples.tick();
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
        let want = match (self.hovered_copy, self.hovered) {
            (Some(k), _) => Some(Lifted::Copy(k)),
            (None, Some(i)) => Some(Lifted::Photo(i)),
            _ => None,
        };
        if self.lifted.0 == want {
            self.lifted.1 = (self.lifted.1 + dt / LIFT_SECONDS).min(1.0);
        } else {
            self.lifted.1 -= dt / LIFT_SECONDS;
            if self.lifted.1 <= 0.0 {
                self.lifted = (want, 0.0);
            }
        }

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
            lens_at: [0.0; 2],
            lens_radius: 0.0,
            lift: { let t = self.lifted.1; t * t * (3.0 - 2.0 * t) },
            lift_photo: match (self.lifted.0, &self.map) {
                (Some(Lifted::Photo(i)), Some(m)) if i < m.data.n() => i as u32,
                _ => NO_LIFT,
            },
            lift_copy: match (self.lifted.0, &self.map) {
                (Some(Lifted::Copy(k)), Some(m)) if k < m.order.len() => k as u32,
                _ => NO_LIFT,
            },
            pad: [0.0; 2],
        };
        // the lens's camera, and what it wants loaded: the world box under its circle at its sharpness
        let lens = self.lens_view().zip(self.map.as_ref()).map(|(l, m)| {
            let viewport = self.camera.viewport;
            let (size_px, copy_px) = ((l.scale * m.photo_world).max(MIN_PX), m.copy_side * l.scale);
            let reach = Vec2::splat(l.radius);
            let request = stream::LensRequest { min: l.to_world(l.at - reach, viewport), max: l.to_world(l.at + reach, viewport), center: l.to_world(l.at, viewport), size_px, copy_px };
            let uniform = CameraUniform { center: l.center.into(), scale: l.scale, size_px, copy_px, lens_at: l.at.into(), lens_radius: l.radius, ..cam };
            (uniform, request)
        });
        match &mut self.map {
            Some(m) if self.page == Page::Viewer => {
                if m.flags_dirty {
                    // the shader's flags: the match (0 ... 255) and whether the photo is hidden (map.wgsl HIDDEN)
                    let flags: Vec<u32> = m.flags.iter().zip(&m.hidden_mask).map(|(&f, &h)| f | if h { 256 } else { 0 }).collect();
                    m.renderer.upload_flags(&self.gpu.queue, &flags);
                    m.flags_dirty = false;
                }
                let copies = (&m.order[..], &m.copies[..], cam.copy_px);
                m.streamer.update(&self.gpu.queue, &m.data, view_min, view_max, self.camera.center, size_px, copies, lens.map(|l| l.1));
                m.renderer.draw(&self.gpu.queue, &mut encoder, &view, cam, self.smooth, lens.map(|l| l.0));
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
            self.photos.clear();
            self.ask();
            self.ui.ctx.memory_mut(|m| m.request_focus(egui::Id::new("search-box"))); // ready to type
        }
        if actions.clear_search {
            if self.search.trim().is_empty() {
                self.filters.clear(); // the box was empty already: the cross takes the filters too
            }
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
            self.ask_like(i, false);
        }
        if let Some(i) = actions.add_like {
            self.ask_like(i, true);
        }
        if let Some(i) = actions.show_photo {
            self.go_to_photo(i);
        }
        if let Some(k) = actions.drop_photo {
            let _ = self.photos.remove(k);
            self.ask();
            if self.photos.is_empty() {
                self.ui.ctx.memory_mut(|m| m.request_focus(egui::Id::new("search-box"))); // the last one: back to words
            }
        }
        if actions.filter_done {
            self.finish_filter();
        }
        if let Some(mut e) = actions.edit_filter {
            self.finish_filter(); // the one being written before (which can move the others: the chip is found again by what it is)
            let same = |f: &query::Filter| f.key == e.key && f.not == e.not && f.value_text() == e.value;
            e.at = if e.at == usize::MAX { self.filters.len() } else { self.filters.iter().position(same).unwrap_or(self.filters.len()) };
            self.filter_edit = Some(e);
            self.ui.ctx.memory_mut(|m| m.request_focus(egui::Id::new("filter-edit")));
        }
        if let Some(k) = actions.drop_filter {
            if k < self.filters.len() {
                self.filters.remove(k);
            }
            self.ask();
        }
        if actions.ask {
            self.ask();
        }
        if actions.arrange {
            self.arrange();
        }
        if actions.reblend {
            self.apply_search();
        }
        if actions.hide {
            self.hide_matches();
        }
        if let Some((k, edit)) = actions.unhide {
            self.unhide(k, edit);
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
                if !screen.intersects(rect) || taken.iter().any(|t| t.intersects(rect.expand(6.0))) || self.under_lens(rect, ppp) {
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

    /// The Groups view's words, each above its island with how many photos it has.
    fn draw_island_words(&self, ctx: &egui::Context) {
        let Some(m) = &self.map else { return };
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "group words".into()));
        let ppp = ctx.pixels_per_point();
        for island in &m.islands {
            let Some(word) = m.island_words.get(island.word) else { continue };
            // no wider than the island on screen (cut with "…"), so neighbours' names never run into each other
            let width = island.size.x * self.camera.scale / ppp;
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
            let at = self.camera.to_screen(Vec2::new(island.min.x + island.size.x / 2.0, island.min.y)) / ppp;
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

    /// Zoom to a photo until it nearly fills the window.
    fn go_to_photo(&mut self, i: usize) {
        let Some(m) = &self.map else { return };
        let p = m.data.points[i];
        let fit = self.camera.viewport / (m.data.shape(i) * m.photo_world);
        self.camera.fly_to((Vec2::new(p.x, p.y), 0.85 * fit.min_element()));
    }

    /// A click on the map (a press that hardly moved): on a match's copy beside the world, it flies to the photo; on a region
    /// name, it zooms to the region; on a photo, to the photo.
    fn click_at(&mut self) {
        let Some(c) = self.cursor else { return };
        if let Some(k) = self.pick_copy(c) {
            return self.go_to_match(k);
        }
        let ppp = self.window.scale_factor() as f32;
        let at = egui::pos2(c.x / ppp, c.y / ppp);
        if let Some(&(_, level, index)) = self.label_rects.iter().find(|(rect, _, _)| rect.contains(at)) {
            self.go_to_region(level, index);
        } else if let Some(i) = self.pick(c) {
            self.go_to_photo(i);
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
        // a photo's outline: `side` its longer side on screen at size 1 (points), never smaller than a few points so it can be
        // found; `copy`: its copy beside the world, never bigger
        let outline = |i: usize, at: egui::Pos2, side: f32, width: f32, copy: bool| {
            let shape = if copy { m.data.form(i) } else { m.data.shape(i) };
            let size = (egui::vec2(shape.x, shape.y) * side).max(egui::vec2(8.0, 8.0)) + egui::vec2(6.0, 6.0);
            painter.rect_stroke(egui::Rect::from_center_size(at, size), 3.0, egui::Stroke::new(width, egui::Color32::WHITE), egui::StrokeKind::Outside);
        };
        let (photo_side, copy_side) = (self.size_px() / ppp, m.copy_side * self.camera.scale / ppp);
        if let Some(i) = self.path_from {
            outline(i, to_screen(m.data.points[i]), photo_side, 3.0, false);
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
            outline(m.order[k], to_screen(m.copies[k]), copy_side, 2.0, true);
        }
        match (self.hovered_copy, self.hovered) {
            (Some(k), _) => outline(m.order[k], to_screen(m.data.points[m.order[k]]), photo_side, 2.0, false),
            (None, Some(i)) => {
                if let Some(k) = m.order.iter().position(|&o| o == i) {
                    outline(i, to_screen(m.copies[k]), copy_side, 2.0, true);
                }
            }
            _ => {}
        }
    }

    /// Does this rectangle (points) touch the lens? Words drawn over the map leave the lens clear.
    fn under_lens(&self, rect: egui::Rect, ppp: f32) -> bool {
        self.lens_view().is_some_and(|l| l.touches(Vec2::new(rect.min.x, rect.min.y) * ppp, Vec2::new(rect.max.x, rect.max.y) * ppp))
    }

    /// The lens's rim (its ground and photos are drawn with the map), and how much it magnifies while that is being changed.
    fn draw_lens(&self, ctx: &egui::Context) {
        let Some(l) = self.lens_view() else { return };
        let ppp = ctx.pixels_per_point();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "lens".into()));
        let (at, radius) = (egui::pos2(l.at.x / ppp, l.at.y / ppp), l.radius / ppp);
        painter.circle_stroke(at, radius, egui::Stroke::new(1.5, egui::Color32::from_white_alpha((150.0 * l.shown) as u8)));
        if let Some((text, alpha)) = self.lens.label() {
            let galley = painter.layout_no_wrap(text, egui::FontId::proportional(14.0), egui::Color32::WHITE);
            let rect = egui::Rect::from_center_size(at + egui::vec2(0.0, radius - 22.0), galley.size()).expand2(egui::vec2(8.0, 3.0));
            painter.rect_filled(rect, 8.0, egui::Color32::from_black_alpha((170.0 * alpha * l.shown) as u8));
            painter.galley(rect.min + egui::vec2(8.0, 3.0), galley, egui::Color32::WHITE.gamma_multiply(alpha * l.shown));
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
            if self.under_lens(rect, ppp) {
                return;
            }
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
            if !screen.intersects(rect) || taken.iter().any(|t| t.intersects(rect.expand(3.0))) || self.under_lens(rect, ppp) {
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
            actions.home = self.home.show(&ctx, &self.library, &views, self.library.resume_on_start(), ready_msg, &self.runtime, &self.store, &self.examples);
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
        if SHOW_PHOTO_CAPTIONS {
            self.draw_photo_captions(&ctx);
        }
        self.draw_beside(&ctx);
        self.draw_lens(&ctx);

        // the top of the screen, by job: where you are (left), the search and its results (middle), how the map looks (right)
        let screen_w = ctx.content_rect().width();
        let width = (screen_w - 2.0 * 250.0).clamp(300.0, 620.0);
        let narrow = screen_w < width + 2.0 * 250.0; // no room beside the search: the corners go under it
        let corner_y = if narrow { 64.0 } else { 9.0 }; // beside the box: centred on it

        // left: the way out and which map this is
        egui::Area::new("where".into()).anchor(egui::Align2::LEFT_TOP, [12.0, corner_y]).show(&ctx, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin::symmetric(6, 4)).show(ui, |ui| {
                ui.horizontal(|ui| {
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
                    if let Some(m) = &self.map {
                        ui.label(egui::RichText::new(&m.data.manifest.name).color(theme::MUTED));
                        ui.add_space(6.0);
                    }
                });
            });
        });

        // right: how the photos are placed and drawn
        egui::Area::new("look".into()).anchor(egui::Align2::RIGHT_TOP, [-12.0, corner_y]).show(&ctx, |ui| {
            egui::Frame::NONE.inner_margin(egui::Margin::symmetric(6, 4)).show(ui, |ui| {
                ui.horizontal(|ui| {
                    // they stay open while you change things: they close on a click outside them
                    let stay_open = || egui::containers::menu::MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
                    egui::containers::menu::MenuButton::new("Layout").config(stay_open()).ui(ui, |ui| {
                        ui.set_width(300.0);
                        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                            ui.label(theme::muted("PLACE THE PHOTOS"));
                            let options = [(View::Map, "Map", "similar photos together, as the map was made"), (View::Groups, "Groups", "sorted into groups by words you choose")];
                            if theme::segmented(ui, &mut self.view_choice, &options) && self.view_choice == View::Map {
                                actions.arrange = true;
                            }
                            if self.view_choice == View::Map {
                                ui.label(theme::muted("Photos that look alike sit close together, in regions with names."));
                            } else {
                                ui.label(theme::muted("Each photo joins the word it is most like. In each group the strongest matches are in the middle, the weakest at the edge."));
                                ui.label(theme::muted("The model can tell which word fits a photo better, but not where a word stops fitting at all. So a photo that fits none of them still lands in some group, at its edge."));
                                ui.add_space(6.0);
                                presets(ui, |ui| {
                                    for (name, words) in groups::PRESETS {
                                        if ui.selectable_label(false, name).on_hover_text(words).clicked() {
                                            self.groups_text = words.to_string();
                                            actions.arrange = true;
                                        }
                                    }
                                });
                                ui.add_space(4.0);
                                ui.add(egui::TextEdit::multiline(&mut self.groups_text).hint_text("mountain, city, forest, river").desired_rows(3).desired_width(f32::INFINITY));
                                ui.label(theme::muted("Separate the words with commas or new lines."));
                                ui.add_space(6.0);
                                if theme::button(ui, egui::Button::new("Arrange")).on_hover_text("sort the photos into these groups").clicked() {
                                    actions.arrange = true;
                                }
                            }
                        });
                    });
                    egui::containers::menu::MenuButton::new("Options").config(stay_open()).ui(ui, |ui| {
                        ui.set_width(280.0);
                        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                            ui.label(theme::muted("PHOTO PIXELS"));
                            theme::segmented(ui, &mut self.smooth, &[(true, "Smooth", "pixels blended with their neighbours"), (false, "Nearest", "each pixel as it is: sharp, blocky up close")]);
                        });
                    });
                });
            });
        });

        // middle: the search box, then (only when there is something) the results, the hidden searches, and tips
        let box_id = egui::Id::new("search-box");
        let focused = ctx.memory(|m| m.has_focus(box_id));
        egui::Area::new("top".into()).anchor(egui::Align2::CENTER_TOP, [0.0, 0.0]).show(&ctx, |ui| {
            // no card around it all: the box stands on its own, and what is under it gets a card only while there is something
            egui::Frame::NONE.inner_margin(egui::Margin::same(8)).show(ui, |ui| {
                ui.set_width(width - 16.0);
                ui.set_max_width(width - 16.0); // the rows below wrap at this width instead of widening the card
                // the box: whether search can run (a dot), the words or the photos searched with, a cross to clear
                // the edge stays quiet; typing in the box lights a ring just inside its search line, not around everything under it
                let ring = theme::CARD_STROKE;
                // what shows under the box (results, status, hidden searches, help) joins it: one box, a line between
                let (found, at, on_path) = self.map.as_ref().map_or((0, None, false), |m| (m.order.len(), m.at, m.trail.is_some()));
                let chips = self.map.as_ref().is_some_and(|m| !m.hidden.is_empty()) && self.hidden_open;
                let has_filters = !self.filters.is_empty() || self.filter_edit.is_some();
                let below = |tips: bool| found > 0 || !self.status.is_empty() || chips || tips || has_filters || self.photos.len() == 1;
                let r = theme::RADIUS_SMALL;
                let top_corners = if below(self.tips_open) { egui::CornerRadius { nw: r, ne: r, sw: 0, se: 0 } } else { egui::CornerRadius::same(r) };
                let search_line = egui::Frame::new().fill(ui.visuals().extreme_bg_color).stroke(egui::Stroke::new(1.0, ring)).corner_radius(top_corners).inner_margin(egui::Margin::symmetric(10, 2)).shadow(egui::Shadow { offset: [0, 4], blur: 18, spread: 0, color: egui::Color32::from_black_alpha(110) }).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.set_min_height(32.0);
                        let searchable = self.map.as_ref().is_some_and(|m| m.vectors.is_some());
                        let map_model = self.map.as_ref().map(|m| m.data.manifest.model().to_string());
                        if let Some(why) = self.engine.failed() {
                            theme::dot(ui, theme::DANGER);
                            ui.label(theme::muted("search off")).on_hover_text(why);
                        } else if self.map.is_some() && !searchable {
                            theme::dot(ui, theme::MUTED);
                            ui.label(theme::muted("not searchable")).on_hover_text("this map has no embeddings.bin of the right size");
                        } else if map_model.as_deref().and_then(|m| self.engine.ready(m)).is_some() {
                            let (rect, dot) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 4.0, theme::OK);
                            dot.on_hover_text("Search ready. The AI model that turns your words or photos into numbers is loaded. It runs on this computer, nothing is sent anywhere.");
                        } else if self.map.is_some() {
                            ui.add(egui::Spinner::new().size(12.0)).on_hover_text("the search model is loading");
                        }
                        let clearable = !self.search.is_empty() || !self.photos.is_empty() || self.map.as_ref().is_some_and(|m| !m.order.is_empty());
                        // how many photos are hidden, a small badge at the box's end that opens the hidden searches
                        let hidden_total: usize = self.map.as_ref().map_or(0, |m| m.hidden.iter().map(|h| h.photos.len()).sum());
                        let badge = (hidden_total > 0).then(|| format!("{hidden_total} hidden {}", if self.hidden_open { "⏷" } else { "⏵" }));
                        let badge_w = badge.as_ref().map_or(0.0, |b| ui.fonts_mut(|f| f.layout_no_wrap(b.clone(), egui::FontId::proportional(12.0), theme::MUTED).size().x) + 12.0 + ui.spacing().item_spacing.x);
                        // filters are chips of their own under the box: this adds one
                        let add_filter = theme::muted("+ filter");
                        let add_filter_w = egui::WidgetText::from(add_filter.clone()).into_galley(ui, Some(egui::TextWrapMode::Extend), f32::INFINITY, egui::TextStyle::Button).size().x + 2.0 * ui.spacing().button_padding.x + ui.spacing().item_spacing.x;
                        let cross = if clearable { 26.0 + ui.spacing().item_spacing.x } else { 0.0 } + badge_w + add_filter_w + 26.0 + ui.spacing().item_spacing.x; // and the i
                        if !self.photos.is_empty() {
                            // a search is photos or words, never both: while photos are the search they take the place of the words
                            for (k, (path, tex)) in self.photos.iter().enumerate() {
                                let image = egui::Image::new((tex.id(), egui::vec2(26.0 * tex.aspect_ratio(), 26.0))).corner_radius(4);
                                let thumb = ui.add(image.sense(egui::Sense::click()));
                                // a click goes to it on the map; under the mouse, a small cross in its corner takes it out
                                let corner = egui::Rect::from_center_size(thumb.rect.right_top() + egui::vec2(-5.0, 5.0), egui::vec2(13.0, 13.0));
                                let out = ui.interact(corner, thumb.id.with("out"), egui::Sense::click());
                                if thumb.hovered() || out.hovered() {
                                    let fill = if out.hovered() { theme::DANGER } else { egui::Color32::from_black_alpha(200) };
                                    ui.painter().circle_filled(corner.center(), 6.5, fill);
                                    let (c, r) = (corner.center(), 2.5);
                                    let stroke = egui::Stroke::new(1.4, egui::Color32::WHITE);
                                    ui.painter().line_segment([c + egui::vec2(-r, -r), c + egui::vec2(r, r)], stroke);
                                    ui.painter().line_segment([c + egui::vec2(-r, r), c + egui::vec2(r, -r)], stroke);
                                }
                                let spot = if thumb.hovered() { *self.spots.entry(path.clone()).or_insert_with(|| self.map.as_ref().and_then(|m| find_photo(&m.data, path))) } else { None };
                                if out.on_hover_text("take it out of the search").clicked() {
                                    actions.drop_photo = Some(k);
                                } else if thumb.hovered() && spot.is_none() {
                                    thumb.on_hover_text("not on this map: dropped in or pasted");
                                } else if thumb.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text("go to it on the map").clicked() {
                                    actions.show_photo = spot;
                                }
                            }
                            ui.add_space((ui.available_width() - cross).max(0.0));
                        } else {
                            // ideas (suggest.rs): up / down switch, Tab takes one; taken before the box sees the keys
                            let mut to_end = false;
                            if focused {
                                let (tab, down, up) = ui.input_mut(|i| (i.consume_key(egui::Modifiers::NONE, egui::Key::Tab), i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown), i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)));
                                if (down || up) && let Some(whole) = self.ideas.switch(&self.search, if down { 1 } else { -1 }) {
                                    self.search = whole;
                                    to_end = true;
                                }
                                if tab && let Some(idea) = self.ideas.idea(&self.search) {
                                    self.ideas.took(&idea);
                                    self.search = idea;
                                    to_end = true;
                                }
                            }
                            let idea = self.ideas.idea(&self.search);
                            // empty: the idea is the placeholder; typing: the rest of it shows faint after the words
                            let hint = if self.search.is_empty() { idea.clone().unwrap_or_default() } else { String::new() };
                            // while in the box with an idea showing: the keys for ideas, as key caps at its end
                            let whole = self.ideas.replaces(&self.search);
                            let keys = focused && (idea.is_some() || whole);
                            // the hint's slot, with the gap before it: the box takes the rest and so keeps its width
                            let keys_w = if keys { 170.0 + ui.spacing().item_spacing.x } else { 0.0 };
                            let edit = egui::TextEdit::singleline(&mut self.search).id(box_id).hint_text(hint).frame(egui::Frame::NONE).desired_width(ui.available_width() - cross - keys_w).show(ui);
                            // the keys stay with the box: Tab and up / down would otherwise move to another button
                            ui.memory_mut(|m| m.set_focus_lock_filter(box_id, egui::EventFilter { tab: true, horizontal_arrows: true, vertical_arrows: true, ..Default::default() }));
                            if focused && !self.search.is_empty() && let Some(idea) = &idea {
                                let rest: String = idea.chars().skip(self.search.chars().count()).collect();
                                let at = edit.galley_pos + egui::vec2(edit.galley.rect.width(), 0.0);
                                ui.painter_at(edit.response.rect).text(at, egui::Align2::LEFT_TOP, rest, egui::TextStyle::Body.resolve(ui.style()), theme::MUTED.gamma_multiply(0.8));
                            }
                            if keys {
                                // the arrows in the monospace font: the proportional one has no ↑↓
                                let faint = theme::MUTED.gamma_multiply(0.7);
                                let mut hint = egui::text::LayoutJob::default();
                                hint.append("↑↓", 0.0, egui::TextFormat::simple(egui::FontId::monospace(11.5), faint));
                                let (what, tip) = if whole { (" another search", "up and down: put another search in the box") } else { (" other idea", "up and down: another idea; Enter in the empty box: search it") };
                                hint.append(what, 0.0, egui::TextFormat::simple(egui::FontId::proportional(11.5), faint));
                                // always there (fainter with nothing to take), in a slot of its own width: the box keeps its size
                                let take = if idea.is_some() { faint } else { faint.gamma_multiply(0.45) };
                                hint.append(" · Tab take", 0.0, egui::TextFormat::simple(egui::FontId::proportional(11.5), take));
                                let tip = if idea.is_some() { format!("{tip}; Tab: take the faint rest") } else { tip.to_string() };
                                ui.allocate_ui_with_layout(egui::vec2(170.0, ui.available_height()), egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.set_width(170.0);
                                    ui.add(egui::Label::new(hint).truncate()).on_hover_text(tip);
                                });
                            }
                            // a filter finished in the box (a space after it) leaves it for a chip
                            if edit.response.changed() && let Some((rest, found)) = query::finished(&self.search) {
                                self.search = rest;
                                for f in found {
                                    if !self.filters.contains(&f) {
                                        self.filters.push(f);
                                    }
                                }
                                to_end = true;
                            }
                            if to_end {
                                let mut state = edit.state;
                                state.cursor.set_char_range(Some(egui::text::CCursorRange::one(egui::text::CCursor::new(self.search.chars().count()))));
                                state.store(ui.ctx(), box_id);
                            }
                            if edit.response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                // Enter in the empty box searches the idea it shows
                                if self.search.trim().is_empty() && let Some(idea) = idea {
                                    self.search = idea;
                                }
                                actions.ask = true;
                            }
                        }
                        egui::containers::menu::MenuButton::from_button(egui::Button::new(add_filter).frame(false)).ui(ui, |ui| {
                            ui.set_width(220.0);
                            ui.label(theme::muted("Only photos that pass it can match. Or type it in the box: camera:pixel, then a space."));
                            ui.add_space(4.0);
                            for (key, tip) in [
                                (query::Key::Name, "in the file and folder names: greece"),
                                (query::Key::Camera, "the camera it was taken with: pixel, or pixel, redmi for any of them"),
                                (query::Key::Date, "when it was taken: 2019, 2019-07, or 2016..2018 for a range"),
                                (query::Key::Place, "the town nearest to where it was taken: lviv"),
                            ] {
                                if theme::button(ui, egui::Button::new(key.label())).on_hover_text(tip).clicked() {
                                    actions.edit_filter = Some(FilterEdit { at: usize::MAX, key, not: false, value: String::new() });
                                    ui.close();
                                }
                            }
                        }).0.on_hover_text("only photos that pass a filter can match: by name, camera, date or place");
                        if let Some(badge) = badge {
                            let tip = if self.hidden_open { "fold away the hidden searches" } else { "show the hidden searches, to bring some back or change them" };
                            if ui.add(egui::Button::new(theme::muted(badge)).frame(false).min_size(egui::vec2(0.0, 20.0))).on_hover_text(tip).clicked() {
                                self.hidden_open = !self.hidden_open;
                            }
                        }
                        if clearable && theme::close_button(ui) {
                            if self.photos.is_empty() {
                                actions.clear_search = true;
                            } else {
                                actions.clear_photo = true;
                            }
                        }
                        self.tips_open = theme::info_button(ui);
                    });
                });
                if focused {
                    let inset = search_line.response.rect.shrink(2.0);
                    let corners = if below(self.tips_open) { egui::CornerRadius { nw: r - 2, ne: r - 2, sw: 2, se: 2 } } else { egui::CornerRadius::same(r - 2) };
                    ui.painter().rect_stroke(inset, corners, egui::Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.7)), egui::StrokeKind::Inside);
                }

                // the results: step through them, how many are marked, hide them
                let tips = self.tips_open;
                if !below(tips) {
                    return;
                }
                ui.add_space(-ui.spacing().item_spacing.y - 1.0); // right under the box, sharing its edge
                // as wide as the box: the margins (10 each side) and the stroke come out of the width, not added to it
                let inner = ui.available_width() - 2.0 * 10.0 - 2.0;
                let bottom_corners = egui::CornerRadius { nw: 0, ne: 0, sw: r, se: r };
                egui::Frame::new().fill(ui.visuals().extreme_bg_color).stroke(egui::Stroke::new(1.0, ring)).corner_radius(bottom_corners).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                    ui.set_width(inner);
                    ui.set_max_width(inner);
                    // the filters: a chip each (click: change it, NOT ones in red), and the one being written
                    if has_filters {
                        ui.horizontal_wrapped(|ui| {
                            let editing = self.filter_edit.as_ref().map(|e| e.at);
                            // a chip's name ("Camera"), NOT ones in red, then its values
                            let font = egui::TextStyle::Button.resolve(ui.style());
                            let ink = ui.visuals().text_color();
                            let chip_text = |key: query::Key, not: bool, value: &str| {
                                let mut job = egui::text::LayoutJob::default();
                                let font = font.clone();
                                let (head, color) = if not { (format!("NOT {}", key.label()), theme::DANGER) } else { (key.label().to_string(), theme::MUTED) };
                                job.append(&head, 0.0, egui::TextFormat::simple(font.clone(), color));
                                if !value.is_empty() {
                                    job.append(value, 6.0, egui::TextFormat::simple(font, ink));
                                }
                                job
                            };
                            for k in 0..=self.filters.len() {
                                if editing == Some(k) {
                                    let e = self.filter_edit.as_mut().unwrap();
                                    // its name, which a click turns to NOT and back; the values after it
                                    let head = ui.add(egui::Button::new(chip_text(e.key, e.not, "")).wrap_mode(egui::TextWrapMode::Extend)).on_hover_text("click: leave these out instead (NOT), or back");
                                    if head.clicked() {
                                        e.not = !e.not;
                                        ui.memory_mut(|m| m.request_focus(egui::Id::new("filter-edit")));
                                    }
                                    let edit = ui.add(egui::TextEdit::singleline(&mut e.value).id(egui::Id::new("filter-edit")).hint_text(e.key.hint()).desired_width(140.0));
                                    if edit.lost_focus() && !head.hovered() {
                                        actions.filter_done = true;
                                    }
                                    ui.add_space(6.0);
                                    continue;
                                }
                                let Some(f) = self.filters.get(k) else { continue };
                                let tip = if f.not { "photos this picks are left out; click to change it" } else { "only photos this picks can match; click to change it" };
                                if ui.add(egui::Button::new(chip_text(f.key, f.not, &f.value_text())).wrap_mode(egui::TextWrapMode::Extend)).on_hover_text(tip).clicked() {
                                    actions.edit_filter = Some(FilterEdit { at: k, key: f.key, not: f.not, value: f.value_text() });
                                }
                                ui.add_space(-ui.spacing().item_spacing.x + 2.0);
                                if ui.add(egui::Button::new("×")).on_hover_text("take this filter out").clicked() {
                                    actions.drop_filter = Some(k);
                                }
                                ui.add_space(6.0);
                            }
                        });
                        if found > 0 || !self.status.is_empty() || chips || tips {
                            ui.add_space(4.0);
                        }
                    }
                    if found > 0 {
                        ui.horizontal(|ui| {
                            let step = |ui: &mut egui::Ui, text: &str, tip: &str| ui.add(egui::Button::new(egui::RichText::new(text).size(18.0)).min_size(egui::vec2(28.0, 28.0))).on_hover_text(tip).clicked();
                            if step(ui, "‹", "the previous match (P or [)") {
                                actions.step = -1;
                            }
                            if step(ui, "›", "the next match, best first (N or ])") {
                                actions.step = 1;
                            }
                            if let Some(k) = at {
                                ui.label(format!("{} of", k + 1));
                            }
                            if on_path {
                                ui.label(format!("{found} photos"));
                                if !self.status.is_empty() {
                                    ui.label(theme::muted(&self.status));
                                }
                            } else {
                                // how many to mark, right where they are counted; the similarities they span in its tip
                                let most = self.map.as_ref().map_or(1000.0, |m| m.flags.len() as f32).max(11.0);
                                let tip = format!("{}\n\nMarks this many photos, best first, yellow to red. The model knows which fit better, not where fitting stops: only you can tell. All of them fit? Try more. Junk at the end? Lower it.", self.status);
                                let tip = tip.trim_start();
                                // from the right: Hide these, the number (dragged slowly or typed), and the slider in all the room left, so
                                // nothing leaves a gap and a pixel moves it little even over millions
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if theme::button(ui, egui::Button::new("Hide these")).on_hover_text("hide these matches: they turn faint, stay where they are, and no search marks them; the chip that appears brings them back").clicked() {
                                        actions.hide = true;
                                    }
                                    ui.add_space(4.0);
                                    let number = egui::DragValue::new(&mut self.wanted).range(10.0..=most).speed(0.2).fixed_decimals(0).suffix(" matches");
                                    let number = ui.add(number).on_hover_text(tip);
                                    ui.spacing_mut().slider_width = ui.available_width();
                                    let slider = ui.add(egui::Slider::new(&mut self.wanted, 10.0..=most).logarithmic(true).show_value(false)).on_hover_text(tip);
                                    if number.changed() || slider.changed() {
                                        self.wanted = self.wanted.round();
                                        actions.reblend = true;
                                    }
                                });
                            }
                        });
                    } else if !self.status.is_empty() {
                        // what happened, when there are no results to show it with (searching…, an error, what was hidden)
                        ui.add_space(4.0);
                        ui.label(theme::muted(&self.status));
                    }

                    // the hidden searches, opened from the badge in the box: a chip each
                    if let Some(m) = self.map.as_ref().filter(|_| chips) {
                        ui.add_space(4.0);
                        ui.horizontal_wrapped(|ui| {
                            for (k, h) in m.hidden.iter().enumerate() {
                                // the chip takes the search back into the box; its cross just brings the photos back
                                let chip = egui::Button::new(format!("{} ({})", h.what, h.photos.len())).wrap_mode(egui::TextWrapMode::Extend);
                                if ui.add(chip).on_hover_text("search this again, to change it and hide again (its photos come back meanwhile)").clicked() {
                                    actions.unhide = Some((k, true));
                                }
                                ui.add_space(-ui.spacing().item_spacing.x + 2.0);
                                if ui.add(egui::Button::new("×")).on_hover_text("show these photos again").clicked() {
                                    actions.unhide = Some((k, false));
                                }
                                ui.add_space(6.0);
                            }
                        });
                    }

                    // how to search, while the i is pointed at
                    if tips {
                        if found > 0 || !self.status.is_empty() || chips {
                            ui.add_space(6.0); // a gap only below something
                        }
                        ui.label(theme::muted(
                            "Describe what you want in a short phrase: \"woman on a bike in winter\". Use NOT in capitals to push things down: \
                             \"street at night NOT cars\". Or drop in a photo to find ones like it.\n\
                             Filters (+ filter, or typed in the box and a space after) become chips under it, and only photos they pass can match: \
                             name:greece looks in file and folder names, camera:pixel, date:2019-07, place:lviv in what the photo says about itself; \
                             year:2016,2018 is any of them, year:2016..2018 a range, NOT place:kyiv leaves out.\n\
                             Good at: things, places, scenes, light, weather, mood, style. Bad at: counting, left and right, who someone is.",
                        ));
                    }

                    // one photo searched for: that more can join it, and when that helps
                    if self.photos.len() == 1 {
                        ui.add_space(4.0);
                        ui.label(theme::muted(
                            "One photo brings everything in it: the background, the light, the colours. To find one thing (a place, an object, a style), \
                             add more photos of it, the more different the better: drop them here, or right-click a photo and Add to search. \
                             It doesn't recognise faces: a person is found by what they wear and where they are.",
                        ));
                    }
                });
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
                             Search compares your words, or photos you drop in (words or photos, not both), with every photo and highlights the best matches. \
                             NOT in capitals pushes things down: \"street NOT cars\".",
                        ));
                        ui.add_space(8.0);
                        egui::Grid::new("keys").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                            for (keys, what) in [
                                (&["scroll", "Q", "E"][..], "zoom"),
                                (&["drag", "W", "A", "S", "D"][..], "move"),
                                (&["F"][..], "photos like the one under the mouse"),
                                (&["Space"][..], "hold: a lens over the map (scroll: how much it magnifies)"),
                                (&["click"][..], "zoom to a photo, a region's name or a match"),
                                (&["right-click"][..], "a photo's menu: photos like it, a path to another photo, its file"),
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
                if let Some(i) = self.hovered.filter(|&i| i < m.data.n()) {
                    let (loaded, original) = m.streamer.loaded_px(i);
                    let on_screen = self.size_px() * m.data.size(i);
                    let text = format!("loaded {loaded} px, on screen {on_screen:.0} px, original {original} px");
                    ui.add(egui::Label::new(egui::RichText::new(text).monospace().color(theme::MUTED)).extend());
                }
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
                    let path = m.data.path(photo);
                    if !self.photos.is_empty() && !self.photos.iter().any(|(p, _)| *p == path) {
                        if theme::button(ui, egui::Button::new("Add to search")).on_hover_text("search for what this photo and the ones searched for share").clicked() {
                            actions.add_like = Some(photo);
                        }
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
        if actions.path_from.is_some() || actions.path_to.is_some() || actions.like.is_some() || actions.add_like.is_some() || actions.details.is_some() || actions.done {
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

    }
}

#[derive(Default)]
struct UiActions {
    ask: bool,
    /// The Layout menu: show the photos the way it now says.
    arrange: bool,
    reblend: bool,
    /// "Hide these": the search's matches are hidden.
    hide: bool,
    /// A hidden search's chip (true: back into search) or its cross (false: its photos just come back) was clicked.
    unhide: Option<(usize, bool)>,
    /// A prev / next button was pressed: -1 / 1.
    step: i32,
    clear_photo: bool,
    /// The cross after the search box: no words, no matches.
    clear_search: bool,
    /// The right-click menu: a path from or to this photo, photos like it.
    path_from: Option<usize>,
    path_to: Option<usize>,
    like: Option<usize>,
    add_like: Option<usize>,
    /// A filter to write or change: its place (usize::MAX: a new one) and its text.
    edit_filter: Option<FilterEdit>,
    /// The filter being written or changed is done (Enter, or clicked away from).
    filter_done: bool,
    /// A filter chip's cross: its place in `filters`.
    drop_filter: Option<usize>,
    /// A photo taken out of the search (its place in `photos`).
    drop_photo: Option<usize>,
    /// A photo searched for, by its place on the map: go to it.
    show_photo: Option<usize>,
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

/// A "Presets" button in the Layout menu, opening a list beside it: a submenu, so choosing one leaves the menu open (a combo box's
/// list is a popup of its own, and a click on it closed the menu).
fn presets(ui: &mut egui::Ui, list: impl FnOnce(&mut egui::Ui)) {
    let config = egui::containers::menu::MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
    ui.scope(|ui| {
        ui.set_style(ui.ctx().global_style()); // a button that looks like one, as theme::button
        let button = egui::Button::new("Presets").right_text(egui::containers::menu::SubMenuButton::RIGHT_ARROW).frame(true);
        egui::containers::menu::SubMenuButton::from_button(button).config(config).ui(ui, list);
    });
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
/// The folder every path shares ("/mnt/data/takeout-tagged/"), as a length: names are matched below it, so name: can't
/// match every photo by the disk they are on.
fn common_root<'a>(mut paths: impl Iterator<Item = &'a str>) -> usize {
    let Some(first) = paths.next() else { return 0 };
    let mut len = first.len();
    for p in paths {
        len = first.as_bytes()[..len].iter().zip(p.as_bytes()).take_while(|(a, b)| a == b).count();
    }
    // back to the end of a folder, so a name isn't cut
    first.as_bytes()[..len].iter().rposition(|&b| b == b'/' || b == b'\\').map_or(0, |k| k + 1)
}

/// Which photo of the map is this file, if any (one pass over them all, for a click).
fn find_photo(data: &data::MapData, path: &std::path::Path) -> Option<usize> {
    let rel = path.strip_prefix(&data.bundle).ok();
    data.photos.iter().position(|p| if p.file.is_empty() { std::path::Path::new(&p.source) == path } else { rel.is_some_and(|r| r == std::path::Path::new(&p.file)) })
}

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

    #[test]
    fn names_are_matched_below_the_shared_folder() {
        let paths = ["/mnt/data/takeout/Trip 2019/a.jpg", "/mnt/data/takeout/Home/b.jpg", "/mnt/data/takeout/Tr/c.jpg"];
        assert_eq!(common_root(paths.into_iter()), "/mnt/data/takeout/".len());
        assert_eq!(common_root(["/x/Київ/a.jpg", "/x/Київець/b.jpg"].into_iter()), "/x/".len());
        assert_eq!(common_root(std::iter::empty()), 0);
    }

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
