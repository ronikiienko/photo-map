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
mod mapui;
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
mod player;
mod places;
mod query;
mod screen;
mod script;
#[cfg(test)]
mod shots;
mod search;
mod searchbar;
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
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;
use stream::Streamer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

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
/// A video's mark (draw_video_marks) fades in as its tile's longer side goes from the first to the second (points) ...
const VIDEO_MARK_FROM: (f32, f32) = (36.0, 56.0);
/// ... and says how long it is (or when a copy's frame is) from this size.
const VIDEO_TIME_FROM: f32 = 96.0;

/// A video on screen with its mark (Viewer::video_tiles): on the map, or a match's copy beside it (its position).
struct VideoTile {
    photo: usize,
    copy: Option<usize>,
    /// Where it is on screen, and its longer side (points).
    rect: egui::Rect,
    long: f32,
    /// What its mark says: how long the video is, or when a copy's frame is; and where it starts playing.
    time: f32,
    start: f32,
}

impl VideoTile {
    /// Where its mark is: a pill in the bottom-left corner, as wide as the play sign and a time.
    fn pill(&self) -> egui::Rect {
        let side = (self.rect.width().min(self.rect.height()) * 0.16).clamp(14.0, 22.0);
        let gap = side * 0.3;
        let width = if self.long >= VIDEO_TIME_FROM { side * (1.2 + 0.36 * clock(self.time).len() as f32) } else { side };
        egui::Rect::from_min_size(egui::pos2(self.rect.min.x + gap, self.rect.max.y - gap - side), egui::vec2(width, side))
    }
}

/// A video's mark in `pill`: a play sign (two bars while it plays) and the time, faded by `fade`.
fn draw_pill(painter: &egui::Painter, pill: egui::Rect, time: Option<String>, playing: bool, fade: f32) {
    let side = pill.height();
    painter.rect_filled(pill, side * 0.5, egui::Color32::from_black_alpha((150.0 * fade) as u8));
    let white = egui::Color32::WHITE.gamma_multiply(fade);
    let c = egui::pos2(pill.min.x + side * 0.53, pill.center().y);
    let r = side * 0.27;
    if playing {
        for dx in [-0.45, 0.45] {
            painter.rect_filled(egui::Rect::from_center_size(egui::pos2(c.x + dx * r, c.y), egui::vec2(r * 0.55, r * 2.0)), 1.0, white);
        }
    } else {
        let triangle = vec![egui::pos2(c.x - r * 0.8, c.y - r), egui::pos2(c.x + r, c.y), egui::pos2(c.x - r * 0.8, c.y + r)];
        painter.add(egui::Shape::convex_polygon(triangle, white, egui::Stroke::NONE));
    }
    if let Some(text) = time {
        let galley = painter.layout_no_wrap(text, egui::FontId::proportional(side * 0.62), white);
        painter.galley(egui::pos2(pill.min.x + side * 0.92, pill.center().y - galley.size().y * 0.5), galley, white);
    }
}

/// A video's big play sign in the middle of its tile `rect`, in a dark disc.
fn draw_play_sign(painter: &egui::Painter, rect: egui::Rect) {
    let radius = (rect.width().min(rect.height()) * 0.16).clamp(14.0, 34.0);
    let c = rect.center();
    painter.circle_filled(c, radius, egui::Color32::from_black_alpha(150));
    let r = radius * 0.42;
    let triangle = vec![egui::pos2(c.x - r * 0.7, c.y - r), egui::pos2(c.x + r, c.y), egui::pos2(c.x - r * 0.7, c.y + r)];
    painter.add(egui::Shape::convex_polygon(triangle, egui::Color32::WHITE, egui::Stroke::NONE));
}

/// Seconds as a clock: "0:07", "12:30", "1:02:03".
fn clock(seconds: f32) -> String {
    let s = seconds.max(0.0).round() as u64;
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60) } else { format!("{}:{:02}", s / 60, s % 60) }
}
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
    /// The last search's cosine with every photo's picture (empty: no search); a video's is its best frame's.
    sims: Vec<f32>,
    /// The picture each photo is shown by among the last search's matches: itself, or a video's best frame (search::best_frames).
    best: Vec<usize>,
    /// The last search's scores of the videos' frames (as `sims`, frame by frame; empty: none), and how many photos it marks
    /// (the count slider's): a video's moments are as many of its best frames.
    frame_sims: Vec<f32>,
    wanted: usize,
    /// The picture each copy beside the world shows: the matches' (in the order of `order`), then the moments'.
    pictures: Vec<usize>,
    /// The photos that match the search, best first, and which of them the last next/previous step went to.
    order: Vec<usize>,
    at: Option<usize>,
    /// Where each copy sits beside the world (the matches', then the moments'), and a copy's side (world units).
    copies: Vec<data::Point>,
    copy_side: f32,
    /// "Moments that match": the video whose frames that match the search are shown right of the matches, best first. It stays
    /// while the search changes (each new one picks its frames again), until the search is cleared.
    moments: Option<usize>,
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

    /// Put copies of these photos beside the world, in this order (the search's matches, or a path). `best`: a video's copy
    /// shows the frame that matched the search best, else its poster.
    fn show_beside(&mut self, queue: &wgpu::Queue, photos: Vec<usize>, best: bool) {
        self.order = photos;
        self.at = None;
        if !best || self.order.is_empty() {
            self.moments = None; // they stay through new searches (to try them on the same video), and go with the search
        }
        self.place_copies(queue, best);
    }

    /// The copies beside the world: the matches' (`best`: a video's shows its best frame), then the moments' (show_moments).
    fn place_copies(&mut self, queue: &wgpu::Queue, best: bool) {
        self.pictures = self.order.iter().map(|&i| if best { self.best[i] } else { i }).collect();
        (self.copies, self.copy_side) = beside::place(self.order.len());
        let moments = self.moments.map_or(Vec::new(), |video| self.moments_of(video));
        let right = self.copies.iter().map(|p| p.x).fold(beside::LEFT, f32::max) + self.copy_side / beside::FILL / 2.0;
        self.copies.extend(beside::place_after(moments.len(), self.copy_side, right));
        self.pictures.extend(moments);
        self.renderer.set_copies(queue, &self.data, &self.pictures, &self.copies);
    }

    /// This video's frames that match the last search best, as many as it marks photos (whether or not they made it into the
    /// matches), best first: their picture ids.
    fn moments_of(&self, video: usize) -> Vec<usize> {
        let n = self.data.n();
        let mut frames: Vec<usize> = (0..self.frame_sims.len()).filter(|&k| self.data.frames[k].video == video).collect();
        frames.sort_by(|&a, &b| self.frame_sims[b].total_cmp(&self.frame_sims[a]));
        frames.truncate(self.wanted);
        frames.into_iter().map(|k| n + k).collect()
    }

    /// "Moments that match": this video's best frames for the search, right of the matches (another video's go).
    fn show_moments(&mut self, queue: &wgpu::Queue, video: usize) {
        self.moments = Some(video);
        self.place_copies(queue, true);
    }

    /// Can this photo show its moments: a video, while a search by words or photos is on?
    fn has_moments(&self, photo: usize) -> bool {
        self.data.photos[photo].video.is_some() && !self.frame_sims.is_empty() && self.trail.is_none() && !self.order.is_empty()
    }

    /// The photo copy `k` beside the world is of: a match, or the video of the moments.
    fn copy_photo(&self, k: usize) -> usize {
        match self.order.get(k) {
            Some(&i) => i,
            None => self.moments.expect("a copy past the matches is a moment"),
        }
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
    /// A window, or none (screen.rs).
    screen: screen::Screen,
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
    /// The video playing in its tile, if one is (player.rs).
    player: Option<player::Player>,
    /// A video's frame kept in the map (frames.json), shown while its player goes to a time: which frame, and its picture.
    still: Option<(usize, egui::TextureHandle)>,
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
    /// The right-click menu: on this photo, showing this picture (data.rs, picture ids), on this copy beside the world (if
    /// it was opened on one), at this place (points).
    menu: Option<(usize, usize, Option<usize>, egui::Pos2)>,
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
    answers: Vec<Option<(Vec<f32>, Vec<f32>)>>,
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
    /// Small pictures of the map's photos the empty search box offers (None: unreadable).
    idea_thumbs: HashMap<usize, Option<egui::TextureHandle>>,
    /// The Hidden line is open: a chip for each hidden search.
    hidden_open: bool,
    /// How to search shows: the i at the box's end is pointed at.
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
    /// PHOTO_MAP_SCRIPT (script.rs): the steps being run, the keys and letters they type for egui, and the frame to save.
    script: Option<script::Script>,
    /// How many photos the search showing can mark (not hidden, passing its filters): the most its slider goes to.
    can_match: usize,
    /// F3: how fast frames are drawn and what is loaded, in the corner.
    debug: bool,
    script_events: Vec<egui::Event>,
    shot: Option<PathBuf>,
    /// The script has ended (without a window: the run is over).
    quit: bool,
}

impl Viewer {
    /// `make_screen`: the window (or none) that egui's input comes from, made with the app's egui context.
    fn new(gpu: Gpu, make_screen: impl FnOnce(&egui::Context) -> screen::Screen, library: Library, first: Option<PathBuf>) -> Self {
        let ctx = egui::Context::default();
        theme::apply(&ctx);
        let screen = make_screen(&ctx);
        let wake = screen.waker();
        let waker = || {
            let wake = wake.clone();
            move || wake()
        };
        let engine = search::Engine::start(library.root(), waker());
        let runtime = manager::Manager::new(library.root(), waker());
        let store = modelstore::ModelStore::new(library.root(), waker());
        let examples = examples::Examples::new(library.root(), waker());
        let mut renderer = egui_wgpu::Renderer::new(&gpu.device, gpu.config.format, Default::default());
        renderer.callback_resources.insert(player::VideoGpu::new(&gpu.device, gpu.config.format));
        let size = screen.size();
        let camera = Camera::new(Vec2::new(size.0 as f32, size.1 as f32));
        let mut viewer = Self {
            screen,
            gpu,
            ui: Ui { ctx, renderer },
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
            player: None,
            still: None,
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
            idea_thumbs: HashMap::new(),
            hidden_open: false,
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
            script: script::Script::from_env(),
            debug: false,
            can_match: 0,
            script_events: Vec::new(),
            shot: None,
            quit: false,
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
        self.player = None;
        self.map = None; // free the old map's GPU memory first
        let started = Instant::now();
        let streamer = Streamer::new(&self.gpu.device, &self.gpu.queue, &data);
        let renderer = MapRenderer::new(&self.gpu, &data, &streamer);
        eprintln!("{}: {} photos opened in {:.1}s", data.manifest.name, data.n(), started.elapsed().as_secs_f32());
        self.screen.set_title(&format!("{} — photo map", data.manifest.name));
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
        let vectors = search::Vectors::open(&path, n, data.frames.len(), data.manifest.dim());
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
        self.idea_thumbs.clear();
        self.ideas.set_photos(data.n());
        self.map = Some(OpenMap {
            path,
            data,
            renderer,
            streamer,
            flags: vec![0; n],
            flags_dirty: true,
            photo_world,
            sims: Vec::new(),
            frame_sims: Vec::new(),
            wanted: 0,
            best: (0..n).collect(),
            pictures: Vec::new(),
            order: Vec::new(),
            at: None,
            copies: Vec::new(),
            copy_side: 0.0,
            moments: None,
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
        let wake = self.screen.waker();
        move || wake()
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
                    unread_videos: state.as_ref().map_or_else(|| b.job.unread_videos.clone(), |s| s.unread_videos.clone()),
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
            Action::InstallFfmpeg => self.runtime.start(&manager::Manager::ffmpeg_offer()),
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
        self.player = None;
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
        self.lens.view(self.camera.center, self.camera.scale, self.camera.viewport, self.screen.scale())
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

    /// The picture under the mouse: the hovered photo's, or what its copy beside the world shows (a video's best frame).
    fn hovered_picture(&self) -> Option<usize> {
        match (self.hovered_copy, &self.map) {
            (Some(k), Some(m)) => m.pictures.get(k).copied(),
            _ => self.hovered,
        }
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
        let small = img.thumbnail(512, 512).to_rgba8(); // big enough to show enlarged under the mouse
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
                    filters.iter().all(|f| f.passes(name, p))
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
                    m.frame_sims.clear();
                    m.trail = None;
                    m.show_beside(&self.gpu.queue, Vec::new(), false);
                    self.status = if unknown {
                        "this map knows no cameras, dates or places: make it again to read them".into()
                    } else {
                        format!("no photo fits {filters_text}")
                    };
                    return;
                }
                let n = photos.len();
                let m = self.map.as_mut().unwrap();
                m.sims = vec![0.0; n];
                m.frame_sims.clear();
                m.best = (0..n).collect();
                self.apply_search(); // the camera stays where it is: the marks light up where they are
                return;
            }
        }
        let Some(m) = &mut self.map else { return };
        if text.is_empty() && self.photos.is_empty() {
            m.flags.fill(0);
            m.flags_dirty = true;
            m.sims.clear();
            m.frame_sims.clear();
            m.trail = None;
            m.show_beside(&self.gpu.queue, Vec::new(), false);
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
                    self.answers[k] = Some((vectors.cosines(&v), vectors.frame_cosines(&v)));
                    if self.asked.iter().any(Option::is_some) {
                        continue; // the other terms are still on their way
                    }
                    // the photos' scores, and the frames' the same way (a video then scores as its best frame)
                    let (cos, frame_cos): (Vec<Vec<f32>>, Vec<Vec<f32>>) = std::mem::take(&mut self.answers).into_iter().flatten().unzip();
                    m.sims = self.logic.combine(&cos);
                    m.best = (0..m.sims.len()).collect();
                    m.frame_sims.clear();
                    if !m.data.frames.is_empty() {
                        m.frame_sims = self.logic.combine(&frame_cos);
                        m.best = search::best_frames(&mut m.sims, &m.frame_sims, &m.data.frames);
                    }
                    self.apply_search(); // the camera stays where it is: the marks light up where they are
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
        self.can_match = shown;
        if shown == 0 && self.names.is_some() {
            m.flags.fill(0);
            m.flags_dirty = true;
            m.show_beside(&self.gpu.queue, Vec::new(), false);
            self.status = "no photo fits the filters".into();
            return None;
        }
        if shown == 0 {
            m.flags.fill(0);
            m.flags_dirty = true;
            m.show_beside(&self.gpu.queue, Vec::new(), false);
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
        m.wanted = k;
        for i in 0..n {
            m.flags[i] = if score[i] >= cut { 1 + ((score[i] - cut) / span).clamp(0.0, 1.0).mul_add(254.0, 0.5) as u32 } else { 0 };
        }
        let mut order: Vec<usize> = (0..n).filter(|&i| m.flags[i] > 0).collect();
        order.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
        m.trail = None;
        m.show_beside(&self.gpu.queue, order, true);
        // how many is already in "6 of 100" beside it
        self.status = if self.names_only { format!("{}: {shown} photos", self.shown) } else { format!("{}: similarity {cut:.2} to {best:.2}", self.shown) };
        m.flags_dirty = true;
        Some(m.beside_box())
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
        m.frame_sims.clear();
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
        m.show_beside(&self.gpu.queue, path, false);
        self.camera.fly_to(self.camera.fit(m.beside_box().0, m.beside_box().1));
    }

    /// F over a photo: the same as dropping that photo's file on the window (it goes in the search box and is searched), instead of
    /// the photos searched for or (`add`) besides them.
    /// "Moments that match": the video's best frames for the search, right of the matches, and the camera on them.
    fn show_moments(&mut self, video: usize) {
        let Some(m) = &mut self.map else { return };
        m.show_moments(&self.gpu.queue, video);
        let moments = &m.copies[m.order.len()..];
        if moments.is_empty() {
            return;
        }
        let half = Vec2::splat(m.copy_side / beside::FILL / 2.0);
        let (lo, hi) = moments.iter().map(|p| Vec2::new(p.x, p.y)).fold((Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)), |(lo, hi), p| (lo.min(p - half), hi.max(p + half)));
        self.camera.fly_to(self.camera.fit(lo, hi));
    }

    /// Search with this picture (a photo, or a video's frame: data.rs, picture ids).
    fn ask_like(&mut self, picture: usize, add: bool) {
        let Some(file) = self.map.as_ref().map(|m| m.data.picture_path(picture)) else { return };
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
        let wanted_scale = STEP_TO * self.screen.scale() / m.photo_world;
        self.camera.fly_to((Vec2::new(p.x, p.y), self.camera.scale.max(wanted_scale)));
    }

    /// Borderless, like mote-control: exclusive forces a modeset and Wayland ignores it.
    fn toggle_fullscreen(&self) {
        self.screen.toggle_fullscreen();
    }

    fn key(&mut self, key: KeyCode, down: bool) {
        if let (true, KeyCode::F11) = (down, key) {
            return self.toggle_fullscreen();
        }
        if let (true, KeyCode::F3) = (down, key) {
            self.debug = !self.debug;
            return;
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
            if let (true, Some(i)) = (down, self.hovered_picture()) {
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
        let egui_took = self.screen.on_event(&self.ui.ctx, &event);
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
                let ppp = self.screen.scale();
                self.menu = self.hovered.zip(self.cursor).map(|(i, c)| (i, self.hovered_picture().unwrap_or(i), self.hovered_copy, egui::pos2(c.x / ppp, c.y / ppp)));
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
                self.screen.redraw();
            }
            WindowEvent::HoveredFileCancelled => {
                self.home.set_drag_over(false);
                self.screen.redraw();
            }
            // one event per dropped file: kept until the next frame, then handed over together
            WindowEvent::DroppedFile(path) if self.page == Page::Maps => {
                self.dropped.push(path);
                self.screen.redraw();
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
            WindowEvent::Focused(true) => self.screen.redraw(),
            WindowEvent::Occluded(hidden) => {
                self.hidden = hidden;
                self.screen.redraw();
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn frame(&mut self) {
        if !self.hidden {
            self.screen.redraw(); // first, so that the frames go on when the surface can't be drawn to
        }
        let size = self.screen.size();
        if size != (self.gpu.config.width, self.gpu.config.height) && size.0 > 0 && size.1 > 0 {
            self.gpu.resize(size.0, size.1);
            self.camera.viewport = Vec2::new(size.0 as f32, size.1 as f32);
        }
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_times.push(dt);
        if self.frame_times.len() > 60 {
            self.frame_times.remove(0);
        }
        self.run_script();
        // what is fitted stays below the search box as drawn last frame (it grows with chips), the results row it gets once a search is
        // answered, and the heading over the matches
        let bar = self.ui.ctx.memory(|m| m.area_rect(egui::Id::new("top"))).map_or(0.0, |r| r.bottom());
        self.camera.top = (bar + 56.0).max(84.0) * self.screen.scale();
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
            (_, Some(k), Some(m)) => Some(m.copy_photo(k)),
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

        let Some(frame) = self.gpu.target() else {
            let size = self.screen.size();
            self.gpu.resize(size.0, size.1); // the surface wants configuring again
            return;
        };
        let view = frame.texture().create_view(&Default::default());
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
                (Some(Lifted::Copy(k)), Some(m)) if k < m.copies.len() => k as u32,
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
                let copies = (&m.pictures[..], &m.copies[..], cam.copy_px);
                m.streamer.update(&self.gpu.queue, &m.data, view_min, view_max, self.camera.center, size_px, copies, lens.map(|l| l.1));
                m.renderer.draw(&self.gpu.queue, &mut encoder, &view, cam, self.smooth, lens.map(|l| l.0));
            }
            _ => gpu::clear(&mut encoder, &view),
        }
        self.draw_ui(&mut encoder, &view);
        self.gpu.queue.submit([encoder.finish()]);
        if let Some(path) = self.shot.take() {
            script::save_frame(&self.gpu.device, &self.gpu.queue, frame.texture(), self.gpu.config.format, &path);
        }
        if let gpu::Target::Surface(frame) = frame {
            self.screen.pre_present();
            self.gpu.queue.present(frame);
        }
        self.screen.redraw();
    }

    /// The script's step for this frame (script.rs), done as if by hand: through the same events the window sends.
    fn run_script(&mut self) {
        let open = self.map.is_some() && self.opening.is_none();
        let searched = self.asked.iter().all(Option::is_none) && !self.status.starts_with("searching") && !self.status.starts_with("loading the search model");
        let Some(step) = self.script.as_mut().and_then(|s| s.next(open, searched)) else { return };
        let ppp = self.screen.scale() as f64;
        let device_id = winit::event::DeviceId::dummy();
        let at = |x: f32, y: f32| winit::dpi::PhysicalPosition::new(x as f64 * ppp, y as f64 * ppp);
        match step {
            script::Step::Size(w, h) => self.screen.request_size(w, h),
            script::Step::Open(name) => self.open(self.library.dir().join(format!("{name}.photomap"))),
            script::Step::Move(x, y) => self.event(WindowEvent::CursorMoved { device_id, position: at(x, y) }),
            script::Step::Photo(i) => {
                let Some(p) = self.map.as_ref().and_then(|m| m.data.points.get(i).copied()) else { return };
                let on_screen = self.camera.to_screen(Vec2::new(p.x, p.y));
                self.event(WindowEvent::CursorMoved { device_id, position: winit::dpi::PhysicalPosition::new(on_screen.x as f64, on_screen.y as f64) });
            }
            script::Step::Click(place, button) => {
                if let Some((x, y)) = place {
                    self.event(WindowEvent::CursorMoved { device_id, position: at(x, y) });
                }
                self.event(WindowEvent::MouseInput { device_id, state: ElementState::Pressed, button });
                self.event(WindowEvent::MouseInput { device_id, state: ElementState::Released, button });
            }
            script::Step::Scroll(lines) => {
                self.event(WindowEvent::MouseWheel { device_id, delta: MouseScrollDelta::LineDelta(0.0, lines), phase: winit::event::TouchPhase::Moved });
            }
            script::Step::Focus => self.ui.ctx.memory_mut(|m| m.request_focus(egui::Id::new("search-box"))),
            script::Step::Type(text) => self.script_events.push(egui::Event::Text(text)),
            script::Step::Key(key) => {
                for pressed in [true, false] {
                    self.script_events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE });
                }
            }
            script::Step::AppKey(key) => {
                self.key(key, true);
                self.key(key, false);
            }
            script::Step::Shot(name) => self.shot = self.script.as_ref().map(|s| s.dir.join(format!("{name}.png"))),
            script::Step::Maps => self.go_to_maps(),
            // SAFETY: `_exit` only ends the process (as main does at the end)
            script::Step::Quit if self.screen.is_headless() => self.quit = true,
            script::Step::Quit => unsafe { libc::_exit(0) },
        }
    }

    /// A map being opened: one card in the middle.
    fn opening_card(&self, ctx: &egui::Context) {
        let Some((path, _, progress)) = &self.opening else { return };
        let (text, fraction) = progress.now();
        let name = path.file_stem().map_or(String::new(), |s| s.to_string_lossy().into_owned());
        mapui::opening(ctx, &name, &text, fraction);
    }

    fn draw_ui(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let raw = self.screen.egui_input(&mut self.script_events);
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
        if let Some(file) = &actions.reveal {
            files::show_in_folder(file);
        }
        if let Some(file) = &actions.open_file {
            files::open(file);
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
        if let Some(video) = actions.moments {
            self.show_moments(video);
        }
        if let Some(k) = actions.go_to_copy {
            self.go_to_copy(k);
        }
        if actions.close_moments
            && let Some(m) = &mut self.map
        {
            m.moments = None;
            m.place_copies(&self.gpu.queue, m.trail.is_none());
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
        if let Some(videos) = actions.set_kind {
            self.finish_filter();
            let f = query::Filter::kind(videos);
            match self.filters.iter().position(|f| f.key == query::Key::Kind) {
                Some(k) => self.filters[k] = f,
                None => self.filters.push(f),
            }
            self.ask();
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

        self.screen.platform_output(output.platform_output);
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
        // the boxes over the map (last frame's): no name is drawn under them
        let mut taken: Vec<egui::Rect> = ["top", "where", "look", "help", "help-card"].iter().filter_map(|id| ctx.memory(|mem| mem.area_rect(egui::Id::new(*id)))).collect();
        // a name already drawn near here (a finer region often has its parent's name): once is enough
        let mut names: Vec<(&str, egui::Pos2)> = Vec::new();
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
                let near = 2.5 * size * l.name.chars().count().max(4) as f32;
                let again = names.iter().any(|(n, p)| *n == l.name && p.distance(rect.center()) < near);
                if again || !screen.contains_rect(rect) || taken.iter().any(|t| t.intersects(rect.expand(6.0))) || self.under_lens(rect, ppp) {
                    continue;
                }
                taken.push(rect);
                names.push((&l.name, rect.center()));
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

    /// A click on the map (a press that hardly moved): on a video (big enough to be marked), it plays or pauses; on a region
    /// name, it zooms to the region. A photo does nothing: moving is the wheel's, dragging's and N / P's (a match's copy's menu
    /// has Go to on the map).
    fn click_at(&mut self) {
        let Some(c) = self.cursor else { return };
        let ppp = self.screen.scale();
        let at = egui::pos2(c.x / ppp, c.y / ppp);
        if let Some(t) = self.video_tiles(ppp).into_iter().find(|t| t.rect.contains(at)) {
            match &mut self.player {
                Some(p) if p.photo == t.photo && p.copy == t.copy => {
                    if p.playing() { p.pause() } else { p.play() }
                }
                _ => self.play(t),
            }
            return;
        }
        if let Some(&(_, level, index)) = self.label_rects.iter().find(|(rect, _, _)| rect.contains(at)) {
            self.go_to_region(level, index);
        }
    }

    /// Fly to copy `k` beside the world's photo: a match (it becomes the current one, as N / P), or a moment's video.
    fn go_to_copy(&mut self, k: usize) {
        let Some(m) = &self.map else { return };
        if k < m.order.len() {
            self.go_to_match(k);
        } else if k < m.copies.len() {
            self.go_to_photo(m.copy_photo(k));
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
        if let (Some(video), Some(first)) = (m.moments, m.copies.get(m.order.len())) {
            let top = self.camera.to_screen(Vec2::new(first.x - m.copy_side / 2.0, first.y - m.copy_side / 2.0)) / ppp;
            let count = m.copies.len() - m.order.len();
            let heading = format!("{count} {} of {}, best first", if count == 1 { "moment" } else { "moments" }, m.data.photos[video].title);
            let galley = painter.layout_no_wrap(heading, egui::FontId::proportional(16.0), theme::MUTED);
            painter.galley(egui::pos2(top.x, top.y - galley.size().y - 6.0), galley, theme::MUTED);
        }
        if let Some(k) = m.at {
            outline(m.order[k], to_screen(m.copies[k]), copy_side, 2.0, true);
        }
        match (self.hovered_copy, self.hovered) {
            (Some(k), _) => outline(m.copy_photo(k), to_screen(m.data.points[m.copy_photo(k)]), photo_side, 2.0, false),
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

    /// The videos on screen big enough to be marked (VIDEO_MARK_FROM): on the map, and the matches' copies beside it.
    fn video_tiles(&self, ppp: f32) -> Vec<VideoTile> {
        let Some(m) = &self.map else { return Vec::new() };
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(self.camera.viewport.x, self.camera.viewport.y) / ppp);
        let mut tiles = Vec::new();
        let mut add = |photo: usize, copy: Option<usize>, at: Vec2, form: Vec2, long: f32, time: f32, start: f32| {
            if long < VIDEO_MARK_FROM.0 {
                return;
            }
            let rect = egui::Rect::from_center_size(egui::pos2(at.x, at.y), egui::vec2(form.x, form.y) * long);
            if screen.intersects(rect) && !self.under_lens(rect, ppp) {
                tiles.push(VideoTile { photo, copy, rect, long, time, start });
            }
        };
        // on the map: how long each video is; it plays from the start
        let long = self.size_px() / ppp;
        if long * m.data.max_size() >= VIDEO_MARK_FROM.0 {
            let (view_min, view_max) = (self.camera.to_world(Vec2::ZERO), self.camera.to_world(self.camera.viewport));
            m.data.grid.within(&m.data.points, view_min, view_max, |i| {
                if let Some(duration) = m.data.photos[i].video.map(|v| v.duration).filter(|_| !m.hidden_mask[i]) {
                    let at = self.camera.to_screen(Vec2::new(m.data.points[i].x, m.data.points[i].y)) / ppp;
                    let shape = m.data.shape(i);
                    add(i, None, at, shape / shape.max_element(), long * shape.max_element(), duration, 0.0);
                }
            });
        }
        // the copies (matches and moments): when their frame is, and it plays from there (a video's poster: how long it is, from
        // the start)
        let copy_long = m.copy_side * self.camera.scale / ppp;
        for (k, (&picture, p)) in m.pictures.iter().zip(&m.copies).enumerate() {
            let i = m.copy_photo(k);
            let Some(duration) = m.data.photos[i].video.map(|v| v.duration) else { continue };
            let frame = picture.checked_sub(m.data.n()).map(|f| m.data.frames[f].time);
            let at = self.camera.to_screen(Vec2::new(p.x, p.y)) / ppp;
            add(i, Some(k), at, m.data.picture_form(picture), copy_long, frame.unwrap_or(duration), frame.unwrap_or(0.0));
        }
        tiles
    }

    /// A video is marked in its bottom-left corner once it is big enough to tell (VIDEO_MARK_FROM): a play sign, and from
    /// VIDEO_TIME_FROM how long it is. A copy of one beside the world (a search's match) says when its frame is in the video.
    /// Under the mouse, a big play sign in its middle: a click anywhere on it plays it. The one playing has its controls
    /// instead (player_ui).
    fn draw_video_marks(&self, ctx: &egui::Context) {
        let ppp = ctx.pixels_per_point();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "video marks".into()));
        let mouse = self.cursor.map(|c| egui::pos2(c.x / ppp, c.y / ppp));
        for t in self.video_tiles(ppp) {
            if self.player.as_ref().is_some_and(|p| p.photo == t.photo && p.copy == t.copy) {
                continue;
            }
            let fade = ((t.long - VIDEO_MARK_FROM.0) / (VIDEO_MARK_FROM.1 - VIDEO_MARK_FROM.0)).clamp(0.0, 1.0);
            let time = (t.long >= VIDEO_TIME_FROM).then(|| clock(t.time));
            draw_pill(&painter, t.pill(), time, false, fade);
            if mouse.is_some_and(|m| t.rect.contains(m)) {
                draw_play_sign(&painter, t.rect);
            }
        }
    }

    /// The video playing: its newest frame over its tile, and, under the mouse or while paused, its controls along the tile's
    /// bottom (play or pause, how far, and a bar to go anywhere in it); paused and under the mouse also a big play sign in its
    /// middle (a click on the tile pauses or plays it). It stops when its tile leaves the screen or gets too small to mark.
    fn player_ui(&mut self, ctx: &egui::Context) {
        let Some(p) = &self.player else { return };
        let ppp = ctx.pixels_per_point();
        let Some(tile) = self.video_tiles(ppp).into_iter().find(|t| t.photo == p.photo && t.copy == p.copy) else {
            self.player = None;
            return;
        };
        let Some(p) = &mut self.player else { return };
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, "video".into()));
        let corner = (0.04 * tile.rect.width().min(tile.rect.height())).min(10.0 / ppp); // as the map's own corners (map.wgsl)
        if let Some(frame) = p.paint(tile.rect, corner) {
            painter.add(frame);
        }
        if p.playing() {
            ctx.request_repaint();
        }
        let hovered = self.cursor.is_some_and(|c| tile.rect.contains(egui::pos2(c.x / ppp, c.y / ppp)));
        // the controls (not while it plays unwatched): the pill (play or pause, and the time), and the bar beside it
        let mut toggle = false;
        let mut seek = None;
        let mut dragging = None;
        if hovered || !p.playing() {
            let pill = tile.pill();
            let bar = egui::Rect::from_min_max(egui::pos2(pill.max.x + pill.height() * 0.5, pill.min.y), egui::pos2(tile.rect.max.x - pill.height() * 0.3, pill.max.y));
            let area = pill.union(bar);
            egui::Area::new(egui::Id::new("video controls")).fixed_pos(area.min).order(egui::Order::Middle).show(ctx, |ui| {
                let (_, response) = ui.allocate_exact_size(area.size(), egui::Sense::hover());
                let pill_r = ui.interact(pill, egui::Id::new("video pill"), egui::Sense::click());
                let bar_r = ui.interact(bar, egui::Id::new("video bar"), egui::Sense::click_and_drag());
                toggle = pill_r.clicked();
                let fraction = |x: f32| ((x - bar.min.x) / bar.width().max(1.0)).clamp(0.0, 1.0);
                dragging = bar_r.interact_pointer_pos().filter(|_| bar_r.dragged() || bar_r.is_pointer_button_down_on()).map(|at| fraction(at.x));
                if bar_r.drag_stopped() || bar_r.clicked() {
                    seek = bar_r.interact_pointer_pos().map(|at| fraction(at.x) * p.duration);
                }
                let shown = dragging.map_or(p.position(), |f| f * p.duration);
                let painter = ui.painter();
                draw_pill(painter, pill, Some(clock(shown)), p.playing(), 1.0);
                // the bar: a track, how far it is, and a knob there (bigger under the mouse)
                let y = bar.center().y;
                let track = egui::Rect::from_min_max(egui::pos2(bar.min.x, y - 2.0), egui::pos2(bar.max.x, y + 2.0));
                painter.rect_filled(track.expand(3.0), 5.0, egui::Color32::from_black_alpha(130));
                painter.rect_filled(track, 2.0, egui::Color32::from_white_alpha(70));
                let x = bar.min.x + bar.width() * (shown / p.duration.max(1e-3)).clamp(0.0, 1.0);
                painter.rect_filled(egui::Rect::from_min_max(track.min, egui::pos2(x, track.max.y)), 2.0, egui::Color32::WHITE);
                let hot = bar_r.hovered() || dragging.is_some();
                painter.circle_filled(egui::pos2(x, y), if hot { 7.0 } else { 5.0 }, egui::Color32::WHITE);
                if pill_r.hovered() || bar_r.hovered() {
                    ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                let _ = response;
            });
        }
        if toggle {
            if p.playing() { p.pause() } else { p.play() }
        }
        if let Some(t) = seek {
            p.seek(t);
        }
        // until ffmpeg's frame from a new time comes, and while the bar is dragged: the map's own frame nearest that time (made
        // small when the map was), at once
        let still = dragging.map(|f| f * p.duration).or_else(|| p.waiting().then(|| p.position()));
        if let (Some(t), Some(m)) = (still, &self.map) {
            let near = m.data.frames.iter().enumerate().filter(|(_, f)| f.video == p.photo).min_by(|a, b| (a.1.time - t).abs().total_cmp(&(b.1.time - t).abs())).map(|(k, _)| k);
            if let Some(k) = near {
                if self.still.as_ref().is_none_or(|s| s.0 != k) {
                    self.still = photo::load(&m.data.bundle.join(&m.data.frames[k].file)).map(|img| {
                        let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
                        (k, ctx.load_texture("video still", color, egui::TextureOptions::LINEAR))
                    });
                }
                if let Some((_, tex)) = &self.still {
                    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                    // a pixel bigger, as the video (video.wgsl): over the poster's soft edge
                    let px = 1.0 / ppp;
                    painter.add(egui::epaint::RectShape::filled(tile.rect.expand(px), corner + px, egui::Color32::WHITE).with_texture(tex.id(), uv));
                }
            }
        }
        if hovered && !p.playing() {
            draw_play_sign(&painter, tile.rect); // over its picture; not while it plays (it would be in the way)
        }
    }

    /// Play this video tile: on the map from the start, a copy beside the world from its frame.
    fn play(&mut self, t: VideoTile) {
        let (Some(m), Some(ffmpeg)) = (&self.map, self.runtime.ffmpeg.clone()) else {
            self.status = "videos are played with ffmpeg: download it in Runtime (on the Maps page)".into();
            return;
        };
        let photo = &m.data.photos[t.photo];
        let video = files::original(&m.data, t.photo);
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(self.wake());
        self.player = Some(player::Player::new(&ffmpeg, &video, t.photo, t.copy, photo.video.map_or(t.time, |v| v.duration), t.start, photo.video.map_or((photo.width, photo.height), |v| (v.width, v.height)), wake));
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
                mapui::notice(&ctx, text);
            }
            self.opening_card(&ctx);
            return;
        }
        if SHOW_REGION_NAMES {
            self.draw_labels(&ctx);
        }
        self.draw_missing(&ctx);
        self.draw_video_marks(&ctx);
        self.player_ui(&ctx);
        if SHOW_PHOTO_CAPTIONS {
            self.draw_photo_captions(&ctx);
        }
        self.draw_beside(&ctx);
        self.draw_lens(&ctx);

        // the top of the screen, by job: where you are (left), the search and its results (middle), how the map looks (right)
        let screen_w = ctx.content_rect().width();
        let width = (screen_w - 2.0 * 250.0).clamp(300.0, 620.0);
        let y = mapui::corners_y(screen_w, width);
        mapui::back(&ctx, y, self.map.as_ref().map(|m| m.data.manifest.name.as_str()), actions);
        mapui::look(&ctx, y, mapui::Look { view: &mut self.view_choice, groups_text: &mut self.groups_text, smooth: &mut self.smooth }, actions);

        // middle: the search box, then (only when there is something) the results, and the hidden searches (searchbar.rs)
        let engine = match &self.map {
            _ if self.engine.failed().is_some() => searchbar::Engine::Off(self.engine.failed().unwrap_or_default().to_string()),
            None => searchbar::Engine::NoMap,
            Some(m) if m.vectors.is_none() => searchbar::Engine::NotSearchable,
            Some(m) if self.engine.ready(m.data.manifest.model()).is_some() => searchbar::Engine::Ready,
            Some(_) => searchbar::Engine::Loading,
        };
        let m = self.map.as_ref();
        let bar = searchbar::Bar {
            search: &mut self.search,
            ideas: &mut self.ideas,
            hidden_open: &mut self.hidden_open,
            wanted: &mut self.wanted,
            filters: &mut self.filters,
            filter_edit: &mut self.filter_edit,
            spots: &mut self.spots,
            status: &self.status,
            photos: &self.photos,
            engine,
            found: m.map_or(0, |m| m.order.len()),
            at: m.and_then(|m| m.at),
            on_path: m.is_some_and(|m| m.trail.is_some()),
            most: self.can_match as f32,
            all_marked: self.names_only,
            hidden: m.map_or(Vec::new(), |m| m.hidden.iter().map(|h| (h.what.clone(), h.photos.len())).collect()),
            moments: m.is_some_and(|m| m.moments.is_some()),
            data: m.map(|m| &m.data),
            thumb: &mut |photo| {
                let data = &m?.data;
                let thumbs = &mut self.idea_thumbs;
                thumbs
                    .entry(photo)
                    .or_insert_with(|| {
                        let img = photo::fit(&photo::load_for(&data.path(photo), |_, _| 256)?, 512, false);
                        let color = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw());
                        Some(ctx.load_texture("idea photo", color, egui::TextureOptions::LINEAR))
                    })
                    .clone()
            },
        };
        searchbar::show(&ctx, width, bar, actions);

        self.opening_card(&ctx);
        mapui::help(&ctx, &mut self.help_open);
        self.debug_overlay(&ctx);

        if let (Some((photo, picture, copy, at)), Some(m)) = (self.menu, &self.map) {
            let file = files::original(&m.data, photo);
            let source = &m.data.photos[photo].source;
            let file_tip = if file.as_os_str() == source.as_str() || source.is_empty() { file.display().to_string() } else { format!("{source} is not there any more: the map's own copy, {}", file.display()) };
            let path = m.data.picture_path(picture);
            let menu = mapui::PhotoMenu {
                photo,
                picture,
                copy,
                at,
                title: m.data.photos[photo].title.clone(),
                ends_path: self.path_from.is_some_and(|from| from != photo),
                moments: m.has_moments(photo),
                can_add: !self.photos.is_empty() && !self.photos.iter().any(|(p, _)| *p == path),
                file,
                file_tip,
            };
            mapui::photo_menu(&ctx, &menu, actions);
        }
        if actions.path_from.is_some() || actions.path_to.is_some() || actions.like.is_some() || actions.add_like.is_some() || actions.moments.is_some() || actions.go_to_copy.is_some() || actions.details.is_some() || actions.done {
            self.menu = None;
        }
        if let (Some((photo, at)), Some(m)) = (actions.details, &self.map) {
            self.details = Some((photo, at, files::details(&m.data, photo)));
        }
        if let (Some((photo, at, details)), Some(m)) = (&self.details, &self.map) {
            let file = files::original(&m.data, *photo);
            if mapui::details(&ctx, *at, &m.data.photos[*photo].title, details, &file, actions) {
                self.details = None;
            }
        }
    }

    /// F3: how fast it draws and what it has loaded, in the corner, for working on the app.
    fn debug_overlay(&self, ctx: &egui::Context) {
        let Some(m) = self.map.as_ref().filter(|_| self.debug) else { return };
        let avg = self.frame_times.iter().sum::<f32>() / self.frame_times.len().max(1) as f32;
        egui::Area::new("fps".into()).anchor(egui::Align2::RIGHT_BOTTOM, [-14.0, -14.0]).show(ctx, |ui| {
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
    /// The right-click menu: a path from or to this photo, photos like this picture (a photo, or a video's frame).
    path_from: Option<usize>,
    path_to: Option<usize>,
    like: Option<usize>,
    add_like: Option<usize>,
    /// "Moments that match": this video's best frames for the search, beside its matches; or (the chip's cross) no more.
    moments: Option<usize>,
    close_moments: bool,
    /// Go to on the map (a copy beside the world's menu): fly to its photo on the map.
    go_to_copy: Option<usize>,
    /// A filter to write or change: its place (usize::MAX: a new one) and its text.
    edit_filter: Option<FilterEdit>,
    /// The filter being written or changed is done (Enter, or clicked away from).
    filter_done: bool,
    /// A filter chip's cross: its place in `filters`.
    drop_filter: Option<usize>,
    /// Only photos or only videos: in place of the one there was.
    set_kind: Option<bool>,
    /// A photo taken out of the search (its place in `photos`).
    drop_photo: Option<usize>,
    /// A photo searched for, by its place on the map: go to it.
    show_photo: Option<usize>,
    /// "Details": the photo and where the menu was.
    details: Option<(usize, egui::Pos2)>,
    /// "Show in folder" and "Open" on a photo's file.
    reveal: Option<PathBuf>,
    open_file: Option<PathBuf>,
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
        let gpu = Gpu::new(window.clone());
        self.viewer = Some(Viewer::new(gpu, |ctx| screen::Screen::window(window, ctx), library, first));
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

/// Which photo of the map is this file, if any (one pass over them all, for a click): a video's frame counts as its video.
fn find_photo(data: &data::MapData, path: &std::path::Path) -> Option<usize> {
    let rel = path.strip_prefix(&data.bundle).ok();
    let in_map = |file: &str| rel.is_some_and(|r| r == std::path::Path::new(file));
    let photo = data.photos.iter().position(|p| if p.file.is_empty() { std::path::Path::new(&p.source) == path } else { in_map(&p.file) });
    // a video's frame (searched with from a match beside the world) is found as its video
    photo.or_else(|| data.frames.iter().find(|f| in_map(&f.file)).map(|f| f.video))
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

/// The app without a window: `script` is run on the maps in `library`, frames drawn into a texture `size` points big at `ppp` pixels
/// a point, until its steps are done. For the tests (shots.rs) and PHOTO_MAP_HEADLESS.
fn run_headless(library: Library, size: (f32, f32), ppp: f32, script: script::Script) {
    let (w, h) = ((size.0 * ppp) as u32, (size.1 * ppp) as u32);
    let mut viewer = Viewer::new(Gpu::headless(w, h), |_| screen::Screen::headless(w, h, ppp), library, None);
    viewer.script = Some(script);
    while !viewer.quit {
        // at the window's pace: what moves (the camera, the photos sliding) moves by the clock
        let started = Instant::now();
        viewer.frame();
        std::thread::sleep(std::time::Duration::from_secs_f32(1.0 / 60.0).saturating_sub(started.elapsed()));
    }
}

fn main() {
    let maps = data_root().join("maps");
    let _ = std::fs::create_dir_all(&maps);
    let first = std::env::args().nth(1).map(PathBuf::from);
    // PHOTO_MAP_SCRIPT with PHOTO_MAP_HEADLESS: the script is run without a window
    if std::env::var_os("PHOTO_MAP_HEADLESS").is_some() {
        let Some(script) = script::Script::from_env() else {
            eprintln!("PHOTO_MAP_HEADLESS needs a script: PHOTO_MAP_SCRIPT=steps.txt");
            return;
        };
        run_headless(Library::new(maps), (1200.0, 780.0), 1.0, script);
        // SAFETY: `_exit` only ends the process (as below).
        unsafe { libc::_exit(0) };
    }
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
