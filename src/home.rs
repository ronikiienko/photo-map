//! The Maps page: the maps you can open, the ones being made, and the way to start a new one.

use crate::build::{self, Stage, State};
use crate::library::Library;
use crate::manager::{self, Advice, Manager};
use crate::modelstore::{self, ModelStore};
use crate::storage::{self, Sizes};
use crate::theme;
use egui::{Align, Align2, Color32, Layout, RichText, Sense, TextureHandle, Ui, vec2};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

pub enum Action {
    Open(PathBuf),
    Create { name: String, sources: Vec<PathBuf>, names_as_captions: bool, every: usize, model: String },
    Pause(PathBuf),
    SetBackend(crate::backends::Backend, bool),
    GetModel(String),
    StopModelFetch,
    RemoveModel(String),
    Continue(PathBuf),
    Discard(PathBuf),
    Delete(PathBuf),
    Resume(bool),
    /// Fetch what the runtime offer says (manager.rs).
    InstallRuntime,
    StopInstall,
    RemovePack(String),
    DismissOffer,
    /// Close the app and open it again (the runtime libraries it started with can't be changed while it runs).
    Restart,
}

/// A map being made, as the page shows it.
pub struct BuildView {
    pub dir: PathBuf,
    pub name: String,
    /// Where the photos came from, in words.
    pub source: String,
    pub stage: Stage,
    pub running: bool,
    /// What the models run on while it is made ("on the graphics card"), once a model is loaded.
    pub backend: String,
    /// Waiting for the map before it to finish.
    pub queued: bool,
    /// Being made by another window of the app.
    pub elsewhere: bool,
    pub state: Option<State>,
}

enum Cover {
    Loading(Receiver<Option<egui::ColorImage>>),
    Ready(TextureHandle),
    None,
}

/// A map's sizes, measured on a thread (a big map has many files).
enum Measured {
    Loading(Receiver<Sizes>),
    Ready(Sizes),
}

struct NewMap {
    /// The folders and photo files the map is made from.
    sources: Vec<PathBuf>,
    name: String,
    name_edited: bool,
    /// Which list of sources `photos` counts (the list as text).
    counted: String,
    count: Option<Receiver<usize>>,
    /// How many photos the sources hold, all of them.
    photos: Option<usize>,
    /// Use every Nth photo.
    every: usize,
    names_as_captions: bool,
    /// The id of the model that will make the photos' vectors (models.rs).
    model: String,
}

impl NewMap {
    /// More folders and photos for the list: each once, in order.
    fn add(&mut self, paths: Vec<PathBuf>) {
        for p in paths {
            if !self.sources.contains(&p) {
                self.sources.push(p);
            }
        }
        self.sources.sort();
    }
}

pub struct Home {
    covers: HashMap<PathBuf, Cover>,
    /// The little photos on a running build's card, by the id the build gave them.
    recent: HashMap<u64, TextureHandle>,
    sizes: HashMap<PathBuf, Measured>,
    /// The map whose breakdown is open.
    details: Option<PathBuf>,
    /// Something is being dragged over the window.
    drag_over: bool,
    new_map: Option<NewMap>,
    confirm_discard: Option<PathBuf>,
    confirm_delete: Option<PathBuf>,
    /// The window on the runtime libraries is open.
    runtime_open: bool,
    confirm_remove: Option<String>,
    /// The window on the models is open, and the model asked about removing.
    models_open: bool,
    confirm_remove_model: Option<String>,
}

const COVER: f32 = 200.0;

impl Home {
    pub fn new() -> Self {
        Self { covers: HashMap::new(), recent: HashMap::new(), sizes: HashMap::new(), details: None, drag_over: false, new_map: None, confirm_discard: None, confirm_delete: None, runtime_open: false, confirm_remove: None, models_open: false, confirm_remove_model: None }
    }

    pub fn open_new_map(&mut self) {
        self.new_map = Some(NewMap { sources: Vec::new(), name: String::new(), name_edited: false, counted: String::new(), count: None, photos: None, every: 1, names_as_captions: false, model: crate::models::DEFAULT.to_string() });
    }

    pub fn set_drag_over(&mut self, over: bool) {
        self.drag_over = over;
    }

    /// Folders and photo files dropped on the page: open the New map dialog with them (more drops add to the list). False if none
    /// of them was a folder or a photo.
    pub fn dropped(&mut self, paths: Vec<PathBuf>) -> bool {
        self.drag_over = false;
        let usable: Vec<PathBuf> = paths.into_iter().filter(|p| p.is_dir() || build::is_photo(p)).collect();
        if usable.is_empty() {
            return false;
        }
        if self.new_map.is_none() {
            self.open_new_map();
        }
        self.new_map.as_mut().unwrap().add(usable);
        true
    }

    pub fn forget_cover(&mut self, map: &Path) {
        self.covers.remove(map);
        self.sizes.remove(map);
    }

    pub fn show(&mut self, ctx: &egui::Context, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>, rt: &Manager, store: &ModelStore) -> Vec<Action> {
        let mut actions = Vec::new();
        egui::Area::new("maps-background".into()).anchor(Align2::LEFT_TOP, [0.0, 0.0]).order(egui::Order::Background).show(ctx, |ui| {
            let screen = ctx.content_rect();
            ui.painter().rect_filled(screen, 0.0, theme::BACKGROUND);
            ui.allocate_rect(screen, Sense::hover());
        });
        egui::Area::new("maps-page".into()).anchor(Align2::LEFT_TOP, [0.0, 0.0]).order(egui::Order::Middle).show(ctx, |ui| {
            let screen = ctx.content_rect();
            let width = (screen.width() - 64.0).clamp(320.0, 1100.0);
            ui.set_width(screen.width());
            ui.set_height(screen.height());
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(32.0);
                    ui.allocate_ui(vec2(width, 0.0), |ui| {
                        ui.set_width(width);
                        self.page(ui, lib, builds, resume, ready_msg, &mut actions);
                    });
                    ui.add_space(32.0);
                });
            });
        });
        if self.drag_over {
            egui::Area::new("drop-hint".into()).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).order(egui::Order::Tooltip).interactable(false).show(ctx, |ui| {
                theme::card().inner_margin(egui::Margin::symmetric(28, 20)).stroke(egui::Stroke::new(1.5, theme::ACCENT)).show(ui, |ui| {
                    ui.label(RichText::new("Drop a folder or photos to make a map").size(17.0));
                });
            });
        }
        self.dialogs(ctx, builds, lib, rt, store, &mut actions);
        actions
    }

    fn page(&mut self, ui: &mut Ui, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>, actions: &mut Vec<Action>) {
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            ui.label(RichText::new("Maps").size(26.0).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new("+  New map").strong()).min_size(vec2(112.0, 34.0))).clicked() {
                    self.open_new_map();
                }
                if ui.button("Models").on_hover_text("The models maps are made and searched with: get or remove them").clicked() {
                    self.models_open = true;
                }
                if ui.button("Runtime").on_hover_text("The libraries the models run on: what is installed and what could be").clicked() {
                    self.runtime_open = true;
                }
            });
        });
        ui.add_space(16.0);
        if let Some(msg) = ready_msg {
            theme::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(msg);
            });
            ui.add_space(16.0);
        }

        if !builds.is_empty() {
            ui.label(RichText::new("In progress").size(17.0));
            ui.add_space(4.0);
            for b in builds {
                self.build_card(ui, b, actions);
                ui.add_space(8.0);
            }
            ui.add_space(16.0);
        }

        if lib.maps.is_empty() {
            if builds.is_empty() {
                theme::card().inner_margin(egui::Margin::same(32)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.vertical_centered(|ui| {
                        ui.label(RichText::new("No maps yet").size(20.0));
                        ui.add_space(4.0);
                        ui.label(theme::muted("Pick a folder of photos and they are laid out so that similar ones sit together."));
                        ui.add_space(12.0);
                        if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new("Make your first map").strong()).min_size(vec2(180.0, 34.0))).clicked() {
                            self.open_new_map();
                        }
                    });
                });
            }
        } else {
            ui.label(RichText::new("Your maps").size(17.0));
            ui.add_space(8.0);
            let per_row = ((ui.available_width() + 12.0) / (COVER + 12.0)).floor().max(1.0) as usize;
            for row in lib.maps.chunks(per_row) {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = 12.0;
                    for entry in row {
                        self.map_card(ui, entry, actions);
                    }
                });
                ui.add_space(12.0);
            }
        }

        ui.add_space(8.0);
        let mut on = resume;
        if ui.checkbox(&mut on, "Continue unfinished maps when the app starts").changed() {
            actions.push(Action::Resume(on));
        }
    }

    fn build_card(&mut self, ui: &mut Ui, b: &BuildView, actions: &mut Vec<Action>) {
        theme::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            // the page centres its content; the card reads from the left
            ui.with_layout(Layout::top_down(Align::Min), |ui| {
                let failed = b.state.as_ref().is_some_and(|s| s.error.is_some());
                ui.horizontal(|ui| {
                    let colour = if failed { Color32::from_rgb(235, 110, 100) } else if b.running { theme::ACCENT } else { theme::MUTED };
                    theme::dot(ui, colour);
                    ui.label(RichText::new(&b.name).size(17.0));
                    ui.label(theme::muted(format!("from {}", b.source)));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.add_enabled(!b.elsewhere, egui::Button::new("Discard")).clicked() {
                            self.confirm_discard = Some(b.dir.clone());
                        }
                        if b.elsewhere {
                            // nothing to press: another window runs it
                        } else if b.running {
                            if ui.button("Pause").clicked() {
                                actions.push(Action::Pause(b.dir.clone()));
                            }
                        } else if b.queued {
                            if ui.button("Cancel").clicked() {
                                actions.push(Action::Pause(b.dir.clone()));
                            }
                        } else if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.5), true, egui::Button::new(if failed { "Try again" } else { "Continue" })).clicked() {
                            actions.push(Action::Continue(b.dir.clone()));
                        }
                    });
                });
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    let steps: Vec<Stage> = Stage::ALL.into_iter().filter(|&s| s != Stage::Finish).collect();
                    for (i, s) in steps.iter().enumerate() {
                        step(ui, *s, b.stage, i + 1 == steps.len());
                    }
                });
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);
                match &b.state {
                    Some(s) if s.error.is_some() => {
                        ui.label(RichText::new(s.error.clone().unwrap_or_default()).color(Color32::from_rgb(235, 110, 100)));
                        ui.label(theme::muted("Nothing is lost: it carries on from the last finished piece."));
                    }
                    Some(s) if b.running => {
                        let frac = if s.total > 0 { (s.done as f32 / s.total as f32).clamp(0.0, 1.0) } else { 0.0 };
                        if b.stage == Stage::Embed {
                            self.film(ui, s);
                        }
                        ui.add(egui::ProgressBar::new(frac).desired_height(8.0).fill(theme::ACCENT));
                        let mut line = format!("{}", b.stage.label());
                        if s.total > 0 {
                            // only these stages count photos; the others count steps of their own (shown as a percentage)
                            let by_photo = matches!(b.stage, Stage::Scan | Stage::Copy | Stage::Embed);
                            if by_photo {
                                line += &format!(" · {} of {}", thousands(s.done), thousands(s.total));
                            } else {
                                line += &format!(" · {:.0}%", frac * 100.0);
                            }
                            let rate = s.rate().unwrap_or(0.0);
                            if rate > 0.0 && by_photo {
                                line += &format!(" · {rate:.1} photos/s");
                            }
                            if rate > 0.0 && s.done < s.total {
                                line += &format!(" · about {} left", duration((s.total - s.done) as f32 / rate));
                            }
                        }
                        if !s.message.is_empty() {
                            line += &format!(" · {}", s.message);
                        }
                        if b.stage == Stage::Embed && !b.backend.is_empty() {
                            line += &format!(" · {}", b.backend);
                        }
                        ui.label(theme::muted(line));
                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(RichText::new(b.stage.explain()).size(13.0).color(theme::MUTED));
                    }
                    _ if b.elsewhere => {
                        ui.label(theme::muted("This map is being made in another window of the app. It shows here when it is finished."));
                    }
                    _ if b.queued => {
                        ui.label(theme::muted("Waiting for the map before it to finish"));
                    }
                    _ => {
                        ui.label(theme::muted(format!("Paused at: {}. Nothing is lost — continue any time, or discard it.", b.stage.label())));
                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(RichText::new(b.stage.explain()).size(13.0).color(theme::MUTED));
                    }
                }
            });
        });
    }

    /// The photos the model has just read, as a row of little squares; the newest comes in on the right and the oldest drops off the left.
    fn film(&mut self, ui: &mut Ui, s: &State) {
        let ctx = ui.ctx().clone();
        self.recent.retain(|id, _| s.recent.iter().any(|(r, _)| r == id));
        for (id, img) in &s.recent {
            self.recent.entry(*id).or_insert_with(|| {
                let size = [img.width() as usize, img.height() as usize];
                ctx.load_texture(format!("recent-{id}"), egui::ColorImage::from_rgba_unmultiplied(size, img.as_raw()), egui::TextureOptions::LINEAR)
            });
        }
        if s.recent.is_empty() {
            return;
        }
        let side = 52.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            // fixed slots, so the row doesn't jump while it fills up
            for slot in 0..build::RECENT {
                let i = (slot + s.recent.len()).checked_sub(build::RECENT);
                let (rect, _) = ui.allocate_exact_size(vec2(side, side), Sense::hover());
                match i.and_then(|i| s.recent.get(i)).and_then(|(id, _)| self.recent.get(id)) {
                    Some(tex) => {
                        // older ones fade
                        let age = (slot + 1) as f32 / build::RECENT as f32;
                        egui::Image::new(tex).corner_radius(6.0).tint(Color32::WHITE.gamma_multiply(0.35 + 0.65 * age)).paint_at(ui, rect);
                    }
                    None => {
                        ui.painter().rect_filled(rect, 6.0, Color32::from_rgb(28, 31, 39));
                    }
                }
            }
            ui.add_space(8.0);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 3.0;
                ui.label(theme::muted("Just read by the model. This is what it made of the last photo:"));
                if !s.vector_head.is_empty() {
                    // every number is printed with a sign and three decimals, in a font where all characters are as wide: the line never changes width
                    let numbers: Vec<String> = s.vector_head.iter().map(|x| format!("{x:+.3}")).collect();
                    ui.label(RichText::new(format!("[ {}  … ]", numbers.join("  "))).monospace().size(13.0).color(theme::ACCENT));
                    ui.label(theme::muted(format!("the first {} of its 768 numbers", build::HEAD)));
                }
            });
        });
        ui.add_space(6.0);
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
    }

    fn map_card(&mut self, ui: &mut Ui, e: &crate::library::Entry, actions: &mut Vec<Action>) {
        let ctx = ui.ctx().clone();
        let response = ui
            .allocate_ui(vec2(COVER, 0.0), |ui| {
                ui.set_width(COVER);
                let (rect, response) = ui.allocate_exact_size(vec2(COVER, COVER + 94.0), Sense::click());
                let hovered = response.hovered();
                let p = ui.painter();
                p.rect_filled(rect, theme::RADIUS, if hovered { Color32::from_rgb(30, 33, 42) } else { Color32::from_rgb(24, 26, 33) });
                p.rect_stroke(rect, theme::RADIUS, egui::Stroke::new(1.0, if hovered { theme::ACCENT.gamma_multiply(0.7) } else { theme::CARD_STROKE }), egui::StrokeKind::Inside);
                let image = egui::Rect::from_min_size(rect.min + vec2(6.0, 6.0), vec2(COVER - 12.0, COVER - 12.0));
                match self.cover(&ctx, &e.path) {
                    Some(tex) => {
                        egui::Image::new(&tex).corner_radius(8.0).paint_at(ui, image);
                    }
                    None => {
                        p.rect_filled(image, 8.0, Color32::from_rgb(32, 35, 44));
                    }
                }
                p.text(rect.min + vec2(12.0, COVER + 6.0), Align2::LEFT_TOP, &e.name, egui::FontId::proportional(15.0), theme::TEXT);
                let detail = if e.created.is_empty() { format!("{} photos", thousands(e.photos)) } else { format!("{} photos · {}", thousands(e.photos), e.created) };
                p.text(rect.min + vec2(12.0, COVER + 28.0), Align2::LEFT_TOP, detail, egui::FontId::proportional(12.0), theme::MUTED);
                // the model that made its vectors (it is the one that searches it)
                let model = crate::models::spec(&e.model).map_or("unknown model", |s| s.label);
                p.text(rect.min + vec2(12.0, COVER + 46.0), Align2::LEFT_TOP, model, egui::FontId::proportional(12.0), theme::MUTED);
                // total on disk, and a bar of what it is made of; "details" opens the explanation
                let sizes = self.sizes(&ctx, &e.path);
                let text = sizes.map_or("measuring…".to_string(), |s| format!("{} on disk", storage::human(s.total())));
                let line = rect.min + vec2(12.0, COVER + 68.0);
                let p = ui.painter();
                p.text(line, Align2::LEFT_TOP, text, egui::FontId::proportional(12.0), theme::TEXT);
                let bar = egui::Rect::from_min_size(rect.min + vec2(12.0, COVER + 86.0), vec2(COVER - 24.0, 4.0));
                p.rect_filled(bar, 2.0, Color32::from_rgb(38, 42, 52));
                if let Some(s) = sizes {
                    let total = s.total().max(1) as f32;
                    let mut x = bar.left();
                    for (i, part) in s.parts().iter().enumerate() {
                        let w = bar.width() * part.bytes as f32 / total;
                        p.rect_filled(egui::Rect::from_min_size(egui::pos2(x, bar.top()), vec2(w.max(1.0), 4.0)), 0.0, part_colour(i));
                        x += w;
                    }
                }
                let details = egui::Rect::from_min_size(rect.min + vec2(COVER - 62.0, COVER + 62.0), vec2(56.0, 20.0));
                let more = ui.interact(details, ui.id().with(("details", &e.path)), Sense::click());
                ui.painter().text(details.right_center() - vec2(4.0, 0.0), Align2::RIGHT_CENTER, "Details", egui::FontId::proportional(12.0), if more.hovered() { theme::ACCENT } else { theme::MUTED });
                if more.clicked() {
                    self.details = Some(e.path.clone());
                }
                (response, more)
            })
            .inner;
        let (response, more) = response;
        if response.clicked() && !more.hovered() {
            actions.push(Action::Open(e.path.clone()));
        }
        response.context_menu(|ui| {
            // egui's menus shrink the padding of their buttons to 2 px: put back the app's, so the text has room inside its highlight
            ui.spacing_mut().button_padding = vec2(10.0, 5.0);
            ui.set_min_width(170.0);
            if ui.button("Show in file manager").clicked() {
                show_in_file_manager(&e.path);
                ui.close();
            }
            if ui.button("Delete map…").clicked() {
                self.confirm_delete = Some(e.path.clone());
                ui.close();
            }
        });
    }

    fn sizes(&mut self, ctx: &egui::Context, map: &Path) -> Option<Sizes> {
        let entry = self.sizes.entry(map.to_path_buf()).or_insert_with(|| {
            let (tx, rx) = channel();
            let (path, ctx) = (map.to_path_buf(), ctx.clone());
            std::thread::spawn(move || {
                let _ = tx.send(storage::measure(&path));
                ctx.request_repaint();
            });
            Measured::Loading(rx)
        });
        if let Measured::Loading(rx) = entry {
            if let Ok(s) = rx.try_recv() {
                *entry = Measured::Ready(s);
            }
        }
        match entry {
            Measured::Ready(s) => Some(*s),
            Measured::Loading(_) => None,
        }
    }

    /// What takes the space in one map, part by part, with what each part is.
    fn details_window(&mut self, ctx: &egui::Context, lib: &Library) {
        let Some(path) = self.details.clone() else { return };
        let Some(entry) = lib.maps.iter().find(|m| m.path == path) else {
            self.details = None;
            return;
        };
        let Some(sizes) = self.sizes(ctx, &path) else { return };
        let mut closed = false;
        let response = theme::modal(ctx, "storage", 480.0, None, |ui| {
            closed = theme::window_header(ui, &format!("What takes space in “{}”", entry.name));
            ui.label(RichText::new(format!("{} on disk", storage::human(sizes.total()))).size(17.0));
            ui.label(theme::muted(format!("for {} photos", thousands(entry.photos))));
            ui.add_space(8.0);
            let total = sizes.total().max(1) as f32;
            for (i, part) in sizes.parts().iter().enumerate() {
                ui.horizontal(|ui| {
                    theme::dot(ui, part_colour(i));
                    ui.label(RichText::new(part.name).strong());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(theme::muted(format!("{} · {:.0}%", storage::human(part.bytes), 100.0 * part.bytes as f32 / total)));
                    });
                });
                ui.label(theme::muted(part.about));
                ui.add_space(6.0);
            }
            ui.add_space(4.0);
            if ui.button("Show in file manager").clicked() {
                show_in_file_manager(&path);
            }
            if sizes.linked > 0 {
                ui.separator();
                ui.label(RichText::new(format!("{} of photos are links, not copies", storage::human(sizes.linked))).strong());
                ui.label(theme::muted("The map was made from photos on the same disk, so instead of copying them it points at the originals. They take no extra space here, and they are not counted above. If you delete or move the originals, the map keeps its own link to the data, so it still works."));
            }
        });
        if closed || response.should_close() {
            self.details = None;
        }
    }

    /// A 2 x 2 mosaic of the map's first photos, made once and kept in the map as cover.png.
    fn cover(&mut self, ctx: &egui::Context, map: &Path) -> Option<TextureHandle> {
        let entry = self.covers.entry(map.to_path_buf()).or_insert_with(|| {
            let (tx, rx) = channel();
            let (path, ctx) = (map.to_path_buf(), ctx.clone());
            std::thread::spawn(move || {
                let _ = tx.send(cover_image(&path));
                ctx.request_repaint();
            });
            Cover::Loading(rx)
        });
        if let Cover::Loading(rx) = entry {
            match rx.try_recv() {
                Ok(Some(img)) => *entry = Cover::Ready(ctx.load_texture(format!("cover-{}", map.display()), img, egui::TextureOptions::LINEAR)),
                Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => *entry = Cover::None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        match entry {
            Cover::Ready(t) => Some(t.clone()),
            _ => None,
        }
    }

    fn dialogs(&mut self, ctx: &egui::Context, builds: &[BuildView], lib: &Library, rt: &Manager, store: &ModelStore, actions: &mut Vec<Action>) {
        if let Some(dir) = self.confirm_discard.clone() {
            let name = builds.iter().find(|b| b.dir == dir).map_or(String::new(), |b| b.name.clone());
            let mut close = false;
            let response = theme::modal(ctx, "discard", 380.0, Some("Discard this map?"), |ui| {
                ui.label(format!("“{name}” and the work done on it so far will be deleted. Your photos are not touched."));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if theme::tinted(ui, theme::DANGER, true, egui::Button::new("Discard")).clicked() {
                        actions.push(Action::Discard(dir.clone()));
                        close = true;
                    }
                    if ui.button("Keep it").clicked() {
                        close = true;
                    }
                });
            });
            if close || response.should_close() {
                self.confirm_discard = None;
            }
        }
        if let Some(path) = self.confirm_delete.clone() {
            let name = lib.maps.iter().find(|m| m.path == path).map_or(String::new(), |m| m.name.clone());
            let mut close = false;
            let response = theme::modal(ctx, "delete", 380.0, Some("Delete this map?"), |ui| {
                ui.label(format!("“{name}” will be deleted, with the copies of its photos inside it. The photos you made it from are not touched."));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if theme::tinted(ui, theme::DANGER, true, egui::Button::new("Delete")).clicked() {
                        actions.push(Action::Delete(path.clone()));
                        close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
            if close || response.should_close() {
                self.confirm_delete = None;
            }
        }
        self.details_window(ctx, lib);
        self.new_map_dialog(ctx, lib, rt, store, actions);
        self.runtime_window(ctx, rt, actions);
        self.models_window(ctx, rt, store, actions);
    }

    fn new_map_dialog(&mut self, ctx: &egui::Context, lib: &Library, rt: &Manager, store: &ModelStore, actions: &mut Vec<Action>) {
        let Some(nm) = &mut self.new_map else { return };
        if let Some(rx) = &nm.count {
            if let Ok(n) = rx.try_recv() {
                nm.photos = Some(n);
                nm.count = None;
            }
        }
        let mut close = false;
        let mut create = None;
        let mut open_models = false;
        // the model asked for is one that is installed
        if !crate::models::spec(&nm.model).is_some_and(|s| store.installed(s)) {
            if let Some(first) = crate::models::ALL.iter().find(|s| store.installed(s)) {
                nm.model = first.id.to_string();
            }
        }
        // a click on the backdrop does not close this one (what you typed would be lost): Cancel or Esc does
        theme::modal(ctx, "new-map", 480.0, Some("New map"), |ui| {
            // ONNX Runtime is not optional: nothing runs without it, so its offer is always shown (and can't be dismissed) until it is there
            let runtime_missing = rt.setup.ort.is_none();
            if runtime_missing || rt.nagging() || rt.installing() {
                runtime_offer(ui, rt, !runtime_missing, actions);
            }
            ui.label("Photos");
            if nm.sources.is_empty() {
                ui.label(theme::muted("Add the folders (their sub-folders are included) or photos to make the map from. You can also drop them on this window."));
            } else {
                let mut remove = None;
                egui::ScrollArea::vertical().max_height(120.0).auto_shrink([true, true]).show(ui, |ui| {
                    for (i, p) in nm.sources.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(theme::muted(if p.is_dir() { "folder" } else { "photo" }));
                            if theme::close_button(ui) {
                                remove = Some(i);
                            }
                            ui.add(egui::Label::new(p.display().to_string()).truncate());
                        });
                    }
                });
                if let Some(i) = remove {
                    nm.sources.remove(i);
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Add folder…").clicked() {
                    if let Some(folders) = rfd::FileDialog::new().set_title("Folders with photos").pick_folders() {
                        nm.add(folders);
                    }
                }
                if ui.button("Add photos…").clicked() {
                    if let Some(files) = rfd::FileDialog::new().set_title("Photos").add_filter("Photos", &build::TYPES).pick_files() {
                        nm.add(files);
                    }
                }
            });
            let sources = nm.sources.clone();
            let key = sources.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n");
            if nm.counted != key {
                nm.counted = key;
                nm.photos = None;
                nm.count = None;
                if !sources.is_empty() {
                    let (tx, rx) = channel();
                    let ctx = ui.ctx().clone();
                    let list = sources.clone();
                    std::thread::spawn(move || {
                        let _ = tx.send(build::collect_photos(&list, 1).len());
                        ctx.request_repaint();
                    });
                    nm.count = Some(rx);
                }
                if !nm.name_edited {
                    nm.name = match sources.as_slice() {
                        [] => String::new(),
                        [one] => one.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                        [first, ..] => first.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "New map".into()),
                    };
                }
            }
            ui.add_space(2.0);
            if sources.is_empty() {
                // nothing to count yet
            } else if nm.count.is_some() {
                ui.label(theme::muted("Counting photos…"));
            } else if let Some(n) = nm.photos {
                ui.label(theme::muted(if nm.every > 1 { format!("{} photos found, {} of them used", thousands(n), thousands(n.div_ceil(nm.every))) } else { format!("{} photos found", thousands(n)) }));
            }
            ui.add_space(8.0);
            ui.label("How many");
            theme::segmented(ui, &mut nm.every, &[(1, "All", "every photo"), (2, "1 in 2", "every second photo"), (5, "1 in 5", "every fifth photo"), (10, "1 in 10", "every tenth photo"), (20, "1 in 20", "every twentieth photo")]);
            ui.label(theme::muted("A smaller map takes less time to make, to see whether it is what you want. The photos are picked evenly from the sorted list. Make a bigger one later: photos already understood in your other maps are not done again."));
            ui.add_space(8.0);
            ui.label("Model");
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.spacing_mut().button_padding = egui::vec2(12.0, 5.0);
                for spec in crate::models::ALL.iter().filter(|s| store.installed(s)) {
                    ui.selectable_value(&mut nm.model, spec.id.to_string(), spec.label).on_hover_text(spec.note);
                }
            });
            if !crate::models::ALL.iter().any(|s| store.installed(s)) {
                ui.horizontal(|ui| {
                    ui.label(theme::muted("No model is installed."));
                    if ui.button("Get a model").clicked() {
                        open_models = true;
                    }
                });
            } else if let Some(spec) = crate::models::spec(&nm.model) {
                ui.label(theme::muted(spec.note));
            }
            ui.label(theme::muted("The map is searched with the same model, and vectors from different models are never mixed: photos already understood in your other maps are reused only if those were made with this model too."));
            ui.add_space(8.0);
            ui.label("Name");
            if ui.add(egui::TextEdit::singleline(&mut nm.name).desired_width(340.0)).changed() {
                nm.name_edited = true;
            }
            let clean = build::clean_name(&nm.name);
            let taken = lib.maps.iter().any(|m| m.path == build::finished_dir(lib.dir(), &clean)) || lib.builds.iter().any(|b| b.dir == build::building_dir(lib.dir(), &clean));
            let problem = if clean.is_empty() {
                Some("Give the map a name.")
            } else if taken {
                Some("There is already a map with this name.")
            } else {
                None
            };
            if let Some(p) = problem {
                ui.label(theme::muted(p));
            }
            ui.add_space(8.0);
            ui.checkbox(&mut nm.names_as_captions, "Use file names as captions").on_hover_text("Each photo's file name (without the extension) is shown above it when you zoom in, and search looks at it as well as at the picture. Only worth it if the names describe the photos.");
            ui.add_space(8.0);
            ui.label(theme::muted("It can take a while for many photos. You can pause, close the app and carry on later."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let ok = problem.is_none() && !sources.is_empty() && nm.photos != Some(0) && rt.setup.ort.is_some() && crate::models::spec(&nm.model).is_some_and(|s| store.installed(s));
                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), ok, egui::Button::new(RichText::new("Create").strong())).clicked() {
                    create = Some(Action::Create { name: clean.clone(), sources: sources.clone(), names_as_captions: nm.names_as_captions, every: nm.every, model: nm.model.clone() });
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
        if open_models {
            self.models_open = true;
        }
        if let Some(a) = create {
            actions.push(a);
            close = true;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            close = true;
        }
        if close {
            self.new_map = None;
        }
    }

    /// The window on the models: which are installed, what each takes, and a way to get or remove each.
    fn models_window(&mut self, ctx: &egui::Context, rt: &Manager, store: &ModelStore, actions: &mut Vec<Action>) {
        if !self.models_open {
            return;
        }
        let mut close = false;
        let response = theme::modal(ctx, "models", 560.0, Some("Models"), |ui| {
            ui.label(theme::muted("A map is made and searched with one model. Only the models you have are offered when a map is made. Getting a model is simulated for now: nothing is downloaded."));
            ui.add_space(8.0);
            if rt.setup.ort.is_none() {
                ui.label(RichText::new("ONNX Runtime is not installed: no model can run without it.").color(Color32::from_rgb(235, 110, 100)));
                ui.add_space(4.0);
                runtime_offer(ui, rt, false, actions);
            }
            for spec in crate::models::ALL.iter() {
                theme::card().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(spec.label).strong());
                        ui.label(theme::muted(format!("{} MB", modelstore::size_mb(spec))));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let fetching = store.job.as_ref().filter(|j| j.id == spec.id);
                            if let Some(job) = fetching {
                                if ui.button("Stop").clicked() {
                                    actions.push(Action::StopModelFetch);
                                }
                                ui.add(egui::ProgressBar::new(job.state().0).desired_width(120.0).desired_height(8.0).fill(theme::ACCENT));
                            } else if store.installed(spec) {
                                if ui.button("Remove").clicked() {
                                    self.confirm_remove_model = Some(spec.id.to_string());
                                }
                                ui.label(theme::muted("installed"));
                            } else if ui.add_enabled(!store.fetching(), egui::Button::new("Get")).clicked() {
                                actions.push(Action::GetModel(spec.id.to_string()));
                            }
                        });
                    });
                    ui.label(theme::muted(spec.note));
                });
                ui.add_space(4.0);
            }
            if let Some(id) = self.confirm_remove_model.clone() {
                let label = crate::models::spec(&id).map_or("", |s| s.label);
                ui.add_space(4.0);
                ui.label(format!("Remove {label}? Its files and everything prepared for it (the networks made for the Apple chip or the graphics card, the vocabulary's vectors) are deleted. Maps made with it stay, but can't be searched until you get it again."));
                ui.horizontal(|ui| {
                    if theme::tinted(ui, theme::DANGER, true, egui::Button::new("Remove")).clicked() {
                        actions.push(Action::RemoveModel(id.clone()));
                        self.confirm_remove_model = None;
                    }
                    if ui.button("Keep it").clicked() {
                        self.confirm_remove_model = None;
                    }
                });
            }
            if let Some(e) = &store.failure {
                ui.add_space(8.0);
                ui.label(RichText::new(e).color(Color32::from_rgb(235, 110, 100)));
            }
            ui.add_space(8.0);
            if ui.button("Close").clicked() {
                close = true;
            }
        });
        if close || response.should_close() {
            self.models_open = false;
            self.confirm_remove_model = None;
        }
    }

    /// The window on the runtime libraries: what the models run on now, what is installed and by whom, what could be installed.
    fn runtime_window(&mut self, ctx: &egui::Context, rt: &Manager, actions: &mut Vec<Action>) {
        if !self.runtime_open {
            return;
        }
        let mut close = false;
        let response = theme::modal(ctx, "runtime", 560.0, Some("Runtime"), |ui| {
            ui.label(theme::muted(if cfg!(target_os = "macos") {
                "The library the models run on: ONNX Runtime. It runs them on the processor, and on the Mac's own chip through CoreML."
            } else {
                "The libraries the models run on. The models run on the processor with ONNX Runtime alone; with an NVIDIA graphics card they can use it through CUDA, and TensorRT on top of that."
            }));
            ui.add_space(8.0);
            let row = |ui: &mut Ui, name: &str, what: String| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(name).strong());
                    ui.label(theme::muted(what));
                });
            };
            let mac = !crate::backends::Backend::Cuda.exists_here();
            if !mac {
                row(ui, "Graphics card", match &rt.nvidia {
                    Some(c) => format!("{} (its driver supports CUDA {})", c.name, manager::cuda_version(c.driver_cuda)),
                    None if rt.card_known() => "no NVIDIA card found".into(),
                    None => "looking…".into(),
                });
            }
            row(ui, "ONNX Runtime", rt.setup.ort.as_ref().map_or("not installed".into(), |p| format!("{}{}", manager::describe(p), if mac || rt.setup.ort_cuda { "" } else { "; processor only" })));
            if !mac {
                row(ui, "CUDA and cuDNN", rt.setup.gpu.as_ref().map_or("not installed".into(), |g| manager::describe(&g.cuda)));
                row(ui, "TensorRT", rt.setup.tensorrt.as_ref().map_or("not installed".into(), manager::describe));
            }
            for note in &rt.wrong {
                ui.label(theme::muted(format!("{note}, so it is not used.")));
            }
            ui.add_space(8.0);
            runtime_offer(ui, rt, false, actions);

            ui.add_space(8.0);
            ui.label("Run the models on");
            for b in crate::backends::Backend::ALL.into_iter().filter(|b| b.exists_here()) {
                let available = b.available(&rt.setup) && !(matches!(b, crate::backends::Backend::Cuda | crate::backends::Backend::TensorRt) && rt.nvidia.is_none());
                let mut on = available && rt.backends.is_on(b, &rt.setup);
                let was = on;
                ui.horizontal(|ui| {
                    let response = ui.add_enabled(available, egui::Checkbox::new(&mut on, b.label())).on_hover_text(b.note());
                    if response.changed() && on != was {
                        actions.push(Action::SetBackend(b, on));
                    }
                    if !available {
                        ui.label(theme::muted(if rt.setup.ort.is_none() { "needs ONNX Runtime" } else { "not installed" }));
                    }
                });
            }
            ui.label(theme::muted("The processor is used for whatever the others can't run. Changes apply after a restart."));

            let installed = rt.installed();
            if !installed.is_empty() {
                ui.add_space(8.0);
                ui.label("Installed by photo-map");
                for pack in &installed {
                    ui.horizontal(|ui| {
                        ui.label(format!("{} {}", crate::packs::pack(&pack.id).map_or(pack.id.as_str(), |p| p.label), pack.version));
                        ui.label(theme::muted(format!("{:.0} MB", pack.bytes as f64 / 1048576.0)));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui.add_enabled(!rt.installing(), egui::Button::new("Remove")).clicked() {
                                self.confirm_remove = Some(pack.id.clone());
                            }
                        });
                    });
                }
                ui.label(theme::muted("Libraries you installed yourself, or the system's, are used when they fit but never removed from here."));
            }
            if let Some(id) = self.confirm_remove.clone() {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(format!("Remove {id}? The models run slower or not at all until it is installed again."));
                    if theme::tinted(ui, theme::DANGER, true, egui::Button::new("Remove")).clicked() {
                        actions.push(Action::RemovePack(id.clone()));
                        self.confirm_remove = None;
                    }
                    if ui.button("Keep it").clicked() {
                        self.confirm_remove = None;
                    }
                });
            }
            if let Some(e) = &rt.failure {
                ui.add_space(8.0);
                ui.label(RichText::new(e).color(Color32::from_rgb(235, 110, 100)));
            }
            ui.add_space(8.0);
            if ui.button("Close").clicked() {
                close = true;
            }
        });
        if close || response.should_close() {
            self.runtime_open = false;
            self.confirm_remove = None;
        }
    }
}

/// What the runtime could use that is not installed, as a box with the way to get it - or the install under way, or the note that
/// a restart is needed. `dismissible`: with a way to say "not now" that is remembered.
fn runtime_offer(ui: &mut Ui, rt: &Manager, dismissible: bool, actions: &mut Vec<Action>) {
    if let Some(job) = &rt.job {
        let s = job.state();
        theme::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(format!("Downloading {}…", s.current));
            ui.add(egui::ProgressBar::new(s.fraction).desired_height(8.0).fill(theme::ACCENT));
            ui.horizontal(|ui| {
                ui.label(theme::muted(format!("{} of {}", job.packs.iter().position(|p| p.label == s.current).map_or(1, |i| i + 1), job.packs.len())));
                if ui.button("Stop").clicked() {
                    actions.push(Action::StopInstall);
                }
            });
        });
        ui.add_space(8.0);
        return;
    }
    if rt.restart {
        theme::card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label("The runtime changed. The app has to start again for the models to use it.");
            ui.label(theme::muted("Maps being made are paused and carry on when it is open again."));
            if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new("Restart now").strong())).clicked() {
                actions.push(Action::Restart);
            }
        });
        ui.add_space(8.0);
        return;
    }
    match rt.advice() {
        Advice::Nothing => {}
        Advice::Driver(why) => {
            theme::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(why);
            });
            ui.add_space(8.0);
        }
        Advice::Offer(offer) => {
            theme::card().show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(&offer.why);
                ui.label(theme::muted(format!("Download: about {}. It is kept in the runtime folder of the app and can be removed from the Runtime window.", size_text(offer.mb()))));
                ui.horizontal(|ui| {
                    if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new(offer.title).strong())).clicked() {
                        actions.push(Action::InstallRuntime);
                    }
                    if dismissible && ui.button("Not now").on_hover_text("Don't offer this here again; the Runtime window on the Maps page still has it").clicked() {
                        actions.push(Action::DismissOffer);
                    }
                });
            });
            ui.add_space(8.0);
        }
    }
}

fn size_text(mb: u32) -> String {
    if mb >= 1000 { format!("{:.1} GB", mb as f32 / 1000.0) } else { format!("{mb} MB") }
}

/// One step of the list of steps: a dot (filled when done, ringed when current, empty when still to come) and its name.
/// Plain text on the card, joined by a thin line, so it reads as a path and not as buttons.
fn step(ui: &mut Ui, step: Stage, current: Stage, last: bool) {
    let short = match step {
        Stage::Scan => "Find",
        Stage::Copy => "Copy",
        Stage::Load => "Load",
        Stage::Embed => "Understand",
        Stage::Layout => "Lay out",
        Stage::Regions => "Regions",
        Stage::Names => "Names",
        Stage::Bake => "Prepare",
        Stage::Finish => "Finish",
    };
    let colour = if step < current { theme::OK } else if step == current { theme::ACCENT } else { theme::MUTED.gamma_multiply(0.6) };
    let (dot, _) = ui.allocate_exact_size(vec2(12.0, 12.0), Sense::hover());
    let p = ui.painter();
    if step < current {
        p.circle_filled(dot.center(), 5.0, colour);
    } else {
        p.circle_stroke(dot.center(), 4.5, egui::Stroke::new(1.5, colour));
        if step == current {
            p.circle_filled(dot.center(), 2.5, colour);
        }
    }
    let text = RichText::new(short).size(12.0).color(if step == current { theme::TEXT } else if step < current { theme::MUTED } else { theme::MUTED.gamma_multiply(0.7) });
    ui.label(if step == current { text.strong() } else { text });
    if !last {
        let (line, _) = ui.allocate_exact_size(vec2(18.0, 12.0), Sense::hover());
        ui.painter().hline(line.x_range(), line.center().y, egui::Stroke::new(1.0, if step < current { theme::OK.gamma_multiply(0.6) } else { theme::CARD_STROKE }));
    }
}

/// Open a folder in the system's file manager.
fn show_in_file_manager(folder: &Path) {
    let program = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    if let Ok(mut child) = std::process::Command::new(program).arg(folder).spawn() {
        std::thread::spawn(move || {
            let _ = child.wait(); // so it doesn't linger as a zombie
        });
    }
}

fn part_colour(i: usize) -> Color32 {
    [theme::ACCENT, theme::OK, Color32::from_rgb(230, 170, 90), Color32::from_rgb(200, 130, 220), Color32::from_rgb(110, 200, 210), theme::MUTED][i % 6]
}

/// 28 014 style grouping.
fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push('\u{202f}');
        }
        out.push(c);
    }
    out
}

fn duration(secs: f32) -> String {
    let s = secs.round() as u64;
    if s < 90 {
        format!("{s} s")
    } else if s < 5400 {
        format!("{} min", (s + 30) / 60)
    } else {
        format!("{:.1} h", s as f32 / 3600.0)
    }
}

/// The cover of a map: read cover.png if it is there, else make it from the first four photos and keep it.
fn cover_image(map: &Path) -> Option<egui::ColorImage> {
    let file = map.join("cover.png");
    let img = match image::open(&file) {
        Ok(i) => i.to_rgba8(),
        Err(_) => {
            const SIZE: u32 = 384;
            let mut paths: Vec<PathBuf> = std::fs::read_dir(map.join("originals")).ok()?.flatten().map(|e| e.path()).take(64).collect();
            paths.sort();
            let mut mosaic = image::RgbaImage::from_pixel(SIZE, SIZE, image::Rgba([32, 35, 44, 255]));
            let mut put = 0;
            for p in paths {
                if put == 4 {
                    break;
                }
                let Some(img) = crate::photo::load(&p) else { continue };
                let half = SIZE / 2;
                let small = image::imageops::resize(&image::imageops::crop_imm(&img, (img.width().saturating_sub(img.height().min(img.width()))) / 2, (img.height().saturating_sub(img.width().min(img.height()))) / 2, img.width().min(img.height()), img.width().min(img.height())).to_image(), half - 2, half - 2, image::imageops::FilterType::Triangle);
                image::imageops::replace(&mut mosaic, &small, ((put % 2) * half + 1) as i64, ((put / 2) * half + 1) as i64);
                put += 1;
            }
            if put == 0 {
                return None;
            }
            let _ = mosaic.save(&file);
            mosaic
        }
    };
    Some(egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], img.as_raw()))
}
