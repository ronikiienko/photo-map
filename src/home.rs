//! The Maps page: the maps you can open, the ones being made, and the way to start a new one.

use crate::build::{self, Stage, State};
use crate::examples::{self, Examples};
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
    Create { name: String, sources: Vec<PathBuf>, every: usize, model: String },
    Pause(PathBuf),
    SetBackend(crate::backends::Backend),
    GetModel(String),
    StopModelFetch,
    RemoveModel(String),
    CountModels,
    GetExample(String),
    StopExample,
    RemoveExample(String),
    Continue(PathBuf),
    Discard(PathBuf),
    Delete(PathBuf),
    Resume(bool),
    /// Fetch what the runtime offer says (manager.rs).
    InstallRuntime,
    /// Download ffmpeg, which reads videos.
    InstallFfmpeg,
    StopInstall,
    RemoveRuntime,
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
    /// The videos that couldn't be read, left out (their file names).
    pub unread_videos: Vec<String>,
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
    /// How many photos the sources hold, and whether any source is on another disk than the maps.
    count: Option<Receiver<(usize, usize, bool)>>,
    /// How many photos and videos the sources hold, all of them.
    photos: Option<usize>,
    /// How many of those are videos.
    videos: usize,
    /// Some of the sources are on another disk (the note under the count says so).
    other_disk: bool,
    /// Use every Nth photo.
    every: usize,
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
    confirm_remove: bool,
    /// The window on the models is open, and the model asked about removing.
    models_open: bool,
    examples_open: bool,
    /// The Settings window is open.
    settings_open: bool,
    /// An example whose photos Remove was pressed for, waiting to be confirmed.
    confirm_remove_example: Option<&'static str>,
    confirm_remove_model: Option<String>,
}

const COVER: f32 = 200.0;

/// How many map cards fit side by side in this width (12 between them; a little slack, as widths come back a hair under).
fn per_row(width: f32) -> usize {
    ((width + 12.5) / (COVER + 12.0)).floor().max(1.0) as usize
}

impl Home {
    pub fn new() -> Self {
        Self { covers: HashMap::new(), recent: HashMap::new(), sizes: HashMap::new(), details: None, drag_over: false, new_map: None, confirm_discard: None, confirm_delete: None, runtime_open: false, confirm_remove: false, models_open: false, confirm_remove_model: None, examples_open: false, settings_open: false, confirm_remove_example: None }
    }

    pub fn open_new_map(&mut self) {
        self.new_map = Some(NewMap { sources: Vec::new(), name: String::new(), name_edited: false, counted: String::new(), count: None, photos: None, videos: 0, other_disk: false, every: 1, model: crate::models::DEFAULT.to_string() });
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

    /// `settings` is changed in place by the Settings window (the caller saves and applies it).
    #[allow(clippy::too_many_arguments)]
    pub fn show(&mut self, ctx: &egui::Context, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>, rt: &Manager, store: &ModelStore, examples: &Examples, settings: &mut crate::settings::Settings) -> Vec<Action> {
        let mut actions = Vec::new();
        egui::Area::new("maps-background".into()).anchor(Align2::LEFT_TOP, [0.0, 0.0]).order(egui::Order::Background).show(ctx, |ui| {
            let screen = ctx.content_rect();
            ui.painter().rect_filled(screen, 0.0, theme::BACKGROUND);
            ui.allocate_rect(screen, Sense::hover());
        });
        egui::Area::new("maps-page".into()).anchor(Align2::LEFT_TOP, [0.0, 0.0]).order(egui::Order::Middle).show(ctx, |ui| {
            let screen = ctx.content_rect();
            let width = (screen.width() - 64.0).clamp(320.0, 1100.0);
            // as wide as a whole row of cards, so that the title, the cards and what is under them line up
            // (no wider than the maps there are: four maps in a page for five would leave the title and the cards out of line)
            let per_row = per_row(width).min(lib.maps.len().max(2)) as f32;
            let width = if lib.maps.is_empty() { width } else { per_row * COVER + (per_row - 1.0) * 12.0 };
            ui.set_width(screen.width());
            ui.set_height(screen.height());
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(32.0);
                    ui.allocate_ui_with_layout(vec2(width, 0.0), Layout::top_down(Align::Min), |ui| {
                        ui.set_width(width);
                        self.page(ui, lib, builds, resume, ready_msg, &mut actions);
                    });
                    ui.add_space(32.0);
                });
            });
        });
        if self.drag_over {
            // the whole window takes the drop: dimmed, with an accent edge and the words in the middle
            egui::Area::new("drop-hint".into()).anchor(Align2::LEFT_TOP, [0.0, 0.0]).order(egui::Order::Tooltip).interactable(false).show(ctx, |ui| {
                let screen = ctx.content_rect();
                ui.painter().rect_filled(screen, 0.0, Color32::from_black_alpha(205));
                ui.painter().rect_stroke(screen.shrink(16.0), theme::RADIUS, egui::Stroke::new(2.0, theme::ACCENT), egui::StrokeKind::Inside);
                ui.painter().text(screen.center() - vec2(0.0, 12.0), Align2::CENTER_CENTER, "Drop to make a map", egui::FontId::proportional(22.0), theme::TEXT);
                ui.painter().text(screen.center() + vec2(0.0, 18.0), Align2::CENTER_CENTER, "folders or photos", egui::FontId::proportional(14.0), theme::MUTED);
                ui.allocate_rect(screen, Sense::hover());
            });
        }
        self.dialogs(ctx, builds, lib, rt, store, examples, &mut actions);
        self.settings_window(ctx, settings);
        actions
    }

    fn page(&mut self, ui: &mut Ui, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>, actions: &mut Vec<Action>) {
        // the buttons at the right of the title, or under it when the page is too narrow for both
        let narrow = ui.available_width() < 600.0;
        if narrow {
            ui.label(RichText::new("Maps").size(26.0).strong());
            ui.add_space(4.0);
        }
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            if !narrow {
                ui.label(RichText::new("Maps").size(26.0).strong());
            }
            ui.with_layout(if narrow { Layout::left_to_right(Align::Center) } else { Layout::right_to_left(Align::Center) }, |ui| {
                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new("+  New map").strong()).min_size(vec2(112.0, 34.0))).clicked() {
                    self.open_new_map();
                }
                if ui.button("Example photos").on_hover_text("Photos to try a map with, if you have none at hand").clicked() {
                    self.examples_open = true;
                }
                if ui.button("Models").on_hover_text("The models maps are made and searched with: get or remove them").clicked() {
                    self.models_open = true;
                    actions.push(Action::CountModels);
                }
                if ui.button("Runtime").on_hover_text("The libraries the models run on: what is there and what can be downloaded").clicked() {
                    self.runtime_open = true;
                }
                if ui.button("Settings").on_hover_text("Zoom speed and interface size").clicked() {
                    self.settings_open = true;
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
                        ui.add_space(4.0);
                        if ui.button("No photos at hand? Get example photos").clicked() {
                            self.examples_open = true;
                        }
                    });
                });
            }
        } else {
            ui.label(RichText::new("Your maps").size(17.0));
            ui.add_space(8.0);
            let per_row = per_row(ui.available_width());
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
        // the box drawn with an edge: the theme's unticked box is almost the page's colour
        ui.visuals_mut().widgets.inactive.bg_stroke = egui::Stroke::new(1.0, theme::MUTED);
        ui.visuals_mut().widgets.hovered.bg_stroke = egui::Stroke::new(1.0, theme::TEXT);
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
                if !b.unread_videos.is_empty() {
                    // which ones, under the mouse (a few: a long list wouldn't fit)
                    let n = b.unread_videos.len();
                    let mut names = b.unread_videos.iter().take(12).cloned().collect::<Vec<_>>().join("\n");
                    if n > 12 {
                        names += &format!("\nand {} more", n - 12);
                    }
                    let text = if n == 1 { "1 video couldn't be read: it is left out".to_string() } else { format!("{n} videos couldn't be read: they are left out") };
                    ui.add_space(2.0);
                    ui.label(RichText::new(text).size(13.0).color(theme::WARNING)).on_hover_text(names);
                }
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
                            let by_photo = matches!(b.stage, Stage::Scan | Stage::Embed);
                            // the videos step counts seconds of video
                            let by_time = s.unit == "seconds of video";
                            if by_time {
                                line += &format!(" · {} of {} of video", video_time(s.done), video_time(s.total));
                            } else if by_photo {
                                line += &format!(" · {} of {}", thousands(s.done), thousands(s.total));
                            } else {
                                line += &format!(" · {:.0}%", frac * 100.0);
                            }
                            let rate = s.rate().unwrap_or(0.0);
                            if rate > 0.0 && by_time {
                                line += &if rate < 10.0 { format!(" · {rate:.1}x real time") } else { format!(" · {rate:.0}x real time") };
                            } else if rate > 0.0 && by_photo {
                                line += &format!(" · {rate:.1} {}/s", s.unit);
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
                // the day it was made, if it fits (the time of day would run past the card's edge on a big map)
                let day = e.created.get(..10).unwrap_or(&e.created);
                let count = photos_and_videos(e.photos, e.videos);
                let font = egui::FontId::proportional(12.0);
                let with_day = format!("{count} · {day}");
                let fits = p.layout_no_wrap(with_day.clone(), font.clone(), theme::MUTED).size().x <= rect.width() - 24.0;
                let detail = if day.is_empty() || !fits { count } else { with_day };
                p.text(rect.min + vec2(12.0, COVER + 28.0), Align2::LEFT_TOP, detail, font, theme::MUTED);
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
                let details = egui::Rect::from_min_size(rect.min + vec2(COVER - 62.0, COVER + 65.5), vec2(56.0, 20.0)); // on the size's line
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
            ui.label(theme::muted(format!("for {}", photos_and_videos(entry.photos, entry.videos))));
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
            ui.separator();
            ui.label(RichText::new("Your photos are not copied").strong());
            ui.label(theme::muted("The map keeps only where each photo is (its path), and takes no space for the photos themselves. If one is moved, renamed or deleted, or its disk is not plugged in, the map still shows its small picture, but zoomed in it says \"not found\"."));
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

    fn dialogs(&mut self, ctx: &egui::Context, builds: &[BuildView], lib: &Library, rt: &Manager, store: &ModelStore, examples: &Examples, actions: &mut Vec<Action>) {
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
                ui.label(format!("“{name}” will be deleted, with everything inside it. The photos you made it from are not touched."));
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
        self.examples_window(ctx, examples, actions);
    }

    fn new_map_dialog(&mut self, ctx: &egui::Context, lib: &Library, rt: &Manager, store: &ModelStore, actions: &mut Vec<Action>) {
        let Some(nm) = &mut self.new_map else { return };
        if let Some(rx) = &nm.count {
            if let Ok((n, videos, other_disk)) = rx.try_recv() {
                nm.photos = Some(n);
                nm.videos = videos;
                nm.other_disk = other_disk;
                nm.count = None;
            }
        }
        let mut close = false;
        let mut create = None;
        let mut open_models = false;
        let mut open_runtime = false;
        // the model asked for is one that is installed
        if !crate::models::spec(&nm.model).is_some_and(|s| store.installed(s)) {
            if let Some(first) = crate::models::ALL.iter().find(|s| store.installed(s)) {
                nm.model = first.id.to_string();
            }
        }
        // a click on the backdrop does not close this one (what you typed would be lost): Cancel or Esc does
        theme::modal(ctx, "new-map", 480.0, Some("New map"), |ui| {
            // what the runtime lacks, with the way to the Runtime window that gets it (the form doesn't install anything itself)
            let lacking = if rt.setup.ort.is_none() {
                Some((RichText::new("ONNX Runtime has not been downloaded: no model can run without it.").color(Color32::from_rgb(235, 110, 100)), "Get it in the Runtime window"))
            } else if rt.installing() {
                Some((theme::muted("The runtime is being downloaded."), "See it in the Runtime window"))
            } else if let Advice::Offer(offer) = rt.advice() {
                Some((theme::muted(offer.why), "Get it in the Runtime window"))
            } else {
                None
            };
            if let Some((text, tip)) = lacking {
                ui.horizontal(|ui| {
                    ui.label(text);
                    if ui.button("Runtime…").on_hover_text(tip).clicked() {
                        open_runtime = true;
                    }
                });
                ui.add_space(8.0);
            }
            ui.label("Photos");
            if nm.sources.is_empty() {
                ui.label(theme::muted("Add the folders (their sub-folders are included), photos or videos to make the map from. Each video is one photo on the map. You can also drop them on this window."));
            } else {
                let mut remove = None;
                egui::ScrollArea::vertical().max_height(120.0).auto_shrink([true, true]).show(ui, |ui| {
                    for (i, p) in nm.sources.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(theme::muted(if p.is_dir() { "folder" } else if build::is_video(p) { "video" } else { "photo" }));
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
                if ui.button("Add photos or videos…").clicked() {
                    if let Some(files) = rfd::FileDialog::new().set_title("Photos or videos").add_filter("Photos and videos", &[&build::TYPES[..], &build::VIDEO_TYPES[..]].concat()).add_filter("Photos", &build::TYPES).add_filter("Videos", &build::VIDEO_TYPES).pick_files() {
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
                    let (list, maps) = (sources.clone(), lib.dir().to_path_buf());
                    std::thread::spawn(move || {
                        let found = build::collect_photos(&list, 1);
                        let _ = tx.send((found.len(), found.iter().filter(|p| build::is_video(p)).count(), list.iter().any(|s| !build::same_disk(s, &maps))));
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
                ui.label(theme::muted("Counting photos and videos…"));
            } else if let Some(n) = nm.photos {
                let found = photos_and_videos(n, nm.videos);
                ui.label(theme::muted(if nm.every > 1 { format!("{found} found, {} of them used", thousands(n.div_ceil(nm.every))) } else { format!("{found} found") }));
                if nm.videos > 0 && rt.ffmpeg.is_none() {
                    // like the runtime above: the need, and the way to the window that gets it
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Videos are read with ffmpeg, which hasn't been downloaded.").color(theme::WARNING));
                        if ui.button("Runtime…").on_hover_text("Get it in the Runtime window").clicked() {
                            open_runtime = true;
                        }
                    });
                }
                if n > 0 {
                    ui.add_space(6.0);
                    ui.label(theme::muted(if nm.other_disk {
                        "The photos are on another disk: the map will read them there and take no space for them. Zoomed out it works without that disk; to see a photo at full size, the disk must be plugged in and the photo where it was."
                    } else {
                        "The map will read the photos where they are and take no space for them. Zoomed out it always works; at full size, a photo that has been moved or deleted shows as not found."
                    }));
                }
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
                    ui.label(theme::muted("No model has been downloaded."));
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
            ui.label(theme::muted("It can take a while for many photos. You can pause, close the app and carry on later."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let ok = problem.is_none() && !sources.is_empty() && nm.photos != Some(0) && rt.setup.ort.is_some() && crate::models::spec(&nm.model).is_some_and(|s| store.installed(s));
                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), ok, egui::Button::new(RichText::new("Create").strong())).clicked() {
                    create = Some(Action::Create { name: clean.clone(), sources: sources.clone(), every: nm.every, model: nm.model.clone() });
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
        if open_models {
            self.models_open = true;
        }
        if open_runtime {
            self.runtime_open = true;
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

    /// The window on the example photos: each to download, and once it is here, to make a map from or to remove.
    fn examples_window(&mut self, ctx: &egui::Context, examples: &Examples, actions: &mut Vec<Action>) {
        if !self.examples_open {
            return;
        }
        let mut close = false;
        let mut make = None;
        let response = theme::modal(ctx, "examples", 520.0, Some("Example photos"), |ui| {
            ui.label(theme::muted("Photos to try the app with. They are downloaded into the app's folder; no map is made until you make one."));
            ui.add_space(8.0);
            for ex in examples::ALL.iter() {
                let here = examples.installed(ex);
                theme::card().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(ex.label).strong());
                        ui.label(theme::muted(format!("{} photos, {}", thousands(ex.photos), bytes_text(ex.size))));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if let Some(job) = examples.job.as_ref().filter(|j| j.id == ex.id) {
                                if ui.button("Stop").clicked() {
                                    actions.push(Action::StopExample);
                                }
                                ui.add(egui::ProgressBar::new(job.state().0).desired_width(120.0).desired_height(8.0).fill(theme::ACCENT));
                            } else if here {
                                if ui.button("Remove").clicked() {
                                    self.confirm_remove_example = Some(ex.id);
                                }
                                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new("Make a map").strong())).clicked() {
                                    make = Some(examples.dir(ex));
                                }
                            } else if ui.add_enabled(examples.job.is_none(), egui::Button::new("Get")).clicked() {
                                actions.push(Action::GetExample(ex.id.to_string()));
                            }
                        });
                    });
                    ui.label(theme::muted(ex.about));
                    if here {
                        ui.horizontal(|ui| {
                            ui.label(theme::muted(examples.dir(ex).display().to_string()));
                            if ui.small_button("Show").clicked() {
                                show_in_file_manager(&examples.dir(ex));
                            }
                        });
                    }
                });
                ui.add_space(4.0);
            }
            if let Some(id) = self.confirm_remove_example {
                ui.add_space(4.0);
                ui.label("Remove these photos? Maps made from them stay, but show them as not found.");
                ui.horizontal(|ui| {
                    if theme::tinted(ui, theme::DANGER, true, egui::Button::new("Remove")).clicked() {
                        actions.push(Action::RemoveExample(id.to_string()));
                        self.confirm_remove_example = None;
                    }
                    if ui.button("Keep them").clicked() {
                        self.confirm_remove_example = None;
                    }
                });
            }
            if let Some(e) = &examples.failure {
                ui.add_space(8.0);
                ui.label(RichText::new(e).color(Color32::from_rgb(235, 110, 100)));
            }
            ui.add_space(8.0);
            if ui.button("Close").clicked() {
                close = true;
            }
        });
        if close || make.is_some() || response.should_close() {
            self.examples_open = false;
            self.confirm_remove_example = None;
        }
        // the New map form, with the photos in it
        if let Some(dir) = make {
            self.dropped(vec![dir]);
        }
    }

    /// The window on the app's settings: zoom speed and interface size (settings.rs).
    fn settings_window(&mut self, ctx: &egui::Context, settings: &mut crate::settings::Settings) {
        if !self.settings_open {
            return;
        }
        let mut close = false;
        let response = theme::modal(ctx, "settings", 380.0, Some("Settings"), |ui| {
            ui.label(theme::muted("ZOOM SPEED"));
            theme::segmented(ui, &mut settings.zoom_speed, &crate::settings::ZOOM_SPEEDS);
            ui.label(theme::muted("The wheel, a trackpad pinch and the zoom keys."));
            ui.add_space(10.0);
            ui.label(theme::muted("INTERFACE SIZE"));
            theme::segmented(ui, &mut settings.ui_size, &crate::settings::UI_SIZES);
            ui.add_space(10.0);
            if ui.button("Close").clicked() {
                close = true;
            }
        });
        if close || response.should_close() {
            self.settings_open = false;
        }
    }

    /// The window on the models: which are installed, what each takes, and a way to get or remove each.
    fn models_window(&mut self, ctx: &egui::Context, rt: &Manager, store: &ModelStore, actions: &mut Vec<Action>) {
        if !self.models_open {
            return;
        }
        let mut close = false;
        let mut open_runtime = false;
        let response = theme::modal(ctx, "models", 560.0, Some("Models"), |ui| {
            ui.label(theme::muted("A map is made and searched with one model. Only the models you have are offered when a map is made. They are downloaded from their publishers on Hugging Face."));
            ui.add_space(8.0);
            if rt.setup.ort.is_none() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("ONNX Runtime has not been downloaded: no model can run without it.").color(Color32::from_rgb(235, 110, 100)));
                    if ui.button("Runtime…").on_hover_text("Get it in the Runtime window").clicked() {
                        open_runtime = true;
                    }
                });
                ui.add_space(4.0);
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
                                ui.label(theme::muted("downloaded"));
                            } else if ui.add_enabled(!store.fetching(), egui::Button::new("Get")).clicked() {
                                actions.push(Action::GetModel(spec.id.to_string()));
                            }
                        });
                    });
                    ui.label(theme::muted(spec.note));
                    if store.installed(spec) {
                        if let Some(t) = store.taken.get(spec.id) {
                            ui.label(theme::muted(format!("On disk: {} for its files, {} prepared for the Apple chip or graphics card and for naming", bytes_text(t.files), bytes_text(t.prepared))));
                        }
                    }
                });
                ui.add_space(4.0);
            }
            let total: u64 = store.taken.values().map(|t| t.files + t.prepared).sum();
            if total > 0 {
                ui.label(theme::muted(format!("Models take {} in all.", bytes_text(total))));
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
        if close || open_runtime || response.should_close() {
            self.models_open = false;
            self.confirm_remove_model = None;
        }
        if open_runtime {
            self.runtime_open = true;
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
            let installed = rt.installed();
            // what photo-map downloaded is shown by its version and size; anything else by where it was found
            let what = |place: Option<&crate::runtime::Place>| match place {
                None => "not downloaded".to_string(),
                Some(p) => match &p.source {
                    crate::runtime::Source::Managed(id) => installed.iter().find(|i| &i.id == id).map_or_else(|| manager::describe(p), |i| format!("{}, {:.0} MB", i.version, i.bytes as f64 / 1048576.0)),
                    _ => manager::describe(p),
                },
            };
            row(ui, "ONNX Runtime", format!("{}{}", what(rt.setup.ort.as_ref()), if rt.setup.ort.is_none() || mac || rt.setup.ort_cuda { "" } else { "; processor only" }));
            if !mac {
                row(ui, "CUDA", what(rt.setup.gpu.as_ref().map(|g| &g.cuda)));
                row(ui, "cuDNN", what(rt.setup.gpu.as_ref().map(|g| &g.cudnn)));
                row(ui, "TensorRT", what(rt.setup.tensorrt.as_ref()));
            }
            for note in &rt.wrong {
                ui.label(theme::muted(format!("{note}, so it is not used.")));
            }
            let ours = installed.iter().find(|i| i.id == crate::packs::ffmpeg().id);
            row(ui, "ffmpeg", match (&rt.ffmpeg, ours) {
                (Some(_), Some(i)) => format!("{}, {:.0} MB (reads videos)", i.version, i.bytes as f64 / 1048576.0),
                (Some(p), None) => format!("found at {} (reads videos)", p.display()),
                (None, _) => "not downloaded: maps can't be made from videos".into(),
            });
            if rt.ffmpeg.is_none() && !rt.installing() {
                ffmpeg_button(ui, actions);
            }
            ui.add_space(8.0);
            runtime_offer(ui, rt, actions);

            ui.add_space(8.0);
            ui.label("Run the models on");
            for b in crate::backends::Backend::ALL.into_iter().filter(|b| b.exists_here()) {
                let available = b.available(&rt.setup) && !(matches!(b, crate::backends::Backend::Cuda | crate::backends::Backend::TensorRt) && rt.nvidia.is_none());
                let on = available && rt.backends.is_on(b, &rt.setup);
                ui.horizontal(|ui| {
                    if ui.add_enabled(available, egui::RadioButton::new(on, b.label())).on_hover_text(b.note()).clicked() && !on {
                        actions.push(Action::SetBackend(b));
                    }
                    if !available {
                        ui.label(theme::muted(if rt.setup.ort.is_none() { "needs ONNX Runtime" } else { "not downloaded" }));
                    }
                });
            }
            ui.label(theme::muted("Whatever the chosen one can't run goes to the processor. Changes apply after a restart."));

            if !installed.is_empty() {
                ui.add_space(8.0);
                ui.label(theme::muted("Libraries that were already on this computer are used when they fit but never removed from here."));
                ui.add_space(4.0);
                if self.confirm_remove {
                    ui.horizontal(|ui| {
                        ui.label("Remove them all? The models run slower or not at all until they are downloaded again.");
                        if theme::tinted(ui, theme::DANGER, true, egui::Button::new("Remove")).clicked() {
                            actions.push(Action::RemoveRuntime);
                            self.confirm_remove = false;
                        }
                        if ui.button("Keep them").clicked() {
                            self.confirm_remove = false;
                        }
                    });
                } else {
                    let total: u64 = installed.iter().map(|p| p.bytes).sum();
                    if ui.add_enabled(!rt.installing(), egui::Button::new(format!("Remove all ({:.0} MB)", total as f64 / 1048576.0))).clicked() {
                        self.confirm_remove = true;
                    }
                }
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
            self.confirm_remove = false;
        }
    }
}

/// What the runtime could use that is not installed, as a box with the way to get it - or the install under way, or the note that
/// a restart is needed.
fn runtime_offer(ui: &mut Ui, rt: &Manager, actions: &mut Vec<Action>) {
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
                let names: Vec<&str> = offer.packs.iter().map(|p| p.label).collect();
                ui.label(theme::muted(format!("{}: about {} in all. It is kept in the runtime folder of the app and can be removed from the Runtime window.", names.join(", "), size_text(offer.mb()))));
                ui.horizontal(|ui| {
                    if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new(offer.title).strong())).clicked() {
                        actions.push(Action::InstallRuntime);
                    }
                });
            });
            ui.add_space(8.0);
        }
    }
}

/// The button that downloads ffmpeg, with its size.
fn ffmpeg_button(ui: &mut Ui, actions: &mut Vec<Action>) {
    if ui.button(format!("Download ffmpeg ({})", size_text(crate::packs::ffmpeg().size_mb))).clicked() {
        actions.push(Action::InstallFfmpeg);
    }
}

fn bytes_text(b: u64) -> String {
    size_text((b / 1_000_000) as u32)
}

fn size_text(mb: u32) -> String {
    if mb >= 1000 { format!("{:.1} GB", mb as f32 / 1000.0) } else { format!("{mb} MB") }
}

/// One step of the list of steps: a dot (filled when done, ringed when current, empty when still to come) and its name.
/// Plain text on the card, joined by a thin line, so it reads as a path and not as buttons.
fn step(ui: &mut Ui, step: Stage, current: Stage, last: bool) {
    let short = match step {
        Stage::Videos => "Videos",
        Stage::Scan => "Find",
        Stage::Embed => "Understand",
        Stage::Layout => "Lay out",
        Stage::Tree => "Link",
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
/// "12 photos", "3 videos", "12 photos and 3 videos": `n` things, `videos` of them videos.
fn photos_and_videos(n: usize, videos: usize) -> String {
    let count = |k: usize, one: &str, many: &str| format!("{} {}", thousands(k), if k == 1 { one } else { many });
    match videos {
        0 => count(n, "photo", "photos"),
        v if v == n => count(v, "video", "videos"),
        v => format!("{} and {}", count(n - v, "photo", "photos"), count(v, "video", "videos")),
    }
}

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

/// An amount of video: "45 s", "38 min", "1 h 54 min".
fn video_time(secs: usize) -> String {
    match secs {
        s if s < 60 => format!("{s} s"),
        s if s < 3600 => format!("{} min", s / 60),
        s => format!("{} h {} min", s / 3600, s % 3600 / 60),
    }
}

/// The cover of a map: read cover.png if it is there, else make it from the first four photos it can read and keep it.
fn cover_image(map: &Path) -> Option<egui::ColorImage> {
    let file = map.join("cover.png");
    let img = match image::open(&file) {
        Ok(i) => i.to_rgba8(),
        Err(_) => {
            const SIZE: u32 = 384;
            let items: Vec<crate::data::Item> = serde_json::from_slice(&std::fs::read(map.join("items.json")).ok()?).ok()?;
            let paths = items.iter().enumerate().take(64).map(|(i, item)| crate::data::item_path(map, i, item));
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

#[cfg(test)]
mod shots {
    use super::*;
    use crate::shots::{app_folder, harness, save};
    use std::time::{Duration, Instant};

    /// The Maps page in one state: the page itself (which windows are open), what it is drawn from, and the maps being made.
    struct Page {
        home: Home,
        lib: Library,
        rt: Manager,
        store: ModelStore,
        examples: Examples,
        builds: Vec<BuildView>,
    }

    fn page(set: impl FnOnce(&mut Home, &Library)) -> Page {
        let root = app_folder("maps-page", &[("Takeout tagged", 18_412, 0, "2026-10-11 14:00"), ("Main folder", 88_105, 0, "2026-09-02 09:12"), ("Istanbul trip, spring 2019", 1_204, 87, "2026-10-30 21:40"), ("Scans", 312, 0, "")]);
        let lib = Library::new(root.join("maps"));
        let mut home = Home::new();
        set(&mut home, &lib);
        Page { home, rt: Manager::new(root.clone(), || {}), store: ModelStore::new(root.clone(), || {}), examples: Examples::new(root, || {}), lib, builds: Vec::new() }
    }

    fn building(running: bool, error: Option<&str>) -> BuildView {
        let now = Instant::now();
        let state = State {
            stage: Stage::Embed,
            done: 4_210,
            total: 12_000,
            unit: "pictures",
            message: "making the photos' numbers".into(),
            stage_started: now - Duration::from_secs(300),
            base: (now - Duration::from_secs(300), 0),
            samples: Default::default(),
            error: error.map(String::from),
            finished: false,
            recent: Default::default(),
            vector_head: vec![0.12, -0.03, 0.4, 0.07, -0.21, 0.0, 0.33, -0.1],
            unread_videos: Vec::new(),
        };
        BuildView { dir: "/maps/new.building".into(), name: "Phone 2024".into(), source: "/photos/phone (12,000 photos)".into(), stage: Stage::Embed, running, backend: "on the graphics card".into(), queued: false, elsewhere: false, unread_videos: Vec::new(), state: Some(state) }
    }

    /// Draws `p` (`size`: the window) and saves it as .local/shots/maps-<name>.png. Covers load on threads: it draws until they are in.
    fn shot(name: &str, size: egui::Vec2, p: Page) {
        let mut h = harness(size, p, |ctx, p: &mut Page| {
            let _ = p.home.show(ctx, &p.lib, &p.builds, false, None, &p.rt, &p.store, &p.examples, &mut crate::settings::Settings::default());
        });
        for _ in 0..20 {
            h.run_steps(2);
            std::thread::sleep(Duration::from_millis(30));
        }
        save(h.render().expect("render"), &format!("maps-{name}"));
    }

    #[test]
    #[ignore = "needs a GPU; writes pictures to look at"]
    fn shots() {
        let _turn = crate::shots::gpu_turn();
        let size = egui::vec2(1200.0, 780.0);
        shot("1-list", size, page(|_, _| {}));
        shot("2-narrow", egui::vec2(640.0, 700.0), page(|_, _| {}));
        shot("3-details", size, page(|h, lib| h.details = Some(lib.maps[0].path.clone())));
        shot("4-new-map", size, page(|h, _| h.open_new_map()));
        let mut videos = page(|h, _| {
            h.open_new_map();
            let nm = h.new_map.as_mut().unwrap();
            nm.sources = vec!["/photos/dogs".into(), "/photos/dogs/walk.mp4".into()];
            nm.counted = "/photos/dogs\n/photos/dogs/walk.mp4".into();
            (nm.photos, nm.videos) = (Some(33), 5);
        });
        videos.rt.ffmpeg = None; // as on a computer without it
        shot("4b-new-map-videos", size, videos);
        shot("5-runtime", size, page(|h, _| h.runtime_open = true));
        shot("6-models", size, page(|h, _| h.models_open = true));
        shot("7-examples", size, page(|h, _| h.examples_open = true));
        shot("7b-settings", size, page(|h, _| h.settings_open = true));
        shot("8-delete", size, page(|h, lib| h.confirm_delete = Some(lib.maps[1].path.clone())));
        shot("9-building", size, Page { builds: vec![building(true, None)], ..page(|_, _| {}) });
        let mut videos = building(true, None);
        if let Some(s) = videos.state.as_mut() {
            let now = Instant::now();
            (s.stage, s.done, s.total, s.unit, s.message, s.base) = (Stage::Videos, 2_280, 6_862, "seconds of video", String::new(), (now - Duration::from_secs(10), 0));
        }
        videos.stage = Stage::Videos;
        shot("9b-building-videos", size, Page { builds: vec![videos], ..page(|_, _| {}) });
        let mut unread = building(true, None);
        unread.unread_videos = vec!["DJI_0041.MP4".into(), "broken.mov".into(), "IMG_2207.MOV".into()];
        shot("9c-videos-unread", size, Page { builds: vec![unread], ..page(|_, _| {}) });
        shot("10-build-failed", size, Page { builds: vec![building(false, Some("the model ran out of memory on the graphics card"))], ..page(|_, _| {}) });
        shot("11-dragging", size, page(|h, _| h.drag_over = true));
    }
}
