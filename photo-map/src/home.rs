//! The Maps page: the maps you can open, the ones being made, and the way to start a new one.

use crate::build::{self, Stage, State};
use crate::library::Library;
use crate::storage::{self, Sizes};
use crate::theme;
use egui::{Align, Align2, Color32, Layout, RichText, Sense, TextureHandle, Ui, vec2};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

pub enum Action {
    Open(PathBuf),
    Create { name: String, sources: Vec<PathBuf>, names_as_captions: bool },
    Pause(PathBuf),
    Continue(PathBuf),
    Discard(PathBuf),
    Delete(PathBuf),
    Resume(bool),
}

/// A map being made, as the page shows it.
pub struct BuildView {
    pub dir: PathBuf,
    pub name: String,
    /// Where the photos came from, in words.
    pub source: String,
    pub stage: Stage,
    pub running: bool,
    /// Waiting for the map before it to finish.
    pub queued: bool,
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
    folder: String,
    /// Folders and photos dropped on the page: used instead of the typed folder.
    dropped: Vec<PathBuf>,
    name: String,
    name_edited: bool,
    counted: String,
    count: Option<Receiver<usize>>,
    photos: Option<usize>,
    names_as_captions: bool,
}

pub struct Home {
    covers: HashMap<PathBuf, Cover>,
    sizes: HashMap<PathBuf, Measured>,
    /// The map whose breakdown is open.
    details: Option<PathBuf>,
    /// Something is being dragged over the window.
    drag_over: bool,
    new_map: Option<NewMap>,
    confirm_discard: Option<PathBuf>,
    confirm_delete: Option<PathBuf>,
}

const COVER: f32 = 200.0;

impl Home {
    pub fn new() -> Self {
        Self { covers: HashMap::new(), sizes: HashMap::new(), details: None, drag_over: false, new_map: None, confirm_discard: None, confirm_delete: None }
    }

    pub fn open_new_map(&mut self) {
        self.new_map = Some(NewMap { folder: String::new(), dropped: Vec::new(), name: String::new(), name_edited: false, counted: String::new(), count: None, photos: None, names_as_captions: false });
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
        let nm = self.new_map.as_mut().unwrap();
        let mut seen: std::collections::HashSet<PathBuf> = nm.dropped.iter().cloned().collect();
        for p in usable {
            if seen.insert(p.clone()) {
                nm.dropped.push(p);
            }
        }
        nm.dropped.sort();
        true
    }

    pub fn forget_cover(&mut self, map: &Path) {
        self.covers.remove(map);
        self.sizes.remove(map);
    }

    pub fn show(&mut self, ctx: &egui::Context, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>) -> Vec<Action> {
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
        self.dialogs(ctx, builds, lib, &mut actions);
        actions
    }

    fn page(&mut self, ui: &mut Ui, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>, actions: &mut Vec<Action>) {
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            ui.label(RichText::new("Maps").size(26.0).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), true, egui::Button::new(RichText::new("+  New map").strong()).min_size(vec2(112.0, 34.0))).clicked() {
                    self.open_new_map();
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
            let failed = b.state.as_ref().is_some_and(|s| s.error.is_some());
            ui.horizontal(|ui| {
                let colour = if failed { Color32::from_rgb(235, 110, 100) } else if b.running { theme::ACCENT } else { theme::MUTED };
                theme::dot(ui, colour);
                ui.label(RichText::new(&b.name).size(17.0));
                ui.label(theme::muted(format!("from {}", b.source)));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.button("Discard").clicked() {
                        self.confirm_discard = Some(b.dir.clone());
                    }
                    if b.running {
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
            ui.add_space(6.0);
            match &b.state {
                Some(s) if s.error.is_some() => {
                    ui.label(RichText::new(s.error.clone().unwrap_or_default()).color(Color32::from_rgb(235, 110, 100)));
                    ui.label(theme::muted("Nothing is lost: it carries on from the last finished piece."));
                }
                Some(s) if b.running => {
                    let frac = if s.total > 0 { (s.done as f32 / s.total as f32).clamp(0.0, 1.0) } else { 0.0 };
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
                    ui.label(theme::muted(line));
                }
                _ if b.queued => {
                    ui.label(theme::muted("Waiting for the map before it to finish"));
                }
                _ => {
                    ui.label(theme::muted(format!("Paused at: {}. Nothing is lost — continue any time, or discard it.", b.stage.label())));
                }
            }
        });
    }

    fn map_card(&mut self, ui: &mut Ui, e: &crate::library::Entry, actions: &mut Vec<Action>) {
        let ctx = ui.ctx().clone();
        let response = ui
            .allocate_ui(vec2(COVER, 0.0), |ui| {
                ui.set_width(COVER);
                let (rect, response) = ui.allocate_exact_size(vec2(COVER, COVER + 76.0), Sense::click());
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
                // total on disk, and a bar of what it is made of; "details" opens the explanation
                let sizes = self.sizes(&ctx, &e.path);
                let text = sizes.map_or("measuring…".to_string(), |s| format!("{} on disk", storage::human(s.total())));
                let line = rect.min + vec2(12.0, COVER + 50.0);
                let p = ui.painter();
                p.text(line, Align2::LEFT_TOP, text, egui::FontId::proportional(12.0), theme::TEXT);
                let bar = egui::Rect::from_min_size(rect.min + vec2(12.0, COVER + 68.0), vec2(COVER - 24.0, 4.0));
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
                let details = egui::Rect::from_min_size(rect.min + vec2(COVER - 62.0, COVER + 44.0), vec2(56.0, 20.0));
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

    fn dialogs(&mut self, ctx: &egui::Context, builds: &[BuildView], lib: &Library, actions: &mut Vec<Action>) {
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
        self.new_map_dialog(ctx, lib, actions);
    }

    fn new_map_dialog(&mut self, ctx: &egui::Context, lib: &Library, actions: &mut Vec<Action>) {
        let Some(nm) = &mut self.new_map else { return };
        if let Some(rx) = &nm.count {
            if let Ok(n) = rx.try_recv() {
                nm.photos = Some(n);
                nm.count = None;
            }
        }
        let mut close = false;
        let mut create = None;
        // a click on the backdrop does not close this one (what you typed would be lost): Cancel or Esc does
        theme::modal(ctx, "new-map", 460.0, Some("New map"), |ui| {
            if nm.dropped.is_empty() {
                ui.label("Folder with photos");
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut nm.folder).desired_width(340.0).hint_text("/path/to/photos, or drop folders and photos here"));
                    if ui.button("Choose…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().set_title("Folder with photos").pick_folder() {
                            nm.folder = p.display().to_string();
                        }
                    }
                });
            } else {
                ui.label("Dropped");
                ui.horizontal(|ui| {
                    ui.label(theme::muted(build::describe(&nm.dropped)));
                    if ui.button("Use a folder instead").clicked() {
                        nm.dropped.clear();
                    }
                });
            }
            let folder = PathBuf::from(nm.folder.trim());
            let sources: Vec<PathBuf> = if !nm.dropped.is_empty() {
                nm.dropped.clone()
            } else if folder.is_dir() {
                vec![folder.clone()]
            } else {
                Vec::new()
            };
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
                        let _ = tx.send(build::collect_photos(&list).len());
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
            ui.add_space(4.0);
            if sources.is_empty() {
                ui.label(theme::muted(if nm.folder.trim().is_empty() { "Pick the folder the photos are in (its sub-folders are included)." } else { "That folder does not exist." }));
            } else if nm.count.is_some() {
                ui.label(theme::muted("Counting photos…"));
            } else if let Some(n) = nm.photos {
                ui.label(theme::muted(format!("{} photos found", thousands(n))));
            }
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
                let ok = problem.is_none() && !sources.is_empty() && nm.photos != Some(0);
                if theme::tinted(ui, theme::ACCENT.gamma_multiply(0.6), ok, egui::Button::new(RichText::new("Create").strong())).clicked() {
                    create = Some(Action::Create { name: clean.clone(), sources: sources.clone(), names_as_captions: nm.names_as_captions });
                }
                if ui.button("Cancel").clicked() {
                    close = true;
                }
            });
        });
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
}

/// One step of the list of steps: a dot (filled when done, ringed when current, empty when still to come) and its name.
/// Plain text on the card, joined by a thin line, so it reads as a path and not as buttons.
fn step(ui: &mut Ui, step: Stage, current: Stage, last: bool) {
    let short = match step {
        Stage::Scan => "Find",
        Stage::Copy => "Copy",
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
