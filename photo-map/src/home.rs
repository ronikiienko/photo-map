//! The Maps page: the maps you can open, the ones being made, and the way to start a new one.

use crate::build::{self, Stage, State};
use crate::library::Library;
use crate::theme;
use egui::{Align, Align2, Color32, Layout, RichText, Sense, TextureHandle, Ui, vec2};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

pub enum Action {
    Open(PathBuf),
    Create { name: String, folder: PathBuf },
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
    pub source: PathBuf,
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

struct NewMap {
    folder: String,
    name: String,
    name_edited: bool,
    counted: String,
    count: Option<Receiver<usize>>,
    photos: Option<usize>,
}

pub struct Home {
    covers: HashMap<PathBuf, Cover>,
    new_map: Option<NewMap>,
    confirm_discard: Option<PathBuf>,
    confirm_delete: Option<PathBuf>,
}

const COVER: f32 = 200.0;

impl Home {
    pub fn new() -> Self {
        Self { covers: HashMap::new(), new_map: None, confirm_discard: None, confirm_delete: None }
    }

    pub fn open_new_map(&mut self) {
        self.new_map = Some(NewMap { folder: String::new(), name: String::new(), name_edited: false, counted: String::new(), count: None, photos: None });
    }

    pub fn forget_cover(&mut self, map: &Path) {
        self.covers.remove(map);
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
        self.dialogs(ctx, builds, lib, &mut actions);
        actions
    }

    fn page(&mut self, ui: &mut Ui, lib: &Library, builds: &[BuildView], resume: bool, ready_msg: Option<&str>, actions: &mut Vec<Action>) {
        ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
            ui.label(RichText::new("Maps").size(26.0).strong());
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.add(egui::Button::new(RichText::new("+  New map").strong()).fill(theme::ACCENT.gamma_multiply(0.6)).min_size(vec2(112.0, 34.0))).clicked() {
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
                        if ui.add(egui::Button::new(RichText::new("Make your first map").strong()).fill(theme::ACCENT.gamma_multiply(0.6)).min_size(vec2(180.0, 34.0))).clicked() {
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
                ui.label(theme::muted(format!("from {}", b.source.display())));
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
                    } else if ui.add(egui::Button::new(if failed { "Try again" } else { "Continue" }).fill(theme::ACCENT.gamma_multiply(0.5))).clicked() {
                        actions.push(Action::Continue(b.dir.clone()));
                    }
                });
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                for s in Stage::ALL {
                    if s == Stage::Finish {
                        continue;
                    }
                    step_chip(ui, s, b.stage);
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
                        line += &format!(" · {} of {}", thousands(s.done), thousands(s.total));
                        let secs = s.stage_started.elapsed().as_secs_f32();
                        if s.done > 0 && secs > 5.0 && s.done < s.total {
                            line += &format!(" · about {} left", duration(secs * (s.total - s.done) as f32 / s.done as f32));
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
                let (rect, response) = ui.allocate_exact_size(vec2(COVER, COVER + 52.0), Sense::click());
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
                response
            })
            .inner;
        if response.clicked() {
            actions.push(Action::Open(e.path.clone()));
        }
        response.context_menu(|ui| {
            if ui.button("Delete map…").clicked() {
                self.confirm_delete = Some(e.path.clone());
                ui.close();
            }
        });
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
            egui::Window::new("Discard this map?").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                ui.label(format!("“{name}” and the work done on it so far will be deleted. Your photos are not touched."));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new("Discard").fill(Color32::from_rgb(150, 60, 55))).clicked() {
                        actions.push(Action::Discard(dir.clone()));
                        close = true;
                    }
                    if ui.button("Keep it").clicked() {
                        close = true;
                    }
                });
            });
            if close {
                self.confirm_discard = None;
            }
        }
        if let Some(path) = self.confirm_delete.clone() {
            let name = lib.maps.iter().find(|m| m.path == path).map_or(String::new(), |m| m.name.clone());
            let mut close = false;
            egui::Window::new("Delete this map?").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
                ui.label(format!("“{name}” will be deleted, with the copies of its photos inside it. The photos you made it from are not touched."));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new("Delete").fill(Color32::from_rgb(150, 60, 55))).clicked() {
                        actions.push(Action::Delete(path.clone()));
                        close = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
            if close {
                self.confirm_delete = None;
            }
        }
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
        egui::Window::new("New map").collapsible(false).resizable(false).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).fixed_size([460.0, 0.0]).show(ctx, |ui| {
            ui.label("Folder with photos");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut nm.folder).desired_width(340.0).hint_text("/path/to/photos"));
                if ui.button("Choose…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().set_title("Folder with photos").pick_folder() {
                        nm.folder = p.display().to_string();
                    }
                }
            });
            let folder = PathBuf::from(nm.folder.trim());
            if nm.counted != nm.folder {
                nm.counted = nm.folder.clone();
                nm.photos = None;
                nm.count = None;
                if folder.is_dir() {
                    let (tx, rx) = channel();
                    let ctx = ui.ctx().clone();
                    let f = folder.clone();
                    std::thread::spawn(move || {
                        let mut v = Vec::new();
                        build::walk_public(&f, &mut v);
                        let _ = tx.send(v.len());
                        ctx.request_repaint();
                    });
                    nm.count = Some(rx);
                }
                if !nm.name_edited {
                    nm.name = folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                }
            }
            ui.add_space(4.0);
            if !folder.is_dir() {
                ui.label(theme::muted(if nm.folder.trim().is_empty() { "Pick the folder the photos are in (its sub-folders are included)." } else { "That folder does not exist." }));
            } else if nm.count.is_some() {
                ui.label(theme::muted("Counting photos…"));
            } else if let Some(n) = nm.photos {
                ui.label(theme::muted(format!("{} files found", thousands(n))));
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
            ui.label(theme::muted("It can take a while for many photos. You can pause, close the app and carry on later."));
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let ok = problem.is_none() && folder.is_dir() && nm.photos != Some(0);
                if ui.add_enabled(ok, egui::Button::new(RichText::new("Create").strong()).fill(theme::ACCENT.gamma_multiply(0.6))).clicked() {
                    create = Some(Action::Create { name: clean.clone(), folder: folder.clone() });
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
        if close {
            self.new_map = None;
        }
    }
}

fn step_chip(ui: &mut Ui, step: Stage, current: Stage) {
    let (fill, text, mark) = if step < current {
        (Color32::from_rgb(30, 45, 38), theme::OK, "✓ ")
    } else if step == current {
        (theme::ACCENT.gamma_multiply(0.3), theme::TEXT, "")
    } else {
        (Color32::from_rgb(30, 33, 41), theme::MUTED, "")
    };
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
    egui::Frame::new().fill(fill).corner_radius(10).inner_margin(egui::Margin::symmetric(8, 2)).show(ui, |ui| {
        ui.label(RichText::new(format!("{mark}{short}")).size(12.0).color(text));
    });
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
