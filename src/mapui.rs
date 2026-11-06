//! The controls on a map's page around the search box (searchbar.rs): the way back and the map's name (top left), how the photos
//! are placed and drawn (top right), the help, a photo's menu and its details, the card while a map opens, a notice. Each is drawn
//! from plain values and says what was asked in `UiActions`, so the tests below can draw them without a map, a window or a GPU.

use crate::{UiActions, View, files, groups, theme};
use egui::{Align, Align2, Color32, Layout, Margin, RichText, vec2};
use std::path::PathBuf;

/// The top bar's corners: how far down they are (under the search box when the window is narrow).
pub fn corners_y(screen_w: f32, width: f32) -> f32 {
    if screen_w < width + 2.0 * 250.0 { 64.0 } else { 9.0 }
}

/// Top left: the back arrow and the map's name.
pub fn back(ctx: &egui::Context, y: f32, name: Option<&str>, actions: &mut UiActions) {
    egui::Area::new("where".into()).anchor(Align2::LEFT_TOP, [12.0, y]).show(ctx, |ui| {
        egui::Frame::NONE.inner_margin(Margin::symmetric(6, 4)).show(ui, |ui| {
            ui.horizontal(|ui| {
                let (rect, back) = ui.allocate_exact_size(vec2(28.0, 28.0), egui::Sense::click());
                let colour = if back.hovered() { Color32::WHITE } else { theme::MUTED };
                if back.hovered() {
                    ui.painter().rect_filled(rect, theme::RADIUS_SMALL, Color32::from_white_alpha(14));
                }
                let c = rect.center();
                let stroke = egui::Stroke::new(1.8, colour);
                ui.painter().line_segment([c + vec2(6.0, 0.0), c + vec2(-6.0, 0.0)], stroke);
                ui.painter().line_segment([c + vec2(-6.0, 0.0), c + vec2(-1.0, -5.0)], stroke);
                ui.painter().line_segment([c + vec2(-6.0, 0.0), c + vec2(-1.0, 5.0)], stroke);
                if back.on_hover_text("back to your maps (Esc)").clicked() {
                    actions.back = true;
                }
                if let Some(name) = name {
                    ui.label(RichText::new(name).color(theme::MUTED));
                    ui.add_space(6.0);
                }
            });
        });
    });
}

/// How the map looks, as the Layout and Options menus change it.
pub struct Look<'a> {
    pub view: &'a mut View,
    pub groups_text: &'a mut String,
    pub ideas: &'a mut groups::Ideas,
    pub smooth: &'a mut bool,
}

/// Top right: the Layout and Options menus.
pub fn look(ctx: &egui::Context, y: f32, look: Look, actions: &mut UiActions) {
    egui::Area::new("look".into()).anchor(Align2::RIGHT_TOP, [-12.0, y]).show(ctx, |ui| {
        egui::Frame::NONE.inner_margin(Margin::symmetric(6, 4)).show(ui, |ui| {
            ui.horizontal(|ui| {
                // they stay open while you change things: they close on a click outside them
                let stay_open = || egui::containers::menu::MenuConfig::new().close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside);
                egui::containers::menu::MenuButton::new("Layout").config(stay_open()).ui(ui, |ui| {
                    ui.set_width(300.0);
                    ui.with_layout(Layout::top_down(Align::Min), |ui| {
                        ui.label(theme::muted("PLACE THE PHOTOS"));
                        let options = [(View::Map, "Map", "similar photos together, as the map was made"), (View::Groups, "Groups", "sorted into groups by words you choose")];
                        if theme::segmented(ui, look.view, &options) && *look.view == View::Map {
                            actions.arrange = true;
                        }
                        if *look.view == View::Map {
                            ui.label(theme::muted("Photos that look alike sit close together, in regions with names."));
                        } else {
                            ui.label(theme::muted("Each photo joins the word it is most like. In each group the strongest matches are in the middle, the weakest at the edge."));
                            ui.label(theme::muted("The model can tell which word fits a photo better, but not where a word stops fitting at all. So a photo that fits none of them still lands in some group, at its edge."));
                            ui.add_space(6.0);
                            // as the search box: the empty box shows a set of words, up / down another (or put another in place of the
                            // words in it), Tab takes it to change, Enter arranges by the words (the one showing, in the empty box)
                            let id = egui::Id::new("groups-box");
                            let mut arrange = false;
                            if ui.memory(|m| m.has_focus(id)) {
                                let key = |k| ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, k));
                                let (down, up, tab, enter) = (key(egui::Key::ArrowDown), key(egui::Key::ArrowUp), look.groups_text.trim().is_empty() && key(egui::Key::Tab), key(egui::Key::Enter));
                                if down || up {
                                    let idea = look.ideas.switch(if down { 1 } else { -1 });
                                    if !look.groups_text.trim().is_empty() {
                                        *look.groups_text = idea.to_string();
                                    }
                                }
                                if tab {
                                    *look.groups_text = look.ideas.idea().to_string();
                                }
                                arrange = enter;
                            }
                            ui.add(egui::TextEdit::multiline(look.groups_text).id(id).hint_text(look.ideas.idea()).desired_rows(2).desired_width(f32::INFINITY));
                            let faint = theme::MUTED.gamma_multiply(0.7);
                            let mut hint = egui::text::LayoutJob::default();
                            hint.append("↑↓", 0.0, egui::TextFormat::simple(egui::FontId::monospace(11.5), faint)); // the proportional font has no ↑↓
                            hint.append(" other words · Enter arrange · commas between them", 0.0, egui::TextFormat::simple(egui::FontId::proportional(11.5), faint));
                            ui.label(hint);
                            ui.add_space(6.0);
                            arrange |= theme::button(ui, egui::Button::new("Arrange")).on_hover_text("sort the photos into these groups").clicked();
                            if arrange {
                                if look.groups_text.trim().is_empty() {
                                    *look.groups_text = look.ideas.idea().to_string();
                                }
                                actions.arrange = true;
                            }
                        }
                    });
                });
                egui::containers::menu::MenuButton::new("Options").config(stay_open()).ui(ui, |ui| {
                    ui.set_width(280.0);
                    ui.with_layout(Layout::top_down(Align::Min), |ui| {
                        ui.label(theme::muted("PHOTO PIXELS"));
                        theme::segmented(ui, look.smooth, &[(true, "Smooth", "pixels blended with their neighbours"), (false, "Nearest", "each pixel as it is: sharp, blocky up close")]);
                    });
                });
            });
        });
    });
}

/// Bottom left: a "?" that opens how the map works and the controls.
pub fn help(ctx: &egui::Context, open: &mut bool) {
    egui::Area::new("help".into()).anchor(Align2::LEFT_BOTTOM, [12.0, -12.0]).show(ctx, |ui| {
        let button = egui::Button::new(RichText::new("?").strong()).min_size(vec2(28.0, 28.0)).selected(*open);
        if ui.add(button).on_hover_text("how this works (H)").clicked() {
            *open = !*open;
        }
    });
    if !*open {
        return;
    }
    egui::Area::new("help-card".into()).anchor(Align2::LEFT_BOTTOM, [12.0, -52.0]).show(ctx, |ui| {
        theme::card().show(ui, |ui| {
            ui.set_width(470.0); // the text wraps as wide as the keys below it
            ui.label(RichText::new("How the map works").size(17.0));
            ui.add_space(4.0);
            ui.label(theme::muted(
                "Photos that look alike sit close together: zoom in and similar ones are side by side, and the names mark what a region \
                 holds. Search lights up the photos that fit best.",
            ));
            ui.add_space(8.0);
            egui::Grid::new("keys").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
                for (keys, what) in [
                    (&["scroll", "Q", "E"][..], "zoom"),
                    (&["drag", "W", "A", "S", "D"][..], "move"),
                    (&["F"][..], "photos like the one under the mouse"),
                    (&["Space"][..], "hold: a lens over the map (scroll: how much it magnifies)"),
                    (&["click"][..], "play or pause a video; zoom to a region's name"),
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
    });
}

/// A photo's right-click menu, as it is drawn.
pub struct PhotoMenu {
    pub photo: usize,
    /// The picture it was opened on (data.rs, picture ids): the photo, or the frame a video's match beside the world shows.
    /// "Photos like this" searches with that.
    pub picture: usize,
    /// The copy beside the world it was opened on (a search's match or a moment), if any: it can be gone to on the map.
    pub copy: Option<usize>,
    /// Where it was opened (points).
    pub at: egui::Pos2,
    pub title: String,
    /// A path was started on another photo: this one can end it.
    pub ends_path: bool,
    /// Photos are searched for and this isn't one of them: it can be added.
    pub can_add: bool,
    /// A video, while a search is on: its moments that match can be shown.
    pub moments: bool,
    /// A video: "Copy image" copies the frame it was opened on.
    pub video: bool,
    /// Its file, and what to say about it (where it is; or that the original is gone and this is the map's copy).
    pub file: PathBuf,
    pub file_tip: String,
}

pub fn photo_menu(ctx: &egui::Context, m: &PhotoMenu, actions: &mut UiActions) {
    let photo = m.photo;
    egui::Area::new("photo menu".into()).fixed_pos(m.at + vec2(4.0, 4.0)).order(egui::Order::Foreground).show(ctx, |ui| {
        theme::card().inner_margin(Margin::symmetric(8, 8)).show(ui, |ui| {
            ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                // one column of buttons as wide as each other, their words at the left, like a menu
                ui.set_width(210.0);
                ui.add(egui::Label::new(theme::muted(&m.title)).truncate());
                ui.add_space(2.0);
                if let Some(k) = m.copy
                    && theme::button(ui, egui::Button::new("Go to on the map")).on_hover_text("fly to it where it is on the map").clicked()
                {
                    actions.go_to_copy = Some(k);
                }
                if m.ends_path {
                    if theme::button(ui, egui::Button::new("Path to here")).on_hover_text("a chain of photos from the start you picked to this one, each like the one before it").clicked() {
                        actions.path_to = Some(photo);
                    }
                } else if theme::button(ui, egui::Button::new("Path from here")).on_hover_text("then right-click the photo it goes to").clicked() {
                    actions.path_from = Some(photo);
                }
                if theme::button(ui, egui::Button::new("Photos like this")).on_hover_text("search with this photo (F)").clicked() {
                    actions.like = Some(m.picture);
                }
                if m.moments && theme::button(ui, egui::Button::new("Moments that match")).on_hover_text("this video's frames that match the search best (as many as the matches), right of the matches").clicked() {
                    actions.moments = Some(photo);
                }
                if m.can_add && theme::button(ui, egui::Button::new("Add to search")).on_hover_text("search for what this photo and the ones searched for share").clicked() {
                    actions.add_like = Some(m.picture);
                }
                ui.separator();
                if theme::button(ui, egui::Button::new("Show in folder")).on_hover_text(&m.file_tip).clicked() {
                    actions.reveal = Some(m.file.clone());
                    actions.done = true;
                }
                if theme::button(ui, egui::Button::new("Open")).on_hover_text("in the app your system opens it with").clicked() {
                    actions.open_file = Some(m.file.clone());
                    actions.done = true;
                }
                if theme::button(ui, egui::Button::new("Copy image")).on_hover_text(if m.video { "this frame, at the video's own size" } else { "the photo, at its own size" }).clicked() {
                    actions.copy_image = Some(m.picture);
                    actions.done = true;
                }
                if theme::button(ui, egui::Button::new("Copy path")).on_hover_text(&m.file_tip).clicked() {
                    actions.copy_path = Some(m.file.clone());
                    actions.done = true;
                }
                if theme::button(ui, egui::Button::new("Details")).on_hover_text("file, date taken, camera, place").clicked() {
                    actions.details = Some((photo, m.at));
                }
            });
        });
    });
}

/// A photo's details card ("Details" in its menu): its title, its lines (files::details) and its file. True when it was closed.
pub fn details(ctx: &egui::Context, at: egui::Pos2, title: &str, details: &files::Details, file: &std::path::Path, actions: &mut UiActions) -> bool {
    let mut close = false;
    egui::Area::new("photo details".into()).fixed_pos(at + vec2(4.0, 4.0)).order(egui::Order::Foreground).constrain(true).show(ctx, |ui| {
        theme::card().inner_margin(Margin::symmetric(12, 10)).show(ui, |ui| {
            ui.set_max_width(420.0);
            close = theme::window_header(ui, title);
            egui::Grid::new("details grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                for (name, value) in &details.lines {
                    ui.label(theme::muted(name));
                    match details.place {
                        Some((lat, lon)) if name == "Place" => {
                            ui.vertical(|ui| {
                                if !value.is_empty() {
                                    ui.add(egui::Label::new(value).wrap()).on_hover_text("the nearest town; place names from GeoNames (geonames.org)");
                                }
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
                if theme::button(ui, egui::Button::new("Show in folder")).clicked() {
                    actions.reveal = Some(file.to_path_buf());
                }
                if theme::button(ui, egui::Button::new("Open")).clicked() {
                    actions.open_file = Some(file.to_path_buf());
                }
            });
        });
    });
    close
}

/// The card in the middle while a map is being opened: its name, what is being done, and how far (if known).
pub fn opening(ctx: &egui::Context, name: &str, text: &str, fraction: Option<f32>) {
    egui::Area::new("opening".into()).anchor(Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
        theme::card().inner_margin(Margin::same(20)).show(ui, |ui| {
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

/// Something that went wrong, at the bottom of the Maps page.
pub fn notice(ctx: &egui::Context, text: &str) {
    egui::Area::new("notice".into()).anchor(Align2::CENTER_BOTTOM, [0.0, -24.0]).show(ctx, |ui| {
        theme::card().inner_margin(Margin::symmetric(14, 10)).show(ui, |ui| ui.label(RichText::new(text).color(Color32::from_rgb(235, 110, 100))));
    });
}

#[cfg(test)]
mod shots {
    use super::*;
    use crate::shots::{harness, save};
    use egui_kittest::kittest::Queryable;

    /// A map's page with no map under it: its controls, in one state.
    struct Page {
        size: egui::Vec2,
        view: View,
        groups_text: String,
        ideas: groups::Ideas,
        smooth: bool,
        help_open: bool,
        menu: Option<PhotoMenu>,
        details: Option<files::Details>,
        opening: Option<(&'static str, Option<f32>)>,
    }

    fn page() -> Page {
        Page { size: vec2(1200.0, 780.0), view: View::Map, groups_text: String::new(), ideas: groups::Ideas::new(), smooth: true, help_open: false, menu: None, details: None, opening: None }
    }

    fn menu(ends_path: bool, can_add: bool) -> PhotoMenu {
        PhotoMenu {
            photo: 0,
            picture: 0,
            copy: None,
            at: egui::pos2(520.0, 300.0),
            title: "IMG_20190612_184455 a rather long name that goes on".into(),
            ends_path,
            can_add,
            moments: false,
            video: false,
            file: "/photos/2019/IMG_20190612_184455.jpg".into(),
            file_tip: "/photos/2019/IMG_20190612_184455.jpg".into(),
        }
    }

    /// Draws `p`, does `then` to it (clicks, hovers), and saves it as .local/shots/map-<name>.png.
    fn shot(name: &str, p: Page, then: impl FnOnce(&mut egui_kittest::Harness<'static, Page>)) {
        let size = p.size;
        let mut h = harness(size, p, |ctx, p: &mut Page| {
            let mut actions = UiActions::default();
            let screen_w = ctx.content_rect().width();
            let width = (screen_w - 2.0 * 250.0).clamp(300.0, 620.0);
            let y = corners_y(screen_w, width);
            back(ctx, y, Some("Takeout tagged"), &mut actions);
            look(ctx, y, Look { view: &mut p.view, groups_text: &mut p.groups_text, ideas: &mut p.ideas, smooth: &mut p.smooth }, &mut actions);
            // where the search box goes
            egui::Area::new("top".into()).anchor(Align2::CENTER_TOP, [0.0, 8.0]).show(ctx, |ui| {
                theme::card().show(ui, |ui| {
                    ui.set_width(width - 24.0);
                    ui.label(theme::muted("(search box)"));
                });
            });
            help(ctx, &mut p.help_open);
            if let Some(m) = &p.menu {
                photo_menu(ctx, m, &mut actions);
            }
            if let Some(d) = &p.details {
                details(ctx, egui::pos2(520.0, 300.0), "IMG_20190612_184455", d, std::path::Path::new("/photos/x.jpg"), &mut actions);
            }
            if let Some((name, fraction)) = p.opening {
                opening(ctx, name, "reading the photos", fraction);
            }
        });
        h.run_steps(3);
        then(&mut h);
        h.run_steps(3);
        save(h.render().expect("render"), &format!("map-{name}"));
    }

    #[test]
    #[ignore = "needs a GPU; writes pictures to look at"]
    fn shots() {
        let _turn = crate::shots::gpu_turn();
        shot("1-plain", page(), |_| {});
        shot("2-narrow", Page { size: vec2(760.0, 600.0), ..page() }, |_| {});
        shot("3-layout-menu", page(), |h| h.get_by_label("Layout").click());
        shot("4-layout-groups", Page { view: View::Groups, groups_text: "mountain, city, forest".into(), ..page() }, |h| h.get_by_label("Layout").click());
        shot("4-layout-groups-empty", Page { view: View::Groups, ..page() }, |h| h.get_by_label("Layout").click());
        shot("5-options-menu", page(), |h| h.get_by_label("Options").click());
        shot("6-help", Page { help_open: true, ..page() }, |_| {});
        shot("7-photo-menu", Page { menu: Some(menu(false, true)), ..page() }, |_| {});
        shot("8-photo-menu-path", Page { menu: Some(menu(true, false)), ..page() }, |h| h.hover_at(egui::pos2(600.0, 345.0)));
        let lines = [("File", "/photos/2019/IMG_20190612_184455.jpg"), ("Size", "3.2 MB, 4032 × 3024"), ("Taken", "2019-06-12 18:44"), ("Camera", "Google Pixel 3"), ("Exposure", "1/120 s, f/1.8, ISO 64")];
        let d = files::Details { lines: lines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(), place: None };
        shot("9-details", Page { details: Some(d), ..page() }, |_| {});
        shot("10-opening", Page { opening: Some(("Takeout tagged", Some(0.42))), ..page() }, |_| {});
    }
}
