//! The search box at the top of the screen and what shows under it: the words or photos searched with, ideas while typing,
//! the filter chips, the results (step through them, how many to mark, hide them), the hidden searches and how to search.
//! It reads plain values (Bar) instead of the app, so a test can draw it on its own: `cargo test --bin photo-map shots`
//! writes each state to .local/shots/*.png.

use crate::query::{self, Filter};
use crate::{FilterEdit, UiActions, suggest, theme};
use std::collections::HashMap;
use std::path::PathBuf;

/// Whether search can run, as the dot at the box's start shows it.
pub enum Engine {
    Off(String),
    NotSearchable,
    Ready,
    Loading,
    NoMap,
}

/// What the box shows and changes.
pub struct Bar<'a> {
    pub search: &'a mut String,
    pub ideas: &'a mut suggest::Suggest,
    pub tips_open: &'a mut bool,
    pub hidden_open: &'a mut bool,
    pub wanted: &'a mut f32,
    pub filters: &'a mut Vec<Filter>,
    pub filter_edit: &'a mut Option<FilterEdit>,
    pub spots: &'a mut HashMap<PathBuf, Option<usize>>,
    pub status: &'a str,
    pub photos: &'a [(PathBuf, egui::TextureHandle)],
    pub engine: Engine,
    /// The matches: how many, which is shown (`at`), whether they are a path instead, and how many photos the map has.
    pub found: usize,
    pub at: Option<usize>,
    pub on_path: bool,
    /// How many photos can match (not hidden, passing the filters), and whether every one of them is marked (the filters are
    /// the whole search).
    pub most: f32,
    pub all_marked: bool,
    /// The hidden searches: what each was and how many photos it hid.
    pub hidden: Vec<(String, usize)>,
    /// The open map's photos, to find a photo searched for on it.
    pub data: Option<&'a crate::data::MapData>,
}

/// The box, `width` wide, centred at the top of the screen.
pub fn show(ctx: &egui::Context, width: f32, bar: Bar, actions: &mut UiActions) {
    // middle: the search box, then (only when there is something) the results, the hidden searches, and tips
    let box_id = egui::Id::new("search-box");
    let focused = ctx.memory(|m| m.has_focus(box_id));
    egui::Area::new("top".into()).anchor(egui::Align2::CENTER_TOP, [0.0, 0.0]).show(ctx, |ui| {
        // no card around it all: the box stands on its own, and what is under it gets a card only while there is something
        egui::Frame::NONE.inner_margin(egui::Margin::same(8)).show(ui, |ui| {
            ui.set_width(width - 16.0);
            ui.set_max_width(width - 16.0); // the rows below wrap at this width instead of widening the card
            // the box: whether search can run (a dot), the words or the photos searched with, a cross to clear
            // the edge stays quiet; typing in the box lights a ring just inside its search line, not around everything under it
            let ring = theme::CARD_STROKE;
            // what shows under the box (results, status, hidden searches, help) joins it: one box, a line between
            let (found, at, on_path) = (bar.found, bar.at, bar.on_path);
            let chips = !bar.hidden.is_empty() && *bar.hidden_open;
            let has_filters = !bar.filters.is_empty() || bar.filter_edit.is_some();
            let below = |tips: bool| found > 0 || !bar.status.is_empty() || chips || tips || has_filters || bar.photos.len() == 1;
            let r = theme::RADIUS_SMALL;
            let top_corners = if below(*bar.tips_open) { egui::CornerRadius { nw: r, ne: r, sw: 0, se: 0 } } else { egui::CornerRadius::same(r) };
            let search_line = egui::Frame::new().fill(ui.visuals().extreme_bg_color).stroke(egui::Stroke::new(1.0, ring)).corner_radius(top_corners).inner_margin(egui::Margin::symmetric(10, 2)).shadow(egui::Shadow { offset: [0, 4], blur: 18, spread: 0, color: egui::Color32::from_black_alpha(110) }).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.set_min_height(32.0);
                    match &bar.engine {
                        Engine::Off(why) => {
                            theme::dot(ui, theme::DANGER);
                            ui.label(theme::muted("search off")).on_hover_text(why);
                        }
                        Engine::NotSearchable => {
                            theme::dot(ui, theme::MUTED);
                            ui.label(theme::muted("not searchable")).on_hover_text("this map has no embeddings.bin of the right size");
                        }
                        Engine::Ready => {
                            let (rect, dot) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                            ui.painter().circle_filled(rect.center(), 4.0, theme::OK);
                            dot.on_hover_text("Search ready. The AI model that turns your words or photos into numbers is loaded. It runs on this computer, nothing is sent anywhere.");
                        }
                        Engine::Loading => {
                            ui.add(egui::Spinner::new().size(12.0)).on_hover_text("the search model is loading");
                        }
                        Engine::NoMap => {}
                    }
                    let clearable = !bar.search.is_empty() || !bar.photos.is_empty() || bar.found > 0;
                    // how many photos are hidden, a small badge at the box's end that opens the hidden searches
                    let hidden_total: usize = bar.hidden.iter().map(|h| h.1).sum();
                    let badge = (hidden_total > 0).then(|| format!("{hidden_total} hidden {}", if *bar.hidden_open { "⏷" } else { "⏵" }));
                    // ideas (suggest.rs): up / down switch, Tab takes one; taken before the box sees the keys
                    let words = bar.photos.is_empty();
                    let mut to_end = false;
                    if words && focused {
                        let (tab, down, up) = ui.input_mut(|i| (i.consume_key(egui::Modifiers::NONE, egui::Key::Tab), i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown), i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp)));
                        if (down || up) && let Some(whole) = bar.ideas.switch(bar.search, if down { 1 } else { -1 }) {
                            *bar.search = whole;
                            to_end = true;
                        }
                        if tab && let Some(idea) = bar.ideas.idea(bar.search) {
                            bar.ideas.took(&idea);
                            *bar.search = idea;
                            to_end = true;
                        }
                    }
                    let idea = if words { bar.ideas.idea(bar.search) } else { None };
                    let whole = words && bar.ideas.replaces(bar.search);
                    // the end of the line, from the right: the i, the cross, the hidden badge, + filter, the keys for ideas; the words or
                    // photos get what is left, so the line is always as wide as the box
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let open = *bar.tips_open;
                    if theme::info_button(ui, open).on_hover_text(if open { "close the tips" } else { "how to search" }).clicked() {
                        *bar.tips_open = !open;
                    }
                    if clearable && theme::close_button(ui) {
                        if bar.photos.is_empty() {
                            actions.clear_search = true;
                        } else {
                            actions.clear_photo = true;
                        }
                    }
                    if let Some(badge) = badge {
                        let tip = if *bar.hidden_open { "fold away the hidden searches" } else { "show the hidden searches, to bring some back or change them" };
                        if ui.add(egui::Button::new(theme::muted(badge)).frame(false).min_size(egui::vec2(0.0, 26.0))).on_hover_text(tip).clicked() {
                            *bar.hidden_open = !*bar.hidden_open;
                        }
                    }
                    // filters are chips of their own under the box: this adds one
                    // a small button of its own (filled, lighting up under the mouse and while its menu is open), not words in the box
                    let add_filter = egui::Button::new(egui::RichText::new("+ Filter").size(12.5)).frame(true).min_size(egui::vec2(0.0, 24.0));
                    ui.scope(|ui| {
                    ui.set_style(ui.ctx().global_style());
                    ui.spacing_mut().button_padding = egui::vec2(8.0, 2.0);
                    egui::containers::menu::MenuButton::from_button(add_filter).ui(ui, |ui| {
                        ui.set_width(230.0);
                        // a line each, none wrapping (a menu justifies wrapped lines, stretching their letters apart)
                        for line in ["Only photos that pass can match.", "Commas mean any of them: pixel, redmi", "Two dots make a range: 2016..2018"] {
                            ui.add(egui::Label::new(theme::muted(line)).wrap_mode(egui::TextWrapMode::Extend));
                        }
                        ui.add_space(4.0);
                        for (key, tip) in [
                            (query::Key::Name, "a word in the file and folder names, like greece"),
                            (query::Key::Camera, "the camera it was taken with, like pixel"),
                            (query::Key::Date, "when it was taken, like 2019-07"),
                            (query::Key::Place, "the town nearest to where it was taken, like lviv"),
                        ] {
                            if theme::button(ui, egui::Button::new(key.label())).on_hover_text(tip).clicked() {
                                actions.edit_filter = Some(FilterEdit { at: usize::MAX, key, not: false, value: String::new() });
                                ui.close();
                            }
                        }
                    }).0.on_hover_text("only photos that pass a filter can match: by name, camera, date or place");
                    });
                    // while in the box with an idea showing: the keys for ideas
                    if focused && (idea.is_some() || whole) {
                        // the arrows in the monospace font: the proportional one has no ↑↓
                        let faint = theme::MUTED.gamma_multiply(0.7);
                        let mut hint = egui::text::LayoutJob::default();
                        hint.append("↑↓", 0.0, egui::TextFormat::simple(egui::FontId::monospace(11.5), faint));
                        let (what, tip) = if whole { (" another search", "up and down: put another search in the box") } else { (" other idea", "up and down: another idea; Enter in the empty box: search it") };
                        hint.append(what, 0.0, egui::TextFormat::simple(egui::FontId::proportional(11.5), faint));
                        // always there (fainter with nothing to take): the words keep their room
                        let take = if idea.is_some() { faint } else { faint.gamma_multiply(0.45) };
                        hint.append(" · Tab take", 0.0, egui::TextFormat::simple(egui::FontId::proportional(11.5), take));
                        let tip = if idea.is_some() { format!("{tip}; Tab: take the faint rest") } else { tip.to_string() };
                        ui.add(egui::Label::new(hint).wrap_mode(egui::TextWrapMode::Extend)).on_hover_text(tip);
                    }
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    if !bar.photos.is_empty() {
                        // a search is photos or words, never both: while photos are the search they take the place of the words
                        for (k, (path, tex)) in bar.photos.iter().enumerate() {
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
                            let spot = if thumb.hovered() { *bar.spots.entry(path.clone()).or_insert_with(|| bar.data.and_then(|d| crate::find_photo(d, path))) } else { None };
                            if out.on_hover_text("take it out of the search").clicked() {
                                actions.drop_photo = Some(k);
                            } else if thumb.hovered() && spot.is_none() {
                                thumb.on_hover_text("not on this map: dropped in or pasted");
                            } else if thumb.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text("go to it on the map").clicked() {
                                actions.show_photo = spot;
                            }
                        }
                    } else {
                        // empty: the idea is the placeholder; typing: the rest of it shows faint after the words
                        let hint = if bar.search.is_empty() { idea.clone().unwrap_or_default() } else { String::new() };
                        let edit = egui::TextEdit::singleline(bar.search).id(box_id).hint_text(hint).frame(egui::Frame::NONE).desired_width(ui.available_width()).show(ui);
                        // the keys stay with the box: Tab and up / down would otherwise move to another button
                        ui.memory_mut(|m| m.set_focus_lock_filter(box_id, egui::EventFilter { tab: true, horizontal_arrows: true, vertical_arrows: true, ..Default::default() }));
                        if focused && !bar.search.is_empty() && let Some(idea) = &idea {
                            let rest: String = idea.chars().skip(bar.search.chars().count()).collect();
                            let at = edit.galley_pos + egui::vec2(edit.galley.rect.width(), 0.0);
                            ui.painter_at(edit.response.rect).text(at, egui::Align2::LEFT_TOP, rest, egui::TextStyle::Body.resolve(ui.style()), theme::MUTED.gamma_multiply(0.8));
                        }
                        if to_end {
                            let mut state = edit.state;
                            state.cursor.set_char_range(Some(egui::text::CCursorRange::one(egui::text::CCursor::new(bar.search.chars().count()))));
                            state.store(ui.ctx(), box_id);
                        }
                        if edit.response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            // Enter in the empty box searches the idea it shows; with filters, it searches them alone
                            if bar.search.trim().is_empty() && bar.filters.is_empty() && let Some(idea) = idea {
                                *bar.search = idea;
                            }
                            actions.ask = true;
                        }
                    }
                    });
                    });
                });
            });
            if focused {
                let inset = search_line.response.rect.shrink(2.0);
                let corners = if below(*bar.tips_open) { egui::CornerRadius { nw: r - 2, ne: r - 2, sw: 2, se: 2 } } else { egui::CornerRadius::same(r - 2) };
                ui.painter().rect_stroke(inset, corners, egui::Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.7)), egui::StrokeKind::Inside);
            }

            // the results: step through them, how many are marked, hide them
            let tips = *bar.tips_open;
            if !below(tips) {
                return;
            }
            ui.add_space(-ui.spacing().item_spacing.y - 1.0); // right under the box, sharing its edge
            // as wide as the box: the margins (10 each side) and the stroke come out of the width, not added to it
            let inner = width - 16.0 - 2.0 * 10.0 - 2.0; // from the box's width, not what is left: a row too wide can't widen the next
            let bottom_corners = egui::CornerRadius { nw: 0, ne: 0, sw: r, se: r };
            egui::Frame::new().fill(ui.visuals().extreme_bg_color).stroke(egui::Stroke::new(1.0, ring)).corner_radius(bottom_corners).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                ui.set_width(inner);
                ui.set_max_width(inner);
                // the filters: a chip each (click: change it, NOT ones in red), and the one being written
                if has_filters {
                    ui.horizontal_wrapped(|ui| {
                        let editing = bar.filter_edit.as_ref().map(|e| e.at);
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
                        for k in 0..=bar.filters.len() {
                            if editing == Some(k) {
                                let e = bar.filter_edit.as_mut().unwrap();
                                fits(ui, chip_width(ui, chip_text(e.key, false, "")) + 40.0 + 130.0 + 6.0);
                                chip(ui, true, |ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    // its name, a NOT button (lit when on: leave these out), then the values
                                    ui.add(egui::Label::new(chip_text(e.key, false, "")).wrap_mode(egui::TextWrapMode::Extend));
                                    let not = egui::RichText::new("NOT").size(11.5).color(if e.not { egui::Color32::WHITE } else { theme::MUTED });
                                    let not = egui::Button::new(not).fill(if e.not { theme::DANGER } else { ui.visuals().extreme_bg_color }).min_size(egui::vec2(0.0, 20.0));
                                    let head = ui.scope(|ui| {
                                        ui.spacing_mut().button_padding = egui::vec2(6.0, 1.0);
                                        ui.add(not)
                                    }).inner.on_hover_text(if e.not { "leaving these out: click to keep only these instead" } else { "click to leave these out instead of keeping only these" });
                                    if head.clicked() {
                                        e.not = !e.not;
                                        ui.memory_mut(|m| m.request_focus(egui::Id::new("filter-edit")));
                                    }
                                    let edit = ui.add(egui::TextEdit::singleline(&mut e.value).id(egui::Id::new("filter-edit")).hint_text(e.key.hint()).frame(egui::Frame::NONE).desired_width(130.0));
                                    let cross = small_cross(ui).on_hover_text("drop this filter");
                                    if cross.clicked() {
                                        e.value.clear();
                                        actions.filter_done = true;
                                    } else if edit.lost_focus() && !head.hovered() {
                                        actions.filter_done = true;
                                    }
                                });
                                continue;
                            }
                            let Some(f) = bar.filters.get(k) else { continue };
                            fits(ui, chip_width(ui, chip_text(f.key, f.not, &f.value_text())));
                            chip(ui, false, |ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                let tip = if f.not { "photos this picks are left out; click to change it" } else { "only photos this picks can match; click to change it" };
                                let text = ui.add(egui::Label::new(chip_text(f.key, f.not, &f.value_text())).wrap_mode(egui::TextWrapMode::Extend).sense(egui::Sense::click()));
                                if text.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(tip).clicked() {
                                    actions.edit_filter = Some(FilterEdit { at: k, key: f.key, not: f.not, value: f.value_text() });
                                }
                                if small_cross(ui).on_hover_text("take this filter out").clicked() {
                                    actions.drop_filter = Some(k);
                                }
                            });
                        }
                    });
                    if found > 0 || !bar.status.is_empty() || chips || tips {
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
                        if on_path || bar.all_marked || bar.most <= 10.0 {
                            // a path, the filters alone, or no more photos able to match than are marked: nothing to choose
                            ui.label(format!("{found} {}", if found == 1 { "photo" } else { "photos" }));
                            if on_path && !bar.status.is_empty() {
                                ui.label(theme::muted(bar.status));
                            }
                        } else {
                            // how many to mark, right where they are counted; the similarities they span in its tip
                            let most = bar.most.max(11.0);
                            let tip = format!("{}\n\nMarks this many photos, best first, yellow to red. The model knows which fit better, not where fitting stops: only you can tell. All of them fit? Try more. Junk at the end? Lower it.", bar.status);
                            let tip = tip.trim_start();
                            // from the right: Hide these, the number (dragged slowly or typed), and the slider in all the room left, so
                            // nothing leaves a gap and a pixel moves it little even over millions
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                if theme::button(ui, egui::Button::new("Hide these")).on_hover_text("hide these matches: they turn faint, stay where they are, and no search marks them; the chip that appears brings them back").clicked() {
                                    actions.hide = true;
                                }
                                ui.add_space(4.0);
                                let number = egui::DragValue::new(bar.wanted).range(10.0..=most).speed(0.2).fixed_decimals(0).suffix(if ui.available_width() < 260.0 { "" } else { " matches" }); // a narrow box: just the number
                                let number = ui.add(number).on_hover_text(tip);
                                // the slider only where there is room for it (a narrow window keeps the number)
                                let room = ui.available_width() - ui.spacing().item_spacing.x;
                                ui.spacing_mut().slider_width = room;
                                let slider = (room >= 60.0).then(|| ui.add(egui::Slider::new(&mut *bar.wanted, 10.0..=most).logarithmic(true).show_value(false)).on_hover_text(tip));
                                if number.changed() || slider.is_some_and(|s| s.changed()) {
                                    *bar.wanted = (*bar.wanted).round();
                                    actions.reblend = true;
                                }
                            });
                        }
                    });
                } else if !bar.status.is_empty() {
                    // what happened, when there are no results to show it with (searching…, an error, what was hidden)
                    ui.add_space(4.0);
                    ui.label(theme::muted(bar.status));
                }

                // the hidden searches, opened from the badge in the box: a chip each
                if chips {
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        for (k, (what, n)) in bar.hidden.iter().enumerate() {
                            // the chip takes the search back into the box; its cross just brings the photos back
                            let mut job = egui::text::LayoutJob::default();
                            let font = egui::TextStyle::Button.resolve(ui.style());
                            job.append(what, 0.0, egui::TextFormat::simple(font.clone(), ui.visuals().text_color()));
                            job.append(&n.to_string(), 6.0, egui::TextFormat::simple(font, theme::MUTED));
                            fits(ui, chip_width(ui, job.clone()));
                            chip(ui, false, |ui| {
                                ui.spacing_mut().item_spacing.x = 4.0;
                                let text = ui.add(egui::Label::new(job).wrap_mode(egui::TextWrapMode::Extend).sense(egui::Sense::click()));
                                if text.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text("search this again, to change it and hide again (its photos come back meanwhile)").clicked() {
                                    actions.unhide = Some((k, true));
                                }
                                if small_cross(ui).on_hover_text("show these photos again").clicked() {
                                    actions.unhide = Some((k, false));
                                }
                            });
                        }
                    });
                }

                // how to search, while the i is pointed at
                if tips {
                    if found > 0 || !bar.status.is_empty() || chips {
                        ui.add_space(6.0); // a gap only below something
                    }
                    let heading = |ui: &mut egui::Ui, text: &str| {
                        ui.add_space(if text.starts_with("SEARCH WITH WORDS") { 0.0 } else { 6.0 });
                        ui.label(egui::RichText::new(text).size(11.0).color(theme::MUTED).strong());
                    };
                    heading(ui, "SEARCH WITH WORDS");
                    ui.label(theme::muted("Describe what is in the photo, in a short phrase. NOT in capitals pushes things down. Try one:"));
                    if bar.photos.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
                        ui.spacing_mut().button_padding = egui::vec2(8.0, 2.0);
                        for example in ["woman on a bike in winter", "street at night NOT cars", "foggy forest", "birthday cake", "old black and white photo"] {
                            if ui.add(egui::Button::new(egui::RichText::new(example).size(12.5))).on_hover_text("search for this").clicked() {
                                *bar.search = example.to_string();
                                actions.ask = true;
                                *bar.tips_open = false;
                            }
                        }
                    });
                    }
                    heading(ui, "SEARCH WITH PHOTOS");
                    ui.label(theme::muted("Drop photos on the window, or right-click one on the map: Photos like this. With several, what they share is found."));
                    heading(ui, "KEEP ONLY SOME PHOTOS");
                    ui.label(theme::muted("+ Filter: by file name, camera, date or place. It works with words, with photos, or alone."));
                    heading(ui, "WHAT IT CAN AND CAN'T TELL");
                    ui.label(theme::muted("Good at things, places, scenes, light, weather, mood and style. Bad at counting, left and right, and who someone is."));
                }

                // one photo searched for: that more can join it, and when that helps
                if bar.photos.len() == 1 {
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
}

/// A chip in a row: `add` puts its text and cross in one pill, which lights up under the mouse as a button does (a click on its
/// text changes it). The one being written has a ring instead.
fn chip<R>(ui: &mut egui::Ui, editing: bool, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let mut prepared = pill(ui.visuals(), editing).begin(ui);
    let inner = prepared.content_ui.horizontal(add).inner;
    let response = prepared.allocate_space(ui);
    if !editing && ui.rect_contains_pointer(response.rect) {
        let hovered = &ui.visuals().widgets.hovered;
        prepared.frame.fill = hovered.weak_bg_fill;
        prepared.frame.stroke = hovered.bg_stroke;
    }
    prepared.paint(ui);
    inner
}

/// A chip: one rounded pill, its text then a small cross inside it; the one being written has a ring.
fn pill(visuals: &egui::Visuals, editing: bool) -> egui::Frame {
    let stroke = if editing { egui::Stroke::new(1.0, theme::ACCENT.gamma_multiply(0.7)) } else { egui::Stroke::NONE };
    egui::Frame::new().fill(visuals.widgets.inactive.weak_bg_fill).stroke(stroke).corner_radius(theme::RADIUS_SMALL).inner_margin(egui::Margin { left: 9, right: 3, top: 3, bottom: 3 })
}

/// How wide a chip with this text is: its margins, the text, the gap and the cross.
fn chip_width(ui: &egui::Ui, text: egui::text::LayoutJob) -> f32 {
    9.0 + ui.fonts_mut(|f| f.layout_job(text)).size().x + 4.0 + 18.0 + 3.0
}

/// In a row of chips: a new row when the next chip (`width`) doesn't fit in what is left of this one.
fn fits(ui: &mut egui::Ui, width: f32) {
    // a wrapping row's available_width is the whole row's: what is left is from the cursor to its right edge
    let left = ui.max_rect().right() - ui.cursor().min.x;
    if left < width && ui.cursor().min.x > ui.max_rect().min.x + 1.0 {
        ui.end_row();
    }
}

/// The small cross inside a chip: faint, lit when pointed at.
fn small_cross(ui: &mut egui::Ui) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::click());
    let colour = if response.hovered() {
        ui.painter().circle_filled(rect.center(), 8.0, ui.visuals().widgets.hovered.bg_fill);
        egui::Color32::WHITE
    } else {
        theme::MUTED
    };
    let (c, r) = (rect.center(), 3.5);
    let stroke = egui::Stroke::new(1.3, colour);
    ui.painter().line_segment([c + egui::vec2(-r, -r), c + egui::vec2(r, r)], stroke);
    ui.painter().line_segment([c + egui::vec2(-r, r), c + egui::vec2(r, -r)], stroke);
    response
}

/// Pictures of the box in its states, for looking at a change without running the app: .local/shots/<state>.png.
/// `cargo test --bin photo-map searchbar::shots -- --ignored` (it needs a GPU, so it doesn't run with the other tests).
#[cfg(test)]
mod shots {
    use super::*;
    use crate::query::Key;

    /// One state of the box, owning what Bar borrows.
    struct State {
        search: String,
        ideas: suggest::Suggest,
        tips_open: bool,
        hidden_open: bool,
        wanted: f32,
        filters: Vec<Filter>,
        filter_edit: Option<FilterEdit>,
        spots: HashMap<PathBuf, Option<usize>>,
        status: String,
        photos: usize,
        textures: Vec<(PathBuf, egui::TextureHandle)>,
        found: usize,
        at: Option<usize>,
        hidden: Vec<(String, usize)>,
        width: f32,
        focus: bool,
    }

    fn state() -> State {
        State {
            search: String::new(),
            ideas: suggest::Suggest::new(),
            tips_open: false,
            hidden_open: false,
            wanted: 100.0,
            filters: Vec::new(),
            filter_edit: None,
            spots: HashMap::new(),
            status: String::new(),
            photos: 0,
            textures: Vec::new(),
            found: 0,
            at: None,
            hidden: Vec::new(),
            width: 620.0,
            focus: false,
        }
    }

    fn filter(text: &str) -> Filter {
        query::filters(text).1.remove(0)
    }

    /// Draws `s` (`hover`: the mouse there, as a point in the window) and saves it as .local/shots/<name>.png.
    fn shot(name: &str, s: State, hover: Option<egui::Pos2>) {
        shot_then(name, s, |h| {
            if let Some(at) = hover {
                h.hover_at(at);
            }
        });
    }

    /// Draws `s`, does `then` to it (hover, click), and saves it as .local/shots/<name>.png.
    fn shot_then(name: &str, s: State, then: impl FnOnce(&mut egui_kittest::Harness<'_, State>)) {
        let mut harness = egui_kittest::Harness::builder().with_size(egui::vec2(s.width + 80.0, 440.0)).wgpu().build_ui_state(
            |ui, s: &mut State| {
                let ctx = ui.ctx().clone();
                theme::apply(&ctx);
                ui.painter().rect_filled(ctx.content_rect(), 0.0, theme::BACKGROUND); // the map's dark behind it
                if s.focus {
                    ctx.memory_mut(|m| m.request_focus(egui::Id::new("search-box")));
                }
                if s.textures.len() < s.photos {
                    // a made-up photo: a gradient
                    let k = s.textures.len();
                    let (w, h) = (40 + 16 * k, 30);
                    let pixels = (0..w * h).map(|i| egui::Color32::from_rgb((i % w * 5) as u8, (i / w * 7) as u8, 140)).collect();
                    let img = egui::ColorImage::new([w, h], pixels);
                    s.textures.push((PathBuf::from(format!("/photos/{k}.jpg")), ctx.load_texture(format!("photo {k}"), img, egui::TextureOptions::LINEAR)));
                }
                let mut actions = UiActions::default();
                let bar = Bar {
                    search: &mut s.search,
                    ideas: &mut s.ideas,
                    tips_open: &mut s.tips_open,
                    hidden_open: &mut s.hidden_open,
                    wanted: &mut s.wanted,
                    filters: &mut s.filters,
                    filter_edit: &mut s.filter_edit,
                    spots: &mut s.spots,
                    status: &s.status,
                    photos: &s.textures,
                    engine: Engine::Ready,
                    found: s.found,
                    at: s.at,
                    on_path: false,
                    most: 19_000.0,
                    all_marked: false,
                    hidden: s.hidden.clone(),
                    data: None,
                };
                show(&ctx, s.width, bar, &mut actions);
            },
            s,
        );
        harness.run_steps(3);
        then(&mut harness);
        harness.run_steps(8);
        let image = harness.render().expect("render");
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".local/shots");
        std::fs::create_dir_all(&dir).unwrap();
        image.save(dir.join(format!("{name}.png"))).unwrap();
    }

    #[test]
    #[ignore = "needs a GPU; writes pictures to look at"]
    fn shots() {
        let _turn = crate::shots::gpu_turn();
        shot("1-empty", state(), None);
        shot("2-typing", State { search: "street at".into(), focus: true, ..state() }, None);
        let results = || State { search: "shopping cart".into(), found: 100, at: Some(0), status: "shopping cart: similarity 0.12 to 0.25".into(), ..state() };
        shot("3-results", results(), None);
        shot("4-chips", State { filters: vec![filter("place:istanbul"), filter("NOT camera:pixel,redmi"), filter("year:2016..2018")], ..results() }, None);
        shot("4b-chip-hovered", State { filters: vec![filter("place:istanbul"), filter("NOT camera:pixel,redmi")], ..results() }, Some(egui::pos2(100.0, 71.0)));
        shot("5b-editing-a-not-chip", State { filters: vec![filter("place:istanbul")], filter_edit: Some(FilterEdit { at: 1, key: Key::Camera, not: true, value: "pixel".into() }), ..results() }, None);
        shot("5-editing-a-chip", State { filters: vec![filter("place:istanbul")], filter_edit: Some(FilterEdit { at: 1, key: Key::Date, not: false, value: String::new() }), ..results() }, None);
        shot("6-filters-only", State { search: String::new(), filters: vec![filter("year:2019")], status: "date:2019: 2,412 photos".into(), ..results() }, None);
        shot("7-photos", State { photos: 3, search: String::new(), filters: vec![filter("place:lviv")], ..results() }, None);
        shot("8-hidden", State { hidden: vec![("cars".into(), 120), ("like IMG_2041 +2".into(), 40)], hidden_open: true, ..results() }, None);
        shot("9-narrow", State { width: 300.0, filters: vec![filter("place:istanbul"), filter("NOT camera:pixel,redmi")], hidden: vec![("cars".into(), 120)], hidden_open: true, ..results() }, None);
        // the i at the box's right end, pointed at: how to search
        {
            use egui_kittest::kittest::Queryable;
            shot_then("11-filter-hovered", state(), |h| h.get_by_label("+ Filter").hover());
            shot_then("12-filter-menu", state(), |h| h.get_by_label("+ Filter").click());
            shot_then("12b-filter-menu-results", results(), |h| h.get_by_label("+ Filter").click());
        }
        // the i at the box's right end, clicked: how to search
        let click_i = |h: &mut egui_kittest::Harness<'_, State>| {
            let at = egui::pos2(40.0 + 620.0 - 8.0 - 10.0 - 13.0, 27.0);
            h.hover_at(at);
            h.run_steps(2);
            for pressed in [true, false] {
                h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
                h.run_steps(1);
            }
        };
        shot_then("10-help", State { width: 620.0, ..state() }, click_i);
        shot_then("10b-help-with-results", results(), click_i);
        shot_then("10c-help-narrow", State { width: 300.0, ..state() }, |h| {
            let at = egui::pos2(40.0 + 300.0 - 8.0 - 10.0 - 13.0, 27.0);
            h.hover_at(at);
            h.run_steps(2);
            for pressed in [true, false] {
                h.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
                h.run_steps(1);
            }
        });
    }
}
