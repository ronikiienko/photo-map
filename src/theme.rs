//! The look of the app, in one place: palette, text sizes, widget styling, and the few small pieces
//! (card, key chip, status dot, segmented control) the panels are built from.
//!
//! Rules the layout follows:
//! - the map is the content; every panel is a floating card that hugs an edge (top centre, bottom corners) and never covers the middle
//! - one 8 px grid for spacing (4 inside a control, 8 between controls, 12 inside a card), one corner radius per kind of thing
//! - one accent colour for what is selected or active; everything else is grey on a dark translucent card
//! - text: 14 body, 12 for hints and numbers, 17 for card titles; secondary text is always the muted colour

use egui::{Color32, CornerRadius, FontFamily, FontId, Margin, RichText, Stroke, TextStyle, Ui, vec2};

/// The page behind the cards when there is no map to show.
pub const BACKGROUND: Color32 = Color32::from_rgb(21, 23, 28);
pub const ACCENT: Color32 = Color32::from_rgb(124, 164, 255);
pub const OK: Color32 = Color32::from_rgb(110, 205, 140);
pub const TEXT: Color32 = Color32::from_rgb(232, 235, 242);
pub const MUTED: Color32 = Color32::from_rgb(140, 147, 163);
/// Cards sit on the map, so they are a little see-through.
pub const CARD: Color32 = Color32::from_rgba_premultiplied(17, 19, 25, 226);
pub const CARD_STROKE: Color32 = Color32::from_rgba_premultiplied(40, 44, 54, 255);
const CONTROL: Color32 = Color32::from_rgb(38, 42, 52);
const CONTROL_HOVER: Color32 = Color32::from_rgb(52, 58, 73);

/// A button that deletes something.
pub const DANGER: Color32 = Color32::from_rgb(150, 60, 55);

pub const GAP: f32 = 8.0;
pub const RADIUS: u8 = 10; // cards and windows
pub const RADIUS_SMALL: u8 = 7; // controls

pub fn apply(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.panel_fill = CARD;
    v.window_fill = Color32::from_rgb(20, 22, 29);
    v.window_stroke = Stroke::new(1.0, CARD_STROKE);
    v.window_corner_radius = CornerRadius::from(RADIUS);
    v.menu_corner_radius = CornerRadius::from(RADIUS);
    v.window_shadow = egui::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(110) };
    v.popup_shadow = v.window_shadow;
    v.extreme_bg_color = Color32::from_rgb(12, 13, 18); // text fields, slider rails
    v.faint_bg_color = Color32::from_rgb(28, 31, 39);
    v.override_text_color = Some(TEXT);
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = ACCENT.gamma_multiply(0.38);
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    let r = CornerRadius::from(RADIUS_SMALL);
    let w = &mut v.widgets;
    w.noninteractive.bg_stroke = Stroke::new(1.0, CARD_STROKE);
    w.noninteractive.fg_stroke = Stroke::new(1.0, MUTED);
    for (state, fill) in [(&mut w.inactive, CONTROL), (&mut w.hovered, CONTROL_HOVER), (&mut w.active, ACCENT.gamma_multiply(0.55)), (&mut w.open, CONTROL_HOVER)] {
        state.bg_fill = fill;
        state.weak_bg_fill = fill;
        state.corner_radius = r;
        state.expansion = 0.0;
    }
    w.noninteractive.corner_radius = r;
    w.inactive.bg_stroke = Stroke::NONE;
    w.hovered.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
    w.active.bg_stroke = Stroke::new(1.0, ACCENT);
    w.open.bg_stroke = Stroke::new(1.0, ACCENT.gamma_multiply(0.6));
    w.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    w.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    w.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    ctx.set_visuals(v);

    ctx.global_style_mut(|s| {
        s.spacing.item_spacing = vec2(GAP, 6.0);
        s.spacing.button_padding = vec2(10.0, 5.0);
        s.spacing.interact_size.y = 28.0;
        s.spacing.window_margin = Margin::same(12);
        s.spacing.menu_margin = Margin::same(12);
        s.spacing.slider_width = 160.0;
        s.spacing.combo_height = 300.0;
        s.text_styles.insert(TextStyle::Heading, FontId::new(17.0, FontFamily::Proportional));
        s.text_styles.insert(TextStyle::Body, FontId::new(14.0, FontFamily::Proportional));
        s.text_styles.insert(TextStyle::Button, FontId::new(14.0, FontFamily::Proportional));
        s.text_styles.insert(TextStyle::Small, FontId::new(12.0, FontFamily::Proportional));
        s.text_styles.insert(TextStyle::Monospace, FontId::new(12.0, FontFamily::Monospace));
    });
}

/// A floating panel on the map.
pub fn card() -> egui::Frame {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0, CARD_STROKE))
        .corner_radius(CornerRadius::from(RADIUS))
        .inner_margin(Margin::same(12))
        .shadow(egui::Shadow { offset: [0, 4], blur: 18, spread: 0, color: Color32::from_black_alpha(90) })
}

/// Secondary text.
pub fn muted(text: impl Into<String>) -> RichText {
    RichText::new(text).color(MUTED).size(12.0)
}

/// A keyboard key, drawn as a little cap, for the hints.
pub fn key(ui: &mut Ui, text: &str) {
    egui::Frame::new()
        .fill(CONTROL)
        .stroke(Stroke::new(1.0, CARD_STROKE))
        .corner_radius(CornerRadius::from(5u8))
        .inner_margin(Margin::symmetric(6, 1))
        .show(ui, |ui| {
            ui.label(RichText::new(text).monospace().size(11.5).color(TEXT));
        });
}

/// A small filled circle, the height of the text, for a status.
pub fn dot(ui: &mut Ui, colour: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(8.0, 8.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 4.0, colour);
}

/// Pick one of several values, side by side as one block. Returns true when the pick changed.
pub fn segmented<T: PartialEq + Copy>(ui: &mut Ui, value: &mut T, options: &[(T, &str, &str)]) -> bool {
    let before = *value;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.spacing_mut().button_padding = vec2(12.0, 5.0); // room round the text, so the chosen one doesn't hug it
        for &(v, text, tip) in options {
            ui.selectable_value(value, v, text).on_hover_text(tip);
        }
    });
    *value != before
}

/// A button in a colour of its own (the main action, a delete). egui's `Button::fill` is one colour for every state, so it never
/// shows hover or press; this sets all three.
pub fn tinted(ui: &mut Ui, colour: Color32, enabled: bool, button: egui::Button) -> egui::Response {
    ui.scope(|ui| {
        let w = &mut ui.visuals_mut().widgets;
        for (state, fill) in [(&mut w.inactive, colour), (&mut w.hovered, colour.lerp_to_gamma(Color32::WHITE, 0.16)), (&mut w.active, colour.lerp_to_gamma(Color32::WHITE, 0.3))] {
            state.bg_fill = fill;
            state.weak_bg_fill = fill;
        }
        ui.add_enabled(enabled, button)
    })
    .inner
}

/// A close cross that lights up when the pointer is on it and again when it is pressed.
pub fn close_button(ui: &mut Ui) -> bool {
    let (rect, response) = ui.allocate_exact_size(vec2(26.0, 26.0), egui::Sense::click());
    let (fill, cross) = if response.is_pointer_button_down_on() {
        (ACCENT.gamma_multiply(0.5), Color32::WHITE)
    } else if response.hovered() {
        (CONTROL_HOVER, Color32::WHITE)
    } else {
        (Color32::TRANSPARENT, MUTED)
    };
    ui.painter().rect_filled(rect, CornerRadius::from(RADIUS_SMALL), fill);
    let c = rect.center();
    let stroke = Stroke::new(1.6, cross);
    ui.painter().line_segment([c + vec2(-5.0, -5.0), c + vec2(5.0, 5.0)], stroke);
    ui.painter().line_segment([c + vec2(5.0, -5.0), c + vec2(-5.0, 5.0)], stroke);
    response.on_hover_text("close").clicked()
}

/// The top of a window that has no title bar of egui's: the title and a close button. True when it was closed.
pub fn window_header(ui: &mut Ui, title: &str) -> bool {
    let mut closed = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).size(17.0));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            closed = close_button(ui);
        });
    });
    ui.add_space(4.0);
    closed
}

/// A dialog over a dimmed page that blocks the page behind it (egui's `Modal`): it can't be covered by a click outside it.
/// The caller says what closes it: `should_close` on the response is true for a click on the backdrop or Esc.
pub fn modal<T>(ctx: &egui::Context, id: &str, width: f32, title: Option<&str>, content: impl FnOnce(&mut Ui) -> T) -> egui::ModalResponse<T> {
    let frame = egui::Frame::new()
        .fill(Color32::from_rgb(20, 22, 29))
        .stroke(Stroke::new(1.0, CARD_STROKE))
        .corner_radius(CornerRadius::from(RADIUS))
        .inner_margin(Margin::same(16))
        .shadow(egui::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(110) });
    egui::Modal::new(egui::Id::new(id)).frame(frame).backdrop_color(Color32::from_black_alpha(130)).show(ctx, |ui| {
        ui.set_width(width);
        if let Some(title) = title {
            ui.label(RichText::new(title).size(17.0));
            ui.add_space(6.0);
        }
        content(ui)
    })
}
