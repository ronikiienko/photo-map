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
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        for &(v, text, tip) in options {
            ui.selectable_value(value, v, text).on_hover_text(tip);
        }
    });
    *value != before
}
