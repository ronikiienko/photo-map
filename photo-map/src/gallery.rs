//! The matches of a search together in a window, best first, outside their places on the map: a grid of thumbnails. Only the
//! rows in view are drawn, and only their pictures are read (from the bundle's originals, on background threads).

use crate::photo;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

/// A thumbnail's cell (points) and the longer side of the picture read for it (pixels).
const CELL: f32 = 128.0;
const GAP: f32 = 6.0;
const READ_PX: u32 = 256;
/// Pictures kept; when there are more, the ones drawn longest ago go.
const KEEP: usize = 500;

pub struct Gallery {
    pub open: bool,
    /// Per photo: its thumbnail and the frame it was last drawn in.
    textures: HashMap<usize, (egui::TextureHandle, u64)>,
    /// Photos whose picture is being read (or couldn't be).
    asked: HashSet<usize>,
    frame: u64,
    tx: Sender<(usize, egui::ColorImage)>,
    rx: Receiver<(usize, egui::ColorImage)>,
}

impl Gallery {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self { open: false, textures: HashMap::new(), asked: HashSet::new(), frame: 0, tx, rx }
    }

    /// Forget every picture (another map was opened).
    pub fn clear(&mut self) {
        self.textures.clear();
        self.asked.clear();
        while self.rx.try_recv().is_ok() {}
    }

    /// The window. `order`: the photos to show; `current`: the position in it to outline; `file`: a photo's file;
    /// `score`: its similarity to the search, if it has one; `caption`: its caption (asked only for the photo under the
    /// mouse). Returns the position in `order` that was clicked.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        order: &[usize],
        current: Option<usize>,
        file: &dyn Fn(usize) -> PathBuf,
        score: &dyn Fn(usize) -> Option<f32>,
        caption: &dyn Fn(usize) -> String,
    ) -> Option<usize> {
        self.frame += 1;
        while let Ok((i, image)) = self.rx.try_recv() {
            let texture = ctx.load_texture(format!("gallery {i}"), image, egui::TextureOptions::LINEAR);
            self.textures.insert(i, (texture, self.frame));
        }
        let mut clicked = None;
        let mut open = self.open;
        egui::Window::new(format!("matches ({})", order.len()))
            .id(egui::Id::new("gallery"))
            .open(&mut open)
            .default_pos([ctx.content_rect().right() - 580.0, 60.0])
            .default_size([560.0, 620.0])
            .resizable(true)
            .show(ctx, |ui| {
                if order.is_empty() {
                    ui.weak("no matches: search first");
                    return;
                }
                let cols = (((ui.available_width() + GAP) / (CELL + GAP)).floor() as usize).max(1);
                let rows = order.len().div_ceil(cols);
                let row = CELL + GAP;
                egui::ScrollArea::vertical().auto_shrink([false, false]).show_viewport(ui, |ui, view| {
                    ui.set_height(rows as f32 * row);
                    let origin = ui.min_rect().min;
                    let first = (view.min.y / row).floor().max(0.0) as usize;
                    let last = ((view.max.y / row).ceil() as usize + 1).min(rows);
                    for r in first..last {
                        for c in 0..cols {
                            let k = r * cols + c;
                            let Some(&photo) = order.get(k) else { break };
                            let cell = egui::Rect::from_min_size(origin + egui::vec2(c as f32 * (CELL + GAP), r as f32 * row), egui::vec2(CELL, CELL));
                            let response = ui.interact(cell, ui.id().with(("cell", k)), egui::Sense::click());
                            let painter = ui.painter();
                            painter.rect_filled(cell, 4.0, egui::Color32::from_gray(28));
                            match self.textures.get_mut(&photo) {
                                Some((texture, seen)) => {
                                    *seen = self.frame;
                                    let size = texture.size_vec2();
                                    let fit = egui::Rect::from_center_size(cell.center(), size * (CELL / size.max_elem()));
                                    painter.image(texture.id(), fit, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
                                }
                                None if self.asked.insert(photo) => {
                                    let (tx, ctx, path) = (self.tx.clone(), ctx.clone(), file(photo));
                                    rayon::spawn(move || {
                                        if let Some(image) = photo::load(&path) {
                                            let small = photo::fit(&image, READ_PX, false);
                                            let color = egui::ColorImage::from_rgba_unmultiplied([small.width() as usize, small.height() as usize], small.as_raw());
                                            let _ = tx.send((photo, color));
                                            ctx.request_repaint();
                                        }
                                    });
                                }
                                None => {}
                            }
                            if let (Some(sim), true) = (score(photo), response.hovered() || current == Some(k)) {
                                let at = cell.left_bottom() + egui::vec2(5.0, -4.0);
                                let font = egui::FontId::monospace(13.0);
                                painter.text(at + egui::vec2(1.0, 1.0), egui::Align2::LEFT_BOTTOM, format!("{sim:.2}"), font.clone(), egui::Color32::BLACK);
                                painter.text(at, egui::Align2::LEFT_BOTTOM, format!("{sim:.2}"), font, egui::Color32::WHITE);
                            }
                            if current == Some(k) {
                                painter.rect_stroke(cell, 4.0, egui::Stroke::new(2.0, egui::Color32::WHITE), egui::StrokeKind::Inside);
                            }
                            if response.hovered() {
                                response.clone().on_hover_text(caption(photo));
                            }
                            if response.clicked() {
                                clicked = Some(k);
                            }
                        }
                    }
                });
            });
        self.open = open;
        if self.textures.len() > KEEP {
            let mut ages: Vec<u64> = self.textures.values().map(|t| t.1).collect();
            ages.sort_unstable();
            let oldest_kept = ages[ages.len() - KEEP];
            let before: Vec<usize> = self.textures.keys().copied().collect();
            self.textures.retain(|_, t| t.1 >= oldest_kept);
            for photo in before {
                if !self.textures.contains_key(&photo) {
                    self.asked.remove(&photo);
                }
            }
        }
        clicked
    }
}
