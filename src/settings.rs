//! What the user chose that outlives the app: kept in settings.json in the data folder (`crate::data_root`).

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// Videos play without sound (one setting for all).
    pub muted: bool,
    /// How fast the wheel, a trackpad pinch and the zoom keys zoom: one of `ZOOM_SPEEDS`.
    pub zoom_speed: f32,
    /// How big the interface is drawn, over the system's own scale: one of `UI_SIZES`.
    pub ui_size: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self { muted: false, zoom_speed: 1.0, ui_size: 1.0 }
    }
}

/// The choices the Options menu offers: (value, label, tip).
pub const ZOOM_SPEEDS: [(f32, &str, &str); 3] = [(0.5, "Slow", "half as fast as Normal"), (1.0, "Normal", "a wheel notch zooms about 1.8x"), (2.0, "Fast", "twice as fast as Normal")];
pub const UI_SIZES: [(f32, &str, &str); 4] = [(0.8, "80%", "smaller text and buttons"), (1.0, "100%", "the system's own size"), (1.25, "125%", "bigger text and buttons"), (1.5, "150%", "bigger still")];

fn file() -> PathBuf {
    crate::data_root().join("settings.json")
}

impl Settings {
    /// The kept settings; a value not among the choices (an edited file) falls back to its default. Tests get the defaults, so
    /// their pictures don't depend on what was chosen in the app.
    pub fn load() -> Self {
        if cfg!(test) {
            return Self::default();
        }
        let mut s: Self = std::fs::read(file()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        let d = Self::default();
        if !ZOOM_SPEEDS.iter().any(|o| o.0 == s.zoom_speed) {
            s.zoom_speed = d.zoom_speed;
        }
        if !UI_SIZES.iter().any(|o| o.0 == s.ui_size) {
            s.ui_size = d.ui_size;
        }
        s
    }

    pub fn save(&self) {
        if cfg!(test) {
            return;
        }
        let file = file();
        if let Some(dir) = file.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(file, serde_json::to_vec_pretty(self).unwrap_or_default());
    }
}
