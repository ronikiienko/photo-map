//! The maps the app knows: every bundle in the project's maps/ folder (made in Python, see data.rs), in a
//! drop-down to switch between them.

use std::path::{Path, PathBuf};

pub struct Entry {
    pub path: PathBuf,
    pub name: String,
    pub photos: usize,
}

pub struct Library {
    dir: PathBuf,
    pub maps: Vec<Entry>,
}

impl Library {
    pub fn new(dir: PathBuf) -> Self {
        let mut lib = Self { dir, maps: Vec::new() };
        lib.refresh();
        lib
    }

    /// Every finished bundle (one with a manifest) in maps/, by name.
    fn refresh(&mut self) {
        let entries = std::fs::read_dir(&self.dir).into_iter().flatten().filter_map(Result::ok);
        self.maps = entries
            .filter_map(|e| {
                let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(e.path().join("manifest.json")).ok()?).ok()?;
                Some(Entry { path: e.path(), name: manifest["name"].as_str()?.to_string(), photos: manifest["photos"].as_u64()? as usize })
            })
            .collect();
        self.maps.sort_by(|a, b| a.name.cmp(&b.name));
    }

    /// The map opened last time, else the first one.
    pub fn last(&self) -> Option<PathBuf> {
        let last = std::fs::read_to_string(self.dir.join(".last")).ok();
        let found = last.and_then(|name| self.maps.iter().find(|m| m.path.file_name().is_some_and(|f| f == name.trim())));
        found.or(self.maps.first()).map(|m| m.path.clone())
    }

    pub fn remember(&self, bundle: &Path) {
        if let Some(name) = bundle.file_name() {
            let _ = std::fs::write(self.dir.join(".last"), name.to_string_lossy().as_bytes());
        }
    }

    /// The map drop-down, in the top bar (looks at maps/ again each time it's opened); a map picked -> its path.
    pub fn bar(&mut self, ui: &mut egui::Ui, current: Option<&Path>) -> Option<PathBuf> {
        let mut picked = None;
        let shown = current.and_then(|c| self.maps.iter().find(|m| m.path == c)).map_or("no map".into(), |m| m.name.clone());
        let combo = egui::ComboBox::from_id_salt("maps").selected_text(shown).width(180.0).show_ui(ui, |ui| {
            for m in &self.maps {
                let label = format!("{}  ·  {} photos", m.name, m.photos);
                if ui.selectable_label(Some(m.path.as_path()) == current, label).clicked() && Some(m.path.as_path()) != current {
                    picked = Some(m.path.clone());
                }
            }
        });
        if combo.response.clicked() {
            self.refresh();
        }
        picked
    }
}
