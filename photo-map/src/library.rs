//! The maps the app knows: every bundle in the project's maps/ folder. A drop-down to switch between them, and the
//! "New map" window: a folder picked with the system's own dialog, `make` run on a thread, its progress shown live.

use crate::make;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};
use std::thread::JoinHandle;

pub struct Entry {
    pub path: PathBuf,
    pub name: String,
    pub photos: usize,
}

pub enum Action {
    /// Open this bundle.
    Open(PathBuf),
    /// A map is being made: let go of the open one, its GPU memory is needed for embedding.
    Making,
}

pub struct Library {
    /// The EmbeddingTest folder (Python environment for embedding) and its maps/ folder.
    project: PathBuf,
    dir: PathBuf,
    pub maps: Vec<Entry>,
    window_open: bool,
    folder: Option<PathBuf>,
    name: String,
    picking: Option<Receiver<Option<PathBuf>>>,
    making: Option<(PathBuf, JoinHandle<std::io::Result<()>>)>,
    error: Option<String>,
}

impl Library {
    pub fn new(project: &Path) -> Self {
        let dir = project.join("maps");
        let mut lib = Self {
            project: project.to_path_buf(),
            dir,
            maps: Vec::new(),
            window_open: false,
            folder: None,
            name: String::new(),
            picking: None,
            making: None,
            error: None,
        };
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

    pub fn making(&self) -> bool {
        self.making.is_some()
    }

    /// The map drop-down and the "New map" button, in the top bar.
    pub fn bar(&mut self, ui: &mut egui::Ui, current: Option<&Path>) -> Option<Action> {
        let mut action = None;
        let shown = current.and_then(|c| self.maps.iter().find(|m| m.path == c)).map_or("no map".into(), |m| m.name.clone());
        ui.add_enabled_ui(!self.making(), |ui| {
            egui::ComboBox::from_id_salt("maps").selected_text(shown).width(180.0).show_ui(ui, |ui| {
                for m in &self.maps {
                    let label = format!("{}  ·  {} photos", m.name, m.photos);
                    if ui.selectable_label(Some(m.path.as_path()) == current, label).clicked() && Some(m.path.as_path()) != current {
                        action = Some(Action::Open(m.path.clone()));
                    }
                }
            });
        });
        if ui.button("New map…").clicked() {
            self.window_open = true;
        }
        action
    }

    /// The "New map" window, while it's open.
    pub fn window(&mut self, ctx: &egui::Context) -> Option<Action> {
        let mut action = None;
        if let Some(rx) = &self.picking {
            if let Ok(picked) = rx.try_recv() {
                self.picking = None;
                if let Some(folder) = picked {
                    if self.name.is_empty() {
                        self.name = folder.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    }
                    self.folder = Some(folder);
                }
            }
        }
        if self.making.as_ref().is_some_and(|(_, job)| job.is_finished()) {
            let (bundle, job) = self.making.take().unwrap();
            match job.join() {
                Ok(Ok(())) => {
                    self.refresh();
                    self.window_open = false;
                    self.folder = None;
                    self.name.clear();
                    action = Some(Action::Open(bundle));
                }
                Ok(Err(e)) => self.error = Some(format!("failed: {e}")),
                Err(_) => self.error = Some("failed: the map maker crashed".into()),
            }
        }
        if !self.window_open {
            return action;
        }

        let mut open = true;
        egui::Window::new("New photo map").collapsible(false).resizable(false).open(&mut open).show(ctx, |ui| {
            if let Some((bundle, _)) = &self.making {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("making {}", bundle.file_stem().unwrap_or_default().to_string_lossy()));
                });
                ui.separator();
                for line in make::recent(14) {
                    ui.monospace(line);
                }
                return;
            }
            egui::Grid::new("new map").num_columns(2).spacing([10.0, 8.0]).show(ui, |ui| {
                ui.label("Photos");
                ui.horizontal(|ui| {
                    let shown = self.folder.as_ref().map_or("no folder chosen".into(), |f| f.display().to_string());
                    ui.label(shown);
                    if ui.add_enabled(self.picking.is_none(), egui::Button::new("Choose folder…")).clicked() {
                        let (tx, rx) = channel();
                        std::thread::spawn(move || tx.send(rfd::FileDialog::new().set_title("Folder of photos").pick_folder()));
                        self.picking = Some(rx);
                    }
                });
                ui.end_row();
                ui.label("Name");
                ui.text_edit_singleline(&mut self.name);
                ui.end_row();
            });
            let file_name: String = self.name.trim().chars().map(|c| if c.is_alphanumeric() || "-_ ".contains(c) { c } else { '_' }).collect();
            let bundle = self.dir.join(format!("{file_name}.photomap"));
            let problem = if self.folder.is_none() {
                Some("choose a folder of photos")
            } else if file_name.is_empty() {
                Some("give the map a name")
            } else if bundle.join("manifest.json").exists() {
                Some("a map with this name already exists")
            } else {
                None
            };
            if let Some(p) = problem {
                ui.weak(p);
            }
            if let Some(e) = &self.error {
                ui.colored_label(egui::Color32::from_rgb(230, 110, 100), e);
            }
            ui.weak("Embedding takes about 1 minute per 700 photos on the GPU; a stopped run continues where it left off.");
            if ui.add_enabled(problem.is_none(), egui::Button::new("Create")).clicked() {
                self.error = None;
                let (project, folder, target) = (self.project.clone(), self.folder.clone().unwrap(), bundle.clone());
                let job = std::thread::spawn(move || make::run(&project, &target, &[folder]));
                self.making = Some((bundle, job));
                action = Some(Action::Making);
            }
        });
        if !open && self.making.is_none() {
            self.window_open = false;
        }
        action
    }
}
