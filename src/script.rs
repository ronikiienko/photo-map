//! Scripted screenshots of the whole app, for looking at its UI without someone at the window: `PHOTO_MAP_SCRIPT=steps.txt`
//! runs the steps below one after another, as if typed and clicked, and `shot` saves the frame as .local/appshots/<name>.png.
//! Positions are in points (the UI's own units, top left 0 0). Blank lines and `#` comments are skipped.
//!
//!     size 1200 780          the window's inner size
//!     open takeout-tagged    open this map (from the maps folder), and wait until it is open
//!     wait 30                let 30 frames pass
//!     move 600 400           the mouse moves here
//!     click 600 400 [right]  a click here (left unless right / middle)
//!     scroll -3              the wheel turns (lines; negative: towards you)
//!     focus                  the search box takes the keyboard
//!     type shopping cart     these letters are typed
//!     key Enter              this key is pressed (egui's names: Enter, Tab, Escape, ArrowDown, A ...)
//!     appkey KeyH            this key goes to the map's own keys (winit's names: KeyH, Escape, F11 ...)
//!     searched               wait until the search running now has its answer
//!     shot search-results    save the next frame
//!     maps                   back to the list of maps
//!     quit                   close the app (also at the end of the steps)

use std::collections::VecDeque;
use std::path::PathBuf;

pub struct Script {
    steps: VecDeque<String>,
    /// Frames left to wait before the next step.
    wait: u32,
    /// What the step being waited on needs before the next one: a map open, a search answered.
    until: Option<Until>,
    pub dir: PathBuf,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Until {
    Open,
    Searched,
}

/// One step, read.
pub enum Step {
    Size(f64, f64),
    Open(String),
    Move(f32, f32),
    Click(f32, f32, winit::event::MouseButton),
    Scroll(f32),
    Focus,
    Type(String),
    Key(egui::Key),
    AppKey(winit::keyboard::KeyCode),
    Shot(String),
    Maps,
    Quit,
}

impl Script {
    /// The script named by PHOTO_MAP_SCRIPT, if any.
    pub fn from_env() -> Option<Script> {
        let path = std::env::var_os("PHOTO_MAP_SCRIPT")?;
        let text = std::fs::read_to_string(&path).map_err(|e| eprintln!("script {}: {e}", path.to_string_lossy())).ok()?;
        let steps = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')).map(String::from).collect();
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".local/appshots");
        let _ = std::fs::create_dir_all(&dir);
        Some(Script { steps, wait: 10, until: None, dir })
    }

    /// The step to take this frame, if its time has come. `open` / `searched`: whether what is waited for is there.
    pub fn next(&mut self, open: bool, searched: bool) -> Option<Step> {
        match self.until {
            Some(Until::Open) if !open => return None,
            Some(Until::Searched) if !searched => return None,
            Some(_) => {
                self.until = None;
                self.wait = 60; // and a moment for what came to be drawn (the camera flies there)
            }
            None => {}
        }
        if self.wait > 0 {
            self.wait -= 1;
            return None;
        }
        let line = self.steps.pop_front().unwrap_or_else(|| "quit".into());
        eprintln!("script: {line}");
        let (word, rest) = line.split_once(' ').unwrap_or((&line, ""));
        let nums: Vec<f32> = rest.split_whitespace().filter_map(|n| n.parse().ok()).collect();
        let num = |k: usize| nums.get(k).copied().unwrap_or(0.0);
        self.wait = 3; // a few frames after every step, so that what it did is drawn
        let step = match word {
            "size" => Step::Size(num(0) as f64, num(1) as f64),
            "open" => {
                self.until = Some(Until::Open);
                Step::Open(rest.to_string())
            }
            "wait" => {
                self.wait = num(0) as u32;
                return None;
            }
            "move" => Step::Move(num(0), num(1)),
            "click" => {
                use winit::event::MouseButton;
                let button = if rest.contains("right") { MouseButton::Right } else if rest.contains("middle") { MouseButton::Middle } else { MouseButton::Left };
                Step::Click(num(0), num(1), button)
            }
            "scroll" => Step::Scroll(num(0)),
            "focus" => Step::Focus,
            "type" => Step::Type(rest.to_string()),
            "key" => match egui::Key::from_name(rest) {
                Some(k) => Step::Key(k),
                None => {
                    eprintln!("script: no key {rest}");
                    return None;
                }
            },
            "appkey" => match app_key(rest) {
                Some(k) => Step::AppKey(k),
                None => {
                    eprintln!("script: no app key {rest}");
                    return None;
                }
            },
            "searched" => {
                self.until = Some(Until::Searched);
                return None;
            }
            "shot" => Step::Shot(rest.to_string()),
            "maps" => Step::Maps,
            "quit" => Step::Quit,
            _ => {
                eprintln!("script: what is {word}?");
                return None;
            }
        };
        Some(step)
    }
}

/// The winit keys the map listens to, by name.
fn app_key(name: &str) -> Option<winit::keyboard::KeyCode> {
    use winit::keyboard::KeyCode::*;
    Some(match name {
        "Escape" => Escape,
        "F11" => F11,
        "Space" => Space,
        "Tab" => Tab,
        "Enter" => Enter,
        "KeyH" => KeyH,
        "KeyL" => KeyL,
        "KeyN" => KeyN,
        "KeyP" => KeyP,
        "KeyF" => KeyF,
        "KeyR" => KeyR,
        "KeyW" => KeyW,
        "KeyA" => KeyA,
        "KeyS" => KeyS,
        "KeyD" => KeyD,
        "Equal" => Equal,
        "Minus" => Minus,
        "BracketLeft" => BracketLeft,
        "BracketRight" => BracketRight,
        _ => return None,
    })
}

/// The frame in `texture` (the window's, in its `format`) as a png at `path`. The texture must allow COPY_SRC.
pub fn save_frame(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, format: wgpu::TextureFormat, path: &std::path::Path) {
    let (w, h) = (texture.width(), texture.height());
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("shot"), size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let Ok(data) = slice.get_mapped_range() else {
        eprintln!("script: can't read the frame back");
        return;
    };
    let bgra = matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let line = &data[(y * row) as usize..(y * row + w * 4) as usize];
        for p in line.chunks_exact(4) {
            if bgra {
                pixels.extend_from_slice(&[p[2], p[1], p[0], 255]);
            } else {
                pixels.extend_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
    }
    drop(data);
    buffer.unmap();
    match image::RgbaImage::from_raw(w, h, pixels).map(|img| img.save(path)) {
        Some(Ok(())) => eprintln!("script: saved {}", path.display()),
        _ => eprintln!("script: can't save {}", path.display()),
    }
}
