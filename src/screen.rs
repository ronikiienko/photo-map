//! Where the app is shown: a window, or nothing at all (tests, PHOTO_MAP_HEADLESS), where the frames are drawn into a texture and the
//! mouse and keys are what a script (script.rs) says. The viewer asks this for its size, its scale and egui's input, and the rest of
//! it doesn't know which one it has.

use std::sync::Arc;
use std::time::Instant;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::window::{Fullscreen, Window};

pub enum Screen {
    Window { window: Arc<Window>, input: egui_winit::State },
    /// No window: its size in pixels, pixels per point, and egui's events made from the events the script sends.
    Headless { size: (u32, u32), ppp: f32, events: Vec<egui::Event>, pointer: egui::Pos2, started: Instant },
}

impl Screen {
    pub fn window(window: Arc<Window>, ctx: &egui::Context) -> Self {
        let input = egui_winit::State::new(ctx.clone(), ctx.viewport_id(), &window, None, None, None);
        Screen::Window { window, input }
    }

    pub fn headless(width: u32, height: u32, ppp: f32) -> Self {
        Screen::Headless { size: (width, height), ppp, events: Vec::new(), pointer: egui::Pos2::ZERO, started: Instant::now() }
    }

    pub fn is_headless(&self) -> bool {
        matches!(self, Screen::Headless { .. })
    }

    /// Ask for another frame (without a window, frames are drawn one after another anyway).
    pub fn redraw(&self) {
        if let Screen::Window { window, .. } = self {
            window.request_redraw();
        }
    }

    /// What threads call when they have something new to show.
    pub fn waker(&self) -> Arc<dyn Fn() + Send + Sync> {
        match self {
            Screen::Window { window, .. } => {
                let window = window.clone();
                Arc::new(move || window.request_redraw())
            }
            Screen::Headless { .. } => Arc::new(|| {}),
        }
    }

    /// Pixels per point.
    pub fn scale(&self) -> f32 {
        match self {
            Screen::Window { window, .. } => window.scale_factor() as f32,
            Screen::Headless { ppp, .. } => *ppp,
        }
    }

    /// The inside, in pixels.
    pub fn size(&self) -> (u32, u32) {
        match self {
            Screen::Window { window, .. } => {
                let s = window.inner_size();
                (s.width, s.height)
            }
            Screen::Headless { size, .. } => *size,
        }
    }

    /// Ask for a size in points (a window may not get it).
    pub fn request_size(&mut self, w: f64, h: f64) {
        match self {
            Screen::Window { window, .. } => {
                let _ = window.request_inner_size(winit::dpi::LogicalSize::new(w, h));
            }
            Screen::Headless { size, ppp, .. } => *size = ((w * *ppp as f64) as u32, (h * *ppp as f64) as u32),
        }
    }

    pub fn set_title(&self, title: &str) {
        if let Screen::Window { window, .. } = self {
            window.set_title(title);
        }
    }

    pub fn toggle_fullscreen(&self) {
        if let Screen::Window { window, .. } = self {
            let on = window.fullscreen().is_none();
            window.set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
        }
    }

    pub fn pre_present(&self) {
        if let Screen::Window { window, .. } = self {
            window.pre_present_notify();
        }
    }

    /// Give a window event to egui. True when egui took it (the map should not act on it too).
    pub fn on_event(&mut self, ctx: &egui::Context, event: &WindowEvent) -> bool {
        match self {
            Screen::Window { window, input } => input.on_window_event(window, event).consumed,
            Screen::Headless { ppp, events, pointer, .. } => {
                // what egui_winit makes of these, for the ones a script sends
                let modifiers = egui::Modifiers::NONE;
                match event {
                    WindowEvent::CursorMoved { position, .. } => {
                        *pointer = egui::pos2(position.x as f32 / *ppp, position.y as f32 / *ppp);
                        events.push(egui::Event::PointerMoved(*pointer));
                        ctx.egui_is_using_pointer()
                    }
                    WindowEvent::CursorLeft { .. } => {
                        events.push(egui::Event::PointerGone);
                        false
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        let button = match button {
                            MouseButton::Left => egui::PointerButton::Primary,
                            MouseButton::Right => egui::PointerButton::Secondary,
                            MouseButton::Middle => egui::PointerButton::Middle,
                            _ => return false,
                        };
                        events.push(egui::Event::PointerButton { pos: *pointer, button, pressed: *state == ElementState::Pressed, modifiers });
                        ctx.egui_wants_pointer_input()
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        let (unit, delta) = match delta {
                            MouseScrollDelta::LineDelta(x, y) => (egui::MouseWheelUnit::Line, egui::vec2(*x, *y)),
                            MouseScrollDelta::PixelDelta(p) => (egui::MouseWheelUnit::Point, egui::vec2(p.x as f32, p.y as f32) / *ppp),
                        };
                        events.push(egui::Event::MouseWheel { unit, delta, modifiers, phase: egui::TouchPhase::Move });
                        ctx.egui_wants_pointer_input()
                    }
                    _ => false,
                }
            }
        }
    }

    /// egui's input for the next frame, with `more` events (letters and keys a script types) after the screen's own.
    pub fn egui_input(&mut self, more: &mut Vec<egui::Event>) -> egui::RawInput {
        let mut raw = match self {
            Screen::Window { window, input } => input.take_egui_input(window),
            Screen::Headless { size, ppp, events, started, .. } => {
                let mut raw = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size.0 as f32, size.1 as f32) / *ppp)),
                    time: Some(started.elapsed().as_secs_f64()),
                    focused: true,
                    events: std::mem::take(events),
                    ..Default::default()
                };
                raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(*ppp);
                raw
            }
        };
        raw.events.append(more);
        raw
    }

    /// What egui wants of the platform (the cursor's shape, copied text); nothing to do without a window.
    pub fn platform_output(&mut self, output: egui::PlatformOutput) {
        if let Screen::Window { window, input } = self {
            input.handle_platform_output(window, output);
        }
    }
}
