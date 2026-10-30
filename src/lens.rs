//! The lens: while Space is held, a round magnifier follows the mouse and shows the map under it closer up. It grows out of the
//! mouse when it opens and shrinks back into it when it closes, glides a little behind the mouse, and the wheel sets how much it
//! magnifies (the change eases in). The map is drawn a second time with the lens's own camera, clipped to the circle (map.wgsl),
//! and the photos inside it are streamed at the sharpness the lens shows them at, before the rest.

use glam::Vec2;

/// The lens's radius when open, in points (times the screen's pixels per point).
pub const RADIUS: f32 = 260.0;
/// How long it takes to open or close, seconds.
const OPEN_SECONDS: f32 = 0.16;
/// How much it magnifies at first, and the range the wheel can set.
const MAG_START: f32 = 4.0;
const MAG_RANGE: (f32, f32) = (1.5, 24.0);
/// How fast it catches up with the mouse, and the magnification with the wheel (per second; higher is quicker).
const FOLLOW: f32 = 28.0;
const EASE_MAG: f32 = 14.0;

pub struct Lens {
    /// Space is down.
    pub held: bool,
    /// 0 (closed) ... 1 (open), moving at a steady pace; `shown` eases it.
    open: f32,
    /// Where its centre is on screen, px (behind the mouse while it moves).
    at: Vec2,
    mag: f32,
    target: f32,
    /// Seconds since the magnification was last changed (its label shows for a moment).
    since_change: f32,
}

/// What the lens shows this frame.
#[derive(Clone, Copy)]
pub struct LensView {
    /// The lens's camera: the world point it would put at the screen's middle, and screen px per world unit.
    pub center: Vec2,
    pub scale: f32,
    /// Its centre on screen and its radius, px.
    pub at: Vec2,
    pub radius: f32,
    /// How open it is, 0 ... 1 (eased).
    pub shown: f32,
}

impl LensView {
    pub fn to_world(&self, screen: Vec2, viewport: Vec2) -> Vec2 {
        self.center + (screen - viewport * 0.5) / self.scale
    }

    pub fn contains(&self, screen: Vec2) -> bool {
        screen.distance(self.at) <= self.radius
    }

    /// Does a rectangle (screen px) touch the circle?
    pub fn touches(&self, min: Vec2, max: Vec2) -> bool {
        self.at.clamp(min, max).distance(self.at) <= self.radius
    }
}

impl Default for Lens {
    fn default() -> Self {
        Self { held: false, open: 0.0, at: Vec2::ZERO, mag: MAG_START, target: MAG_START, since_change: f32::INFINITY }
    }
}

impl Lens {
    pub fn update(&mut self, dt: f32, cursor: Option<Vec2>) {
        let dt = dt.clamp(0.0, 0.1);
        let step = dt / OPEN_SECONDS;
        self.open = if self.held { (self.open + step).min(1.0) } else { (self.open - step).max(0.0) };
        if let Some(c) = cursor {
            // closed, it waits under the mouse, so it opens right there
            self.at = if self.open == 0.0 { c } else { self.at + (c - self.at) * (1.0 - (-FOLLOW * dt).exp()) };
        }
        self.mag += (self.target - self.mag) * (1.0 - (-EASE_MAG * dt).exp());
        self.since_change += dt;
    }

    /// The wheel while the lens is held: `steps` notches (a fraction for a smooth wheel or a trackpad), up is stronger.
    pub fn scroll(&mut self, steps: f32) {
        self.target = (self.target * 1.25f32.powf(steps)).clamp(MAG_RANGE.0, MAG_RANGE.1);
        self.since_change = 0.0;
    }

    /// How open it is, eased (quick at first, settling at the end); 0 when closed.
    pub fn shown(&self) -> f32 {
        1.0 - (1.0 - self.open).powi(3)
    }

    /// Its camera this frame, given the map's (`center`, `scale`), the viewport and the screen's pixels per point; None while closed.
    /// The world point under its centre stays where it is, magnified; opening grows the circle and the magnification together.
    pub fn view(&self, center: Vec2, scale: f32, viewport: Vec2, ppp: f32) -> Option<LensView> {
        let shown = self.shown();
        if shown <= 0.001 {
            return None;
        }
        let mag = 1.0 + (self.mag - 1.0) * shown;
        let world = center + (self.at - viewport * 0.5) / scale;
        let scale = scale * mag;
        Some(LensView { center: world - (self.at - viewport * 0.5) / scale, scale, at: self.at, radius: RADIUS * ppp * shown, shown })
    }

    /// "4x", while the magnification is being changed and a moment after; how visible it is, 0 ... 1.
    pub fn label(&self) -> Option<(String, f32)> {
        let alpha = (1.5 - self.since_change).clamp(0.0, 1.0);
        (alpha > 0.0).then(|| (format!("{:.1}x", self.target).replace(".0x", "x"), alpha))
    }
}
