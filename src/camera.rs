//! 2D camera with coasting zoom and pan, after mote-control's camera_controller.
//! Constants are "per frame at 60 Hz"; `smooth`/`factor` rescale them so every refresh rate feels the same.

use glam::Vec2;

const HZ: f32 = 60.0;
const PAN_FRICTION: f32 = 0.9;
/// How much of the recorded drag velocity carries over each frame; keeps a pause before release from killing the fling.
const PAN_SMOOTHING: f32 = 0.8;
/// Screen pixels per second below which a coast stops.
const PAN_STOP: f32 = 5.0;
const ZOOM_FRICTION: f32 = 0.9;
/// Log zoom per wheel line (one notch); with the coast a notch zooms about 1.8x in total.
const ZOOM_PER_LINE: f32 = 0.12;
const ZOOM_PER_PIXEL: f32 = 0.0012;
const ZOOM_STOP: f32 = 1e-4;
/// Log zoom per unit of a trackpad pinch (the system's own: the picture grows by what the fingers spread by).
const PINCH_ZOOM: f32 = 1.0;
const KEY_SPEED: f32 = 900.0;
const KEY_FRICTION: f32 = 0.8;
const KEY_ZOOM: f32 = 0.05;
const FLY_CARRY: f32 = 0.85;
/// Zoomed out at most to where the content is this much of its fitted size.
const MIN_ZOOM: f32 = 0.5;
/// Zoomed in at most to where a photo's longer side is this many times the window's shorter side.
const MAX_PHOTO: f32 = 2.0;
/// Content smaller than the screen stays on it, at least this much of the window from its edges.
const MARGIN: f32 = 0.03;

fn factor(f: f32, dt: f32) -> f32 {
    f.max(0.0).powf(dt * HZ)
}

fn smooth<T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>>(cur: T, sample: T, carry: f32, dt: f32) -> T {
    let c = factor(carry, dt);
    cur * c + sample * (1.0 - c)
}

#[derive(Clone, Copy, PartialEq)]
enum Pan {
    Idle,
    /// Held: the view follows the pointer exactly; velocity is only recorded, for the fling.
    Dragging { velocity: Vec2 },
    Coasting { velocity: Vec2 },
}

/// What the camera may show: the content's box (world) and a photo's longer side in it (world units).
#[derive(Clone, Copy)]
pub struct Limits {
    pub lo: Vec2,
    pub hi: Vec2,
    pub photo: f32,
}

#[derive(Default, Clone, Copy)]
pub struct Keys {
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub zoom_in: bool,
    pub zoom_out: bool,
}

pub struct Camera {
    /// World point at the middle of the screen.
    pub center: Vec2,
    /// Screen pixels per world unit.
    pub scale: f32,
    pub viewport: Vec2,
    /// Screen pixels at the top that the bar covers (the search box, the menus): what is fitted stays below them.
    pub top: f32,
    pan: Pan,
    drag_moved: Vec2,
    zoom_velocity: f32,
    pending_zoom: f32,
    /// A trackpad pinch and a two-finger pan, as they arrive: applied at once, with no coast of their own (the system sends its own momentum).
    pending_pinch: f32,
    pending_pan: Vec2,
    key_velocity: Vec2,
    fly: Option<(Vec2, f32)>,
    /// None: anywhere, any zoom.
    pub limits: Option<Limits>,
    /// This frame's scale range (update): the limits', widened to where a scale already outside them eases back from.
    scale_range: (f32, f32),
}

impl Camera {
    pub fn new(viewport: Vec2) -> Self {
        Self {
            top: 0.0,
            center: Vec2::ZERO,
            scale: 1.0,
            viewport,
            pan: Pan::Idle,
            drag_moved: Vec2::ZERO,
            zoom_velocity: 0.0,
            pending_zoom: 0.0,
            pending_pinch: 0.0,
            pending_pan: Vec2::ZERO,
            key_velocity: Vec2::ZERO,
            fly: None,
            limits: None,
            scale_range: (0.0, f32::INFINITY),
        }
    }

    /// The scales the limits allow.
    fn scales(&self, l: &Limits) -> (f32, f32) {
        let most = MAX_PHOTO * self.viewport.min_element() / l.photo.max(1e-6);
        ((self.fit(l.lo, l.hi).1 * MIN_ZOOM).min(most), most)
    }

    /// The centres the limits allow at this scale: any point of the content can be in the middle of the screen below the bar, and
    /// content smaller than that screen stays on it.
    fn centers(&self, l: &Limits, scale: f32) -> (Vec2, Vec2) {
        let shift = Vec2::new(0.0, self.top * 0.5 / scale); // the middle below the bar, from the screen's middle
        let half = (self.viewport * 0.5 - Vec2::new(0.0, self.top * 0.5)) / scale - MARGIN * self.viewport.min_element() / scale;
        (l.lo.min(l.hi - half) - shift, l.hi.max(l.lo + half) - shift)
    }

    pub fn to_world(&self, screen: Vec2) -> Vec2 {
        self.center + (screen - self.viewport * 0.5) / self.scale
    }

    pub fn to_screen(&self, world: Vec2) -> Vec2 {
        (world - self.center) * self.scale + self.viewport * 0.5
    }

    /// Scale and centre that show the world box `min..max` with some margin.
    pub fn fit(&self, min: Vec2, max: Vec2) -> (Vec2, f32) {
        let size = (max - min).max(Vec2::splat(1.0)) + Vec2::splat(400.0);
        // in the screen below the bar: the middle moves down by half of it
        let scale = ((self.viewport - Vec2::new(0.0, self.top)) / size).min_element();
        ((min + max) * 0.5 - Vec2::new(0.0, self.top * 0.5 / scale), scale)
    }

    pub fn jump(&mut self, (center, scale): (Vec2, f32)) {
        self.center = center;
        self.scale = scale;
    }

    /// Glide to a centre and scale; any input cancels it.
    pub fn fly_to(&mut self, (center, scale): (Vec2, f32)) {
        self.fly = Some((center, scale.ln()));
    }

    pub fn scroll_lines(&mut self, lines: f32) {
        self.pending_zoom += lines * ZOOM_PER_LINE;
        self.fly = None;
    }

    pub fn scroll_pixels(&mut self, pixels: f32) {
        self.pending_zoom += pixels * ZOOM_PER_PIXEL;
        self.fly = None;
    }

    /// A trackpad pinch (positive: fingers spreading): zooms about the pointer.
    pub fn pinch(&mut self, delta: f32) {
        self.pending_pinch += delta * PINCH_ZOOM;
        self.fly = None;
    }

    /// Two fingers on the trackpad moved by this many screen pixels: the content follows them.
    pub fn pan_pixels(&mut self, delta: Vec2) {
        self.pending_pan += delta;
        self.fly = None;
    }

    pub fn press(&mut self) {
        self.pan = Pan::Dragging { velocity: Vec2::ZERO };
        self.drag_moved = Vec2::ZERO;
        self.fly = None;
    }

    pub fn release(&mut self) {
        if let Pan::Dragging { velocity } = self.pan {
            self.pan = Pan::Coasting { velocity };
        }
    }

    /// How far (screen pixels) the pointer has moved since the press: a click is a press that hardly moved.
    pub fn drag_distance(&self) -> f32 {
        self.drag_moved.length()
    }

    pub fn dragging(&self) -> bool {
        matches!(self.pan, Pan::Dragging { .. })
    }

    /// Pointer moved by `delta` screen pixels while dragging: the content follows it.
    pub fn drag(&mut self, delta: Vec2) {
        if self.dragging() {
            self.center -= delta / self.scale;
            self.drag_moved += delta;
        }
    }

    fn zoom_about(&mut self, f: f32, screen: Vec2) {
        let before = self.to_world(screen);
        self.scale = (self.scale * f).clamp(self.scale_range.0, self.scale_range.1);
        self.center += before - self.to_world(screen);
    }

    /// Advance one frame. Returns true while anything is moving.
    pub fn update(&mut self, dt: f32, cursor: Option<Vec2>, keys: Keys) -> bool {
        let dt = dt.clamp(1e-6, 0.1);
        let mut moving = false;
        // past the limits (they changed, or the window did) the camera eases back in; input can't take it further out
        // (`near`: close enough to the limit to stop there)
        let ease = |cur: f32, lo: f32, hi: f32, near: f32| {
            let inside = cur.clamp(lo, hi);
            let back = smooth(cur, inside, FLY_CARRY, dt);
            let back = if (back - inside).abs() < near { inside } else { back };
            (lo.min(back), hi.max(back))
        };
        let limits = self.limits.filter(|_| self.fly.is_none());
        let before = self.center;
        self.scale_range = match &limits {
            Some(l) => {
                let (lo, hi) = self.scales(l);
                let (lo, hi) = ease(self.scale.ln(), lo.ln(), hi.ln(), 0.002);
                (lo.exp(), hi.exp())
            }
            None => (0.0, f32::INFINITY),
        };
        if let (Some(l), Some((target, log_scale))) = (self.limits, self.fly) {
            let (lo, hi) = self.scales(&l);
            let log_scale = log_scale.clamp(lo.ln(), hi.ln());
            let (lo, hi) = self.centers(&l, log_scale.exp());
            self.fly = Some((target.clamp(lo, hi), log_scale));
        }

        // Wheel: a scroll sets the zoom velocity; without scroll it decays, so notches blend into one zoom.
        if self.pending_zoom != 0.0 {
            self.zoom_velocity = self.pending_zoom;
            self.pending_zoom = 0.0;
        } else {
            self.zoom_velocity = smooth(self.zoom_velocity, 0.0, ZOOM_FRICTION, dt);
        }
        let key_zoom = KEY_ZOOM * (keys.zoom_in as i32 - keys.zoom_out as i32) as f32;
        if key_zoom != 0.0 {
            self.zoom_velocity = key_zoom;
            self.fly = None;
        }
        if self.zoom_velocity.abs() < ZOOM_STOP {
            self.zoom_velocity = 0.0;
        } else {
            let about = if key_zoom != 0.0 { self.viewport * 0.5 } else { cursor.unwrap_or(self.viewport * 0.5) };
            let was = self.scale;
            self.zoom_about(factor(self.zoom_velocity.exp(), dt), about);
            if self.scale == was {
                self.zoom_velocity = 0.0; // at a limit
            }
            moving = true;
        }

        if self.pending_pinch != 0.0 {
            self.zoom_about(self.pending_pinch.exp(), cursor.unwrap_or(self.viewport * 0.5));
            self.pending_pinch = 0.0;
            moving = true;
        }
        if self.pending_pan != Vec2::ZERO {
            self.center -= self.pending_pan / self.scale;
            self.pending_pan = Vec2::ZERO;
            moving = true;
        }

        match &mut self.pan {
            Pan::Idle => {}
            Pan::Dragging { velocity } => {
                *velocity = smooth(*velocity, self.drag_moved / dt, PAN_SMOOTHING, dt);
                self.drag_moved = Vec2::ZERO;
                moving = true;
            }
            Pan::Coasting { velocity } => {
                *velocity = smooth(*velocity, Vec2::ZERO, PAN_FRICTION, dt);
                if velocity.length() < PAN_STOP {
                    self.pan = Pan::Idle;
                } else {
                    self.center -= *velocity * dt / self.scale;
                    moving = true;
                }
            }
        }

        // Keys move at a fixed screen speed, whatever the zoom, with a short coast on release.
        let dir = Vec2::new(
            (keys.right as i32 - keys.left as i32) as f32,
            (keys.down as i32 - keys.up as i32) as f32,
        );
        if dir != Vec2::ZERO {
            self.key_velocity = dir.normalize() * KEY_SPEED;
            self.fly = None;
        } else {
            self.key_velocity = smooth(self.key_velocity, Vec2::ZERO, KEY_FRICTION, dt);
        }
        if self.key_velocity.length() < PAN_STOP {
            self.key_velocity = Vec2::ZERO;
        } else {
            self.center += self.key_velocity * dt / self.scale;
            moving = true;
        }

        // Fly: ease the centre and the log of the scale toward the target.
        if let Some((target, log_scale)) = self.fly {
            self.center = smooth(self.center, target, FLY_CARRY, dt);
            self.scale = smooth(self.scale.ln(), log_scale, FLY_CARRY, dt).exp();
            if (self.center - target).length() * self.scale < 0.5 && (self.scale.ln() - log_scale).abs() < 0.002 {
                self.fly = None;
            }
            moving = true;
        }

        if let Some(l) = limits {
            // a scale left outside (the limits or the window changed) comes back, about the middle
            let was = self.scale;
            self.zoom_about(1.0, self.viewport * 0.5);
            moving |= self.scale != was;
            let (lo, hi) = self.centers(&l, self.scale);
            let mut stopped = [false; 2];
            for a in 0..2 {
                let (lo, hi) = ease(before[a], lo[a], hi[a], 0.5 / self.scale);
                let c = self.center[a].clamp(lo, hi);
                stopped[a] = c != self.center[a];
                if c != before[a] {
                    moving = true;
                }
                self.center[a] = c;
            }
            let stop = |v: &mut Vec2| {
                for a in 0..2 {
                    if stopped[a] {
                        v[a] = 0.0;
                    }
                }
            };
            if let Pan::Coasting { velocity } = &mut self.pan {
                stop(velocity);
            }
            stop(&mut self.key_velocity);
        }
        moving
    }
}
