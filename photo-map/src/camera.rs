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
const ZOOM_PER_LINE: f32 = 0.06;
const ZOOM_PER_PIXEL: f32 = 0.0006;
const ZOOM_STOP: f32 = 1e-4;
const KEY_SPEED: f32 = 900.0;
const KEY_FRICTION: f32 = 0.8;
const KEY_ZOOM: f32 = 0.05;
const FLY_CARRY: f32 = 0.85;

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
    pan: Pan,
    drag_moved: Vec2,
    zoom_velocity: f32,
    pending_zoom: f32,
    key_velocity: Vec2,
    fly: Option<(Vec2, f32)>,
}

impl Camera {
    pub fn new(viewport: Vec2) -> Self {
        Self {
            center: Vec2::ZERO,
            scale: 1.0,
            viewport,
            pan: Pan::Idle,
            drag_moved: Vec2::ZERO,
            zoom_velocity: 0.0,
            pending_zoom: 0.0,
            key_velocity: Vec2::ZERO,
            fly: None,
        }
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
        ((min + max) * 0.5, (self.viewport / size).min_element())
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
        self.scale *= f;
        self.center += before - self.to_world(screen);
    }

    /// Advance one frame. Returns true while anything is moving.
    pub fn update(&mut self, dt: f32, cursor: Option<Vec2>, keys: Keys) -> bool {
        let dt = dt.clamp(1e-6, 0.1);
        let mut moving = false;

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
            self.zoom_about(factor(self.zoom_velocity.exp(), dt), about);
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
        moving
    }
}
