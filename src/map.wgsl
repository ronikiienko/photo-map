// Every photo is one instanced quad with its real shape. Its pixels come from the level pools (pages.rs):
// level L holds each photo in a 2^L px slot. The shader blends the two levels around the on-screen size,
// using only levels the table says are loaded (always 1 ... 16 px, larger ones when streamed in).
// Edges are antialiased analytically: the quad reaches AA px past the photo, and each pixel is covered by how much
// of it lies inside the photo's rounded rect (by its distance to the edge), blended with premultiplied alpha.
// The photo under the mouse is lifted: drawn last (vs_lift), a little bigger, over a soft shadow.

struct Camera {
    center: vec2f,     // world point at the middle of the screen
    viewport: vec2f,   // screen size in pixels
    scale: f32,        // pixels per world unit
    size_px: f32,      // a photo's longer side on screen
    slide: f32,        // 0 ... 1: how far the photos have moved from `from_xy` to `points` (1 once the view has settled)
    copy_px: f32,      // a match's copy beside the world (beside.rs): its longer side on screen
    lens_at: vec2f,    // the lens (lens.rs): its centre on screen, px ...
    lens_radius: f32,  // ... and radius; 0 when this is not the lens's camera (nothing is clipped)
    lift: f32,         // 0 ... 1: how far the photo under the mouse is lifted ...
    lift_photo: u32,   // ... that photo (NONE if it is a copy or there is none) ...
    lift_copy: u32,    // ... or that match's copy beside the world (NONE if not)
    pad: vec2f,
}

const PAGE: f32 = 4096.0;
const BAKED: u32 = 6u;  // levels from here on (64 px) are decoded from the originals; below are baked, fitted to the slot (small originals enlarged)
const ENTRY: u32 = 17u;   // table u32s per photo: loaded level count, original's longer side, 2 unused, slot per level (13 levels: 1 ... 4096 px)
const COLD = vec3f(0.9, 0.15, 0.15);  // a match just above the cutoff ...
const HOT = vec3f(1.0, 0.92, 0.25);   // ... the strongest ones (flags 1 ... 255)
const MATCH: u32 = 255u;   // a photo's flags: its match (0: none) ...
const HIDDEN: u32 = 256u;  // ... and whether it is hidden (a search hidden with "Hide these"): drawn faint
const FAINT: f32 = 0.12;   // how much of a hidden photo shows over the background
const TINT_FROM: f32 = 24.0;  // a match is tinted solid while it is smaller than this on screen (px) ...
const TINT_UNTIL: f32 = 96.0; // ... and shows its own picture by this size
const GROW: f32 = 3.0;        // a match is drawn this much bigger while it is small, real size by TINT_UNTIL
const OUTLINE: f32 = 3.0;     // px: a match keeps an outline of its colour at every zoom, so it stands out once its picture shows
const AA: f32 = 1.0;  // px the quad reaches past the photo's edge, for the partly covered pixels
const BACKGROUND = vec3f(0.047, 0.051, 0.063);  // the map's background (gpu.rs BACKGROUND)
const LENS_SHADOW: f32 = 26.0;  // px the lens's shadow reaches past its edge
const CORNER: f32 = 0.04;       // corners' radius, of the photo's shorter side ...
const CORNER_MAX: f32 = 10.0;   // ... up to this many px
const LIFT: f32 = 0.0;          // a lifted photo is this much bigger (0: it keeps its size)
const LIFT_SHADOW: f32 = 0.12;  // its shadow reaches this much of its longer side past it ...
const LIFT_SHADOW_MAX: f32 = 22.0;  // ... up to this many px, a third of it lower than the photo
const NONE: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> points: array<vec4f>;  // x, y, shape (1, 0.67 for a 3:2 photo; times its size)
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var pages: texture_2d_array<f32>;
@group(0) @binding(4) var pages_sampler: sampler;
@group(0) @binding(5) var<storage, read> table: array<u32>;
@group(0) @binding(6) var<storage, read> first_page: array<u32>;
@group(0) @binding(7) var<storage, read> was: array<vec4f>;     // x, y and size before the view changed, unused
@group(0) @binding(8) var<storage, read> copies: array<vec4f>;   // the matches' copies: x, y, the photo's number (its bits), unused

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) local: vec2f,  // pixels from the photo's centre
    @location(1) @interpolate(flat) id: u32,
    @location(2) @interpolate(flat) flags: u32,
    @location(3) @interpolate(flat) size: vec2f,  // on-screen width, height
    @location(4) @interpolate(flat) shape: vec2f,
    @location(5) @interpolate(flat) px: f32,  // the photo's longer side on screen, before a match grows or it lifts (picks the level)
    @location(6) @interpolate(flat) shadow: f32,  // px the lifted photo's shadow reaches (0: not lifted)
}

// One corner of photo `id`'s quad, centred on `at` (world), `shape` times `px` on screen, drawn `grow` times bigger,
// lifted by `lift` (0 ... 1).
fn quad(v: u32, id: u32, at: vec2f, shape: vec2f, px: f32, grow: f32, lift: f32) -> VsOut {
    var corners = array(vec2f(0., 0.), vec2f(1., 0.), vec2f(0., 1.), vec2f(0., 1.), vec2f(1., 0.), vec2f(1., 1.));
    let size = max(px * shape, vec2f(1.0)) * grow * (1.0 + LIFT * lift);  // longer side = px, the other follows the shape; never under 1 px
    let shadow = min(LIFT_SHADOW * max(size.x, size.y), LIFT_SHADOW_MAX) * lift;
    let local = (corners[v] - 0.5) * (size + 2.0 * AA + 2.7 * shadow);
    let screen = (at - cam.center) * cam.scale + cam.viewport * 0.5 + local;
    var out: VsOut;
    out.pos = vec4f(screen.x / cam.viewport.x * 2.0 - 1.0, 1.0 - screen.y / cam.viewport.y * 2.0, 0.0, 1.0);
    out.local = local;
    out.id = id;
    out.flags = flags[id];
    out.size = size;
    out.shape = shape / max(shape.x, shape.y);
    out.px = px * max(shape.x, shape.y);
    out.shadow = shadow;
    return out;
}

// How much bigger photo `id` is drawn: a match, while it is small.
fn grow(id: u32) -> f32 {
    return 1.0 + select(0.0, GROW - 1.0, (flags[id] & MATCH) != 0u) * (1.0 - smoothstep(TINT_FROM, TINT_UNTIL, cam.size_px));
}

// A photo on the map, sliding from where it was (and its size then) to where it goes.
fn photo_quad(v: u32, id: u32, lift: f32) -> VsOut {
    let p = points[id];
    let f = was[id];
    let size = mix(f.z, max(p.z, p.w), cam.slide);  // its longer side's size; the shape is 1 long times its size
    let shape = p.zw / max(p.z, p.w) * size;
    return quad(v, id, mix(f.xy, p.xy, cam.slide), shape, cam.size_px, grow(id), lift);
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) id: u32) -> VsOut {
    if id == cam.lift_photo { var none: VsOut; return none; }  // drawn last, by vs_lift
    return photo_quad(v, id, 0.0);
}

// A match's copy beside the world: its own size, never grown (they sit in a grid, side by side), and just its picture (no tint,
// no outline: they are in order, best first, and the tip under the mouse says how close each is).
fn copy_quad(v: u32, k: u32, lift: f32) -> VsOut {
    let c = copies[k];
    let id = bitcast<u32>(c.z);
    let shape = points[id].zw;
    var out = quad(v, id, c.xy, shape / max(shape.x, shape.y), cam.copy_px, 1.0, lift);  // its shape at size 1
    out.flags = 0u;
    return out;
}

@vertex
fn vs_copy(@builtin(vertex_index) v: u32, @builtin(instance_index) k: u32) -> VsOut {
    if k == cam.lift_copy { var none: VsOut; return none; }  // drawn last, by vs_lift
    return copy_quad(v, k, 0.0);
}

// The photo or copy under the mouse, over everything else.
@vertex
fn vs_lift(@builtin(vertex_index) v: u32) -> VsOut {
    if cam.lift_copy != NONE { return copy_quad(v, cam.lift_copy, cam.lift); }
    if cam.lift_photo != NONE && cam.lift_photo < arrayLength(&points) { return photo_quad(v, cam.lift_photo, cam.lift); }
    var none: VsOut;
    return none;
}

// One level of one photo: find its slot on its page, sample inside the photo's part of the slot.
fn sample_level(id: u32, level: u32, uv: vec2f, shape: vec2f) -> vec3f {
    let s = f32(1u << level);
    var long = s;
    if level >= BAKED { long = min(s, f32(table[id * ENTRY + 1u])); }  // decoded levels never enlarge
    let content = shape * long;                                            // the photo's size in this slot, texels
    let texel = clamp(uv * content, vec2f(0.5), max(content - 0.5, vec2f(0.5)));  // stay inside it
    let slot = table[id * ENTRY + 4u + level];
    let per_row = u32(PAGE) >> level;
    let per_page = per_row * per_row;
    let local = slot % per_page;
    let origin = vec2f(f32(local % per_row), f32(local / per_row)) * s;
    return textureSampleLevel(pages, pages_sampler, (origin + texel) / PAGE, first_page[level] + slot / per_page, 0.0).rgb;
}

// Px from `p` to the edge of a rect of this size centred on 0 with corners of radius `r` (negative inside).
fn rounded(p: vec2f, size: vec2f, r: f32) -> f32 {
    let q = abs(p) - size * 0.5 + r;
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let edge = rounded(in.local, in.size, min(CORNER * min(in.size.x, in.size.y), CORNER_MAX));
    var cover = clamp(0.5 - edge, 0.0, 1.0);
    var shadow = 0.0;
    if in.shadow > 0.0 {
        let d = rounded(in.local - vec2f(0.0, in.shadow / 3.0), in.size, min(CORNER_MAX, 0.5 * min(in.size.x, in.size.y)));
        shadow = 0.55 * (1.0 - smoothstep(-in.shadow * 0.5, in.shadow, d)) * clamp(in.shadow / 6.0, 0.0, 1.0);
    }
    var clip = 1.0;
    if cam.lens_radius > 0.0 { clip = clamp(cam.lens_radius - distance(in.pos.xy, cam.lens_at) + 0.5, 0.0, 1.0); }  // inside the lens only
    cover *= clip;
    shadow *= clip;
    if cover <= 0.0 && shadow <= 0.0 { discard; }
    let uv = clamp(in.local / in.size + 0.5, vec2f(0.0), vec2f(1.0));

    // ideal level: slot size = on-screen size; blend it with the next one up, within what is loaded
    let top = f32(table[in.id * ENTRY] - 1u);
    let f = clamp(log2(max(in.px, 1.0)), 0.0, top);
    let lo = floor(f);
    let hi = min(lo + 1.0, top);
    var c = sample_level(in.id, u32(lo), uv, in.shape);
    if hi > lo { c = mix(c, sample_level(in.id, u32(hi), uv, in.shape), f - lo); }
    let found = in.flags & MATCH;
    if found != 0u {  // a match: tinted from afar (red to yellow with the match), its picture up close
        let tint = mix(COLD, HOT, f32(found - 1u) / 254.0);
        c = mix(c, tint, 1.0 - smoothstep(TINT_FROM, TINT_UNTIL, in.px));
        c = mix(tint, c, smoothstep(OUTLINE - 1.0, OUTLINE, -edge));  // the outline, inside the photo's edge
    }
    if (in.flags & HIDDEN) != 0u { c = mix(BACKGROUND, c, FAINT); }  // hidden: almost the background, its place kept
    return vec4f(c * cover, cover + (1.0 - cover) * shadow);  // over its shadow (black)
}


// The lens's ground, drawn before the photos it shows: the background inside the circle (covering the map under it), and a soft
// shadow round it.
@vertex
fn vs_disc(@builtin(vertex_index) v: u32) -> @builtin(position) vec4f {
    var corners = array(vec2f(0., 0.), vec2f(1., 0.), vec2f(0., 1.), vec2f(0., 1.), vec2f(1., 0.), vec2f(1., 1.));
    let screen = cam.lens_at + (corners[v] - 0.5) * 2.0 * (cam.lens_radius + LENS_SHADOW);
    return vec4f(screen.x / cam.viewport.x * 2.0 - 1.0, 1.0 - screen.y / cam.viewport.y * 2.0, 0.0, 1.0);
}

@fragment
fn fs_disc(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let d = distance(pos.xy, cam.lens_at);
    let inside = clamp(cam.lens_radius - d + 0.5, 0.0, 1.0);
    let past = max(d - cam.lens_radius, 0.0) / LENS_SHADOW;
    let shadow = 0.6 * (1.0 - smoothstep(0.0, 1.0, past)) * clamp(cam.lens_radius / 40.0, 0.0, 1.0);  // fades in as the lens grows
    return vec4f(BACKGROUND * inside, inside + (1.0 - inside) * shadow);
}
