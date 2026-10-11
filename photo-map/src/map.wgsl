// Every photo is one instanced quad with its real shape. Its pixels come from the level pools (pages.rs):
// level L holds each photo in a 2^L px slot. The shader blends the two levels around the on-screen size,
// using only levels the table says are loaded (always 1 ... 64 px, larger ones when streamed in).

struct Camera {
    center: vec2f,     // world point at the middle of the screen
    viewport: vec2f,   // screen size in pixels
    scale: f32,        // pixels per world unit
    size_px: f32,      // a photo's longer side on screen
    searching: u32,    // 1 = a search is active: dim everything that isn't a hit
    _pad: u32,
}

const PAGE: f32 = 4096.0;
const ENTRY: u32 = 16u;   // table u32s per photo: loaded level count, original's longer side, 2 unused, slot per level
const STREAMED: u32 = 7u; // levels from here on are streamed; below are fitted to the slot (small originals enlarged)
const HIT: u32 = 1u;
const SELECTED: u32 = 2u;
const RED = vec3f(1.0, 0.3, 0.3);
const YELLOW = vec3f(1.0, 0.85, 0.3);

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> points: array<vec4f>;  // x, y, shape (1, 0.67 for a 3:2 photo)
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var pages: texture_2d_array<f32>;
@group(0) @binding(4) var pages_sampler: sampler;
@group(0) @binding(5) var<storage, read> table: array<u32>;
@group(0) @binding(6) var<storage, read> first_page: array<u32>;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) @interpolate(flat) id: u32,
    @location(2) @interpolate(flat) flags: u32,
    @location(3) @interpolate(flat) size: vec2f,  // on-screen width, height
    @location(4) @interpolate(flat) shape: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) id: u32) -> VsOut {
    var corners = array(vec2f(0., 0.), vec2f(1., 0.), vec2f(0., 1.), vec2f(0., 1.), vec2f(1., 0.), vec2f(1., 1.));
    let corner = corners[v];
    let p = points[id];
    let size = max(cam.size_px * p.zw, vec2f(1.0));  // longer side = size_px, the other follows the shape; never under 1 px
    let px = (p.xy - cam.center) * cam.scale + cam.viewport * 0.5 + (corner - 0.5) * size;
    var out: VsOut;
    out.pos = vec4f(px.x / cam.viewport.x * 2.0 - 1.0, 1.0 - px.y / cam.viewport.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    out.id = id;
    out.flags = flags[id];
    out.size = size;
    out.shape = p.zw;
    return out;
}

// One level of one photo: find its slot on its page, sample inside the photo's part of the slot.
fn sample_level(id: u32, level: u32, uv: vec2f, shape: vec2f) -> vec3f {
    let s = f32(1u << level);
    var long = s;
    if level >= STREAMED { long = min(s, f32(table[id * ENTRY + 1u])); }  // streamed levels never enlarge
    let content = shape * long;                                            // the photo's size in this slot, texels
    let texel = clamp(uv * content, vec2f(0.5), max(content - 0.5, vec2f(0.5)));  // stay inside it
    let slot = table[id * ENTRY + 4u + level];
    let per_row = u32(PAGE) >> level;
    let per_page = per_row * per_row;
    let local = slot % per_page;
    let origin = vec2f(f32(local % per_row), f32(local / per_row)) * s;
    return textureSampleLevel(pages, pages_sampler, (origin + texel) / PAGE, first_page[level] + slot / per_page, 0.0).rgb;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let hit = (in.flags & HIT) != 0u;
    let selected = (in.flags & SELECTED) != 0u;
    // a frame around hits (red) and the selected photo (yellow); tiny squares get tinted instead
    let edge = min(min(in.uv.x, 1.0 - in.uv.x) * in.size.x, min(in.uv.y, 1.0 - in.uv.y) * in.size.y);
    if selected && edge < 3.0 { return vec4f(YELLOW, 1.0); }
    if hit && (edge < 2.0 || cam.size_px < 6.0) { return vec4f(RED, 1.0); }

    // ideal level: slot size = on-screen size; blend it with the next one up, within what is loaded
    let top = f32(table[in.id * ENTRY] - 1u);
    let f = clamp(log2(max(cam.size_px, 1.0)), 0.0, top);
    let lo = floor(f);
    let hi = min(lo + 1.0, top);
    var c = sample_level(in.id, u32(lo), in.uv, in.shape);
    if hi > lo { c = mix(c, sample_level(in.id, u32(hi), in.uv, in.shape), f - lo); }
    let dim = select(1.0, 0.25, cam.searching == 1u && !hit && !selected);
    return vec4f(c * dim, 1.0);
}
