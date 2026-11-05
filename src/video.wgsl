// The playing video's frame (player.rs): NV12 as ffmpeg sends it, brightness in one texture and the two colour channels, at half
// the size each way, in another; turned into colours here. Drawn inside egui's pass over the video's tile, with its corners
// rounded as the map's own (map.wgsl).

struct Video {
    // the tile on screen (px; it may reach past the screen's edges: egui's viewport is clamped to it, so the fragment's own
    // position is used, not the viewport's)
    min: vec2f,
    size: vec2f,
    corner: f32,
    // HD (BT.709) colours, else SD (BT.601)
    hd: f32,
    pad: vec2f,
}

@group(0) @binding(0) var<uniform> v: Video;
@group(0) @binding(1) var luma: texture_2d<f32>;
@group(0) @binding(2) var chroma: texture_2d<f32>;
@group(0) @binding(3) var smooth_sampler: sampler;

// one triangle over the whole viewport
@vertex
fn vs_video(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_video(@builtin(position) at: vec4f) -> @location(0) vec4f {
    let local = at.xy - v.min;
    let uv = local / v.size;
    // limited range (16-235, 16-240), as ffmpeg is asked for
    let y = (textureSample(luma, smooth_sampler, uv).r - 16.0 / 255.0) * (255.0 / 219.0);
    let c = (textureSample(chroma, smooth_sampler, uv).rg - 128.0 / 255.0) * (255.0 / 224.0);
    var rgb: vec3f;
    if v.hd > 0.5 {
        rgb = vec3f(y + 1.5748 * c.y, y - 0.1873 * c.x - 0.4681 * c.y, y + 1.8556 * c.x);
    } else {
        rgb = vec3f(y + 1.402 * c.y, y - 0.344136 * c.x - 0.714136 * c.y, y + 1.772 * c.x);
    }
    // rounded corners, antialiased (premultiplied)
    let half = v.size * 0.5;
    let q = abs(local - half) - half + v.corner;
    let d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - v.corner;
    let a = clamp(0.5 - d, 0.0, 1.0);
    return vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)) * a, a);
}
