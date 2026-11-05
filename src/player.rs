//! Playing a video in its tile (no sound), as players do: ffmpeg decodes it (on the graphics card's decoder when there is one,
//! `-hwaccel auto`) from a time to raw NV12 frames at the video's own size, as fast as it can, but only a few frames ahead (the
//! pipe fills and it waits); the reader thread hands them over at the video's frame rate, and the window uploads the newest and
//! turns it into colours on the card (video.wgsl): 1.5 bytes a pixel instead of RGBA's 4, no converting on the processor.
//! Pausing holds the reader (ffmpeg waits, its next frame ready: playing on is at once); going to another time starts ffmpeg
//! there again.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A frame: brightness (a byte a pixel), then the two colour channels interleaved at half the size each way (NV12).
pub struct Frame {
    size: (u32, u32),
    bytes: Vec<u8>,
}

/// Between the player and its reader thread (one per ffmpeg).
#[derive(Default)]
struct Shared {
    /// The newest frame, not yet drawn.
    frame: Mutex<Option<Frame>>,
    /// How many frames were handed over: the position is the last one's time.
    shown: AtomicU32,
    /// From ffmpeg's own words on the stream: frames a second, and HD colours (BT.709) or SD (BT.601).
    fps: OnceLock<(f32, Option<bool>)>,
    /// Held (paused), or let go for good (ffmpeg stopped); the reader waits on it.
    hold: Mutex<Hold>,
    go: Condvar,
    /// The video has run out.
    ended: AtomicBool,
}

#[derive(Default)]
struct Hold {
    paused: bool,
    stopped: bool,
}

pub struct Player {
    /// The video (a photo of the map), and where it plays: on the map, or in a match's copy beside it (its position).
    pub photo: usize,
    pub copy: Option<usize>,
    pub duration: f32,
    ffmpeg: PathBuf,
    video: PathBuf,
    /// The frames' size (px): the video's own, upright, even sides (for NV12).
    pub size: (u32, u32),
    /// HD colours (BT.709), else SD (BT.601), when the video doesn't say: by size, as players do.
    hd: bool,
    /// The time ffmpeg was started from, and whether it is to play (else paused).
    at: f32,
    playing: bool,
    child: Option<Child>,
    shared: Arc<Shared>,
    wake: Arc<dyn Fn() + Send + Sync>,
    /// A frame has come, so there is one on the card to draw.
    shown: bool,
}

impl Player {
    /// Plays `video` (its own size: `own`, upright) from `at` seconds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(ffmpeg: &Path, video: &Path, photo: usize, copy: Option<usize>, duration: f32, at: f32, own: (u32, u32), wake: Arc<dyn Fn() + Send + Sync>) -> Self {
        let even = |v: u32| (v / 2).max(1) * 2;
        let size = (even(own.0), even(own.1));
        let hd = own.0.max(own.1) >= 1280 || own.0.min(own.1) > 576;
        let mut player = Self { photo, copy, duration, ffmpeg: ffmpeg.to_path_buf(), video: video.to_path_buf(), size, hd, at: at.clamp(0.0, duration), playing: true, child: None, shared: Arc::new(Shared::default()), wake, shown: false };
        player.start();
        player
    }

    /// Where it is (seconds): the time of the frame shown last.
    pub fn position(&self) -> f32 {
        if self.shared.ended.load(Ordering::Relaxed) {
            return self.duration;
        }
        let shown = self.shared.shown.load(Ordering::Relaxed);
        let fps = self.shared.fps.get().map_or(30.0, |f| f.0);
        (self.at + shown.saturating_sub(1) as f32 / fps).min(self.duration)
    }

    pub fn playing(&self) -> bool {
        self.playing && !self.shared.ended.load(Ordering::Relaxed)
    }

    /// Play on (from the start again once it has run out).
    pub fn play(&mut self) {
        self.playing = true;
        if self.shared.ended.load(Ordering::Relaxed) || self.child.is_none() {
            self.at = if self.shared.ended.load(Ordering::Relaxed) { 0.0 } else { self.position() };
            self.start();
        } else {
            self.shared.hold.lock().unwrap().paused = false;
            self.shared.go.notify_all();
        }
    }

    /// Stop where it is: ffmpeg waits with its next frame.
    pub fn pause(&mut self) {
        self.playing = false;
        self.shared.hold.lock().unwrap().paused = true;
    }

    /// Go to `t` seconds (its frame shows even when paused), playing on if it was playing.
    pub fn seek(&mut self, t: f32) {
        self.at = t.clamp(0.0, self.duration);
        self.start();
    }

    /// ffmpeg from `at`, and a reader thread handing its frames over in time; paused, it hands over the first and waits.
    fn start(&mut self) {
        self.stop_ffmpeg();
        self.shared = Arc::new(Shared::default());
        self.shared.hold.lock().unwrap().paused = !self.playing;
        let (w, h) = self.size;
        let child = Command::new(&self.ffmpeg)
            .args(["-nostdin", "-hide_banner", "-nostats", "-loglevel", "info", "-hwaccel", "auto", "-ss"])
            .arg(format!("{:.3}", self.at))
            .arg("-i")
            .arg(&self.video)
            .args(["-an", "-vf"])
            .arg(format!("scale={w}:{h}:out_range=tv"))
            // a frame every 1/fps (a phone's video may vary): the position is counted in frames
            .args(["-fps_mode", "cfr", "-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();
        let Ok(mut child) = child else {
            self.shared.ended.store(true, Ordering::Relaxed);
            return;
        };
        let (mut out, err) = (child.stdout.take().expect("piped"), child.stderr.take().expect("piped"));
        let (shared, wake) = (self.shared.clone(), self.wake.clone());
        let bytes = (w * h * 3 / 2) as usize;
        std::thread::spawn(move || {
            // what ffmpeg says about the stream it sends (before the first frame), then the rest of what it says, unread
            let mut err = BufReader::new(err);
            let mut line = String::new();
            let mut output = false;
            while err.read_line(&mut line).is_ok_and(|n| n > 0) {
                output |= line.starts_with("Output #0");
                if output && line.contains("Video:") {
                    let _ = shared.fps.set(stream_facts(&line));
                    break;
                }
                line.clear();
            }
            std::thread::spawn(move || std::io::copy(&mut err, &mut std::io::sink()));
            let fps = shared.fps.get().map_or(30.0, |f| f.0);
            let mut next = Instant::now();
            loop {
                let mut frame = vec![0u8; bytes];
                if out.read_exact(&mut frame).is_err() {
                    break;
                }
                // in time (after the first: it shows at once, even paused, as a time gone to while paused does)
                let first = shared.shown.load(Ordering::Relaxed) == 0;
                if !first {
                    std::thread::sleep(next.saturating_duration_since(Instant::now()));
                }
                // handed over under the hold: once pause() is done, nothing more shows
                let mut hold = shared.hold.lock().unwrap();
                let mut held = false;
                while hold.paused && !hold.stopped && !first {
                    held = true;
                    hold = shared.go.wait(hold).unwrap();
                }
                if hold.stopped {
                    return;
                }
                *shared.frame.lock().unwrap() = Some(Frame { size: (w, h), bytes: frame });
                shared.shown.fetch_add(1, Ordering::Relaxed);
                drop(hold);
                wake();
                // the next one 1/fps on; after a pause, or behind (a slow decoder), from now, not hurrying to catch up
                let now = Instant::now();
                if held || first || next + Duration::from_millis(100) < now {
                    next = now;
                }
                next += Duration::from_secs_f32(1.0 / fps);
            }
            if !shared.hold.lock().unwrap().stopped {
                shared.ended.store(true, Ordering::Relaxed);
                wake();
            }
        });
        self.child = Some(child);
    }

    fn stop_ffmpeg(&mut self) {
        self.shared.hold.lock().unwrap().stopped = true;
        self.shared.go.notify_all();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Drawing its newest frame over `rect` (points), corners rounded by `corner`: uploaded first when a new one came. None
    /// until the first one has.
    pub fn paint(&mut self, rect: egui::Rect, corner: f32) -> Option<egui::Shape> {
        let frame = self.shared.frame.lock().unwrap().take();
        self.shown |= frame.is_some();
        let hd = self.shared.fps.get().and_then(|f| f.1).unwrap_or(self.hd);
        self.shown.then(|| egui_wgpu::Callback::new_paint_callback(rect, Paint { frame: Mutex::new(frame), rect, corner, hd }).into())
    }
}

/// From ffmpeg's line about a video stream (`... nv12(tv, bt709, progressive), 1920x1080 ..., 59.94 fps, ...`): frames a second
/// (30 if it doesn't say), and whether its colours are HD's (BT.709) or SD's, if it says.
fn stream_facts(line: &str) -> (f32, Option<bool>) {
    let fps = line.split(", ").find_map(|part| part.strip_suffix(" fps")?.parse::<f32>().ok()).filter(|f| *f > 0.0).unwrap_or(30.0);
    let colours = if line.contains("bt709") {
        Some(true)
    } else if line.contains("smpte170m") || line.contains("bt470bg") || line.contains("bt601") {
        Some(false)
    } else {
        None
    };
    (fps, colours)
}

/// The video's frame on the card, and how it is drawn (video.wgsl); kept in egui's renderer (its callback resources).
pub struct VideoGpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    /// The frame's two textures (its size), and their bind group.
    planes: Option<((u32, u32), wgpu::Texture, wgpu::Texture, wgpu::BindGroup)>,
}

impl VideoGpu {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video"),
            source: wgpu::ShaderSource::Wgsl(include_str!("video.wgsl").into()),
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty, count: None };
        let texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("video"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, texture),
                entry(2, texture),
                entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("video"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("video"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_video"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_video"),
                targets: &[Some(wgpu::ColorTargetState { format, blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("video"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, layout, sampler, uniform, planes: None }
    }

    /// The frame onto the card (new textures when its size changed).
    fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, frame: &Frame) {
        let (w, h) = frame.size;
        if self.planes.as_ref().is_none_or(|p| p.0 != frame.size) {
            let plane = |w, h, format| device.create_texture(&wgpu::TextureDescriptor {
                label: Some("video"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let (luma, chroma) = (plane(w, h, wgpu::TextureFormat::R8Unorm), plane(w / 2, h / 2, wgpu::TextureFormat::Rg8Unorm));
            let view = |t: &wgpu::Texture| t.create_view(&Default::default());
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("video"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view(&luma)) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view(&chroma)) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });
            self.planes = Some((frame.size, luma, chroma, bind_group));
        }
        let (_, luma, chroma, _) = self.planes.as_ref().expect("made above");
        let write = |texture: &wgpu::Texture, bytes: &[u8], row: u32, (w, h): (u32, u32)| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            )
        };
        let (y, uv) = frame.bytes.split_at((w * h) as usize);
        write(luma, y, w, (w, h));
        write(chroma, uv, w, (w / 2, h / 2));
    }
}

/// One frame's drawing of the video (egui calls it inside its pass, in order with the rest of the UI).
struct Paint {
    frame: Mutex<Option<Frame>>,
    rect: egui::Rect,
    corner: f32,
    hd: bool,
}

impl egui_wgpu::CallbackTrait for Paint {
    fn prepare(&self, device: &wgpu::Device, queue: &wgpu::Queue, screen: &egui_wgpu::ScreenDescriptor, _: &mut wgpu::CommandEncoder, resources: &mut egui_wgpu::CallbackResources) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = resources.get_mut::<VideoGpu>() else { return Vec::new() };
        if let Some(frame) = self.frame.lock().unwrap().take() {
            gpu.upload(device, queue, &frame);
        }
        let ppp = screen.pixels_per_point;
        let (min, size) = (self.rect.min * ppp, self.rect.size() * ppp);
        let uniform = [min.x, min.y, size.x, size.y, self.corner * ppp, if self.hd { 1.0 } else { 0.0 }, 0.0, 0.0];
        queue.write_buffer(&gpu.uniform, 0, bytemuck::cast_slice(&uniform));
        Vec::new()
    }

    fn paint(&self, _: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &egui_wgpu::CallbackResources) {
        let Some(gpu) = resources.get::<VideoGpu>() else { return };
        let Some((_, _, _, bind_group)) = &gpu.planes else { return };
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop_ffmpeg();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_stream() {
        let line = "  Stream #0:0(eng): Video: rawvideo (NV12 / 0x3231564E), nv12(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], q=2-31, 1491500 kb/s, 59.94 fps, 59.94 tbn (default)";
        assert_eq!(stream_facts(line), (59.94, Some(true)));
        assert_eq!(stream_facts("Video: rawvideo, nv12(tv, smpte170m), 640x480, 25 fps, 25 tbn"), (25.0, Some(false)));
        assert_eq!(stream_facts("Video: rawvideo, nv12, 640x480"), (30.0, None));
    }

    /// video.wgsl is only compiled when a video plays: parse and validate it here.
    #[test]
    fn shader_is_valid() {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("video.wgsl")).expect("video.wgsl does not parse");
        wgpu::naga::valid::Validator::new(Default::default(), wgpu::naga::valid::Capabilities::all()).validate(&module).expect("video.wgsl is not valid");
    }

    /// Frames come at the video's size, from the time asked for, and stop when paused. Needs ffmpeg; no window.
    #[test]
    fn plays_and_pauses() {
        let Some(ffmpeg) = crate::runtime::ffmpeg(&crate::data_root()) else { return };
        let temp = std::env::temp_dir().join(format!("photo-map-player-test-{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();
        let video = temp.join("v.mp4");
        assert!(Command::new(&ffmpeg).args(["-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x180:rate=25"]).arg(&video).status().unwrap().success());
        let mut p = Player::new(&ffmpeg, &video, 0, None, 3.0, 1.0, (320, 180), Arc::new(|| {}));
        assert_eq!(p.size, (320, 180));
        let started = Instant::now();
        while p.shared.frame.lock().unwrap().is_none() && started.elapsed().as_secs() < 5 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(p.shared.frame.lock().unwrap().as_ref().map(|f| f.bytes.len()), Some(320 * 180 * 3 / 2));
        assert!(p.playing() && p.position() >= 1.0);
        p.pause();
        let at = p.position();
        std::thread::sleep(std::time::Duration::from_millis(200));
        assert_eq!(p.position(), at, "paused: it stays");
        assert!(p.child.is_some(), "paused: ffmpeg waits");
        let child = p.child.as_ref().map(Child::id);
        p.play();
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(p.child.as_ref().map(Child::id), child, "played on: the same ffmpeg");
        assert!(p.position() > at && p.position() < at + 0.2, "played on from where it was, at once: {at} -> {}", p.position());
        p.pause();
        p.seek(2.5);
        assert!(!p.playing(), "a seek while paused stays paused");
        let started = Instant::now();
        while p.shared.frame.lock().unwrap().is_none() && started.elapsed().as_secs() < 5 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(p.shared.frame.lock().unwrap().is_some(), "its frame shows");
        assert_eq!(p.position(), 2.5);
        assert_eq!(p.shared.fps.get().map(|f| f.0), Some(25.0));
        p.play();
        std::thread::sleep(std::time::Duration::from_millis(400));
        let moved = p.position() - 2.5;
        assert!((0.3..0.5).contains(&moved), "at the video's speed: 0.4 s on, it moved {moved}");
        std::fs::remove_dir_all(&temp).unwrap();
    }
}
