//! Playing a video in its tile (no sound): ffmpeg decodes it (on the graphics card's decoder when there is one, `-hwaccel
//! auto`) from a time at its own speed (`-re`) to raw NV12 frames at the video's own size (zoomed out it is rarely
//! watched, and refitting to the tile on screen lost sharpness), a thread keeps the newest one, and the window uploads it when it is new and turns it into colours on the card
//! (video.wgsl): 1.5 bytes a pixel instead of RGBA's 4, and no converting on the processor. Pausing stops ffmpeg; playing again,
//! or going to another time, starts it there again.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;


/// A frame: brightness (a byte a pixel), then the two colour channels interleaved at half the size each way (NV12).
pub struct Frame {
    size: (u32, u32),
    bytes: Vec<u8>,
}

/// What the reader thread hands over: the newest frame, when the first came (the clock starts there, not while ffmpeg starts
/// up), and whether the video has run out.
#[derive(Default)]
struct Shared {
    frame: Mutex<Option<Frame>>,
    first: OnceLock<Instant>,
    ended: AtomicBool,
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
    /// HD colours (BT.709), else SD (BT.601): videos rarely say, so by size, as players do.
    hd: bool,
    /// Where it was when it last started or stopped, and whether it plays.
    at: f32,
    since: Option<Instant>,
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
        let mut player = Self { photo, copy, duration, ffmpeg: ffmpeg.to_path_buf(), video: video.to_path_buf(), size, hd, at: at.clamp(0.0, duration), since: None, child: None, shared: Arc::new(Shared::default()), wake, shown: false };
        player.play();
        player
    }

    /// Where it is (seconds).
    pub fn position(&self) -> f32 {
        let t = self.at + self.since.and(self.shared.first.get()).map_or(0.0, |s| s.elapsed().as_secs_f32());
        if self.shared.ended.load(Ordering::Relaxed) { self.duration } else { t.min(self.duration) }
    }

    pub fn playing(&self) -> bool {
        self.since.is_some() && !self.shared.ended.load(Ordering::Relaxed)
    }

    /// Start ffmpeg where it is (from the start again once it has run out).
    pub fn play(&mut self) {
        self.stop_ffmpeg();
        if self.position() >= self.duration - 0.05 {
            self.at = 0.0;
        }
        self.shared = Arc::new(Shared::default());
        let (w, h) = self.size;
        let child = Command::new(&self.ffmpeg)
            .args(["-nostdin", "-loglevel", "error", "-hwaccel", "auto", "-re", "-ss"])
            .arg(format!("{:.3}", self.at))
            .arg("-i")
            .arg(&self.video)
            .args(["-an", "-vf"])
            .arg(format!("scale={w}:{h}:out_range=tv"))
            .args(["-f", "rawvideo", "-pix_fmt", "nv12", "pipe:1"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = child else {
            self.shared.ended.store(true, Ordering::Relaxed);
            return;
        };
        let mut out = child.stdout.take().expect("piped");
        let (shared, wake) = (self.shared.clone(), self.wake.clone());
        let size = (w * h * 3 / 2) as usize;
        std::thread::spawn(move || {
            loop {
                let mut bytes = vec![0u8; size];
                if out.read_exact(&mut bytes).is_err() {
                    break;
                }
                shared.first.get_or_init(Instant::now);
                *shared.frame.lock().unwrap() = Some(Frame { size: (w, h), bytes });
                wake();
            }
            shared.ended.store(true, Ordering::Relaxed);
            wake();
        });
        self.child = Some(child);
        self.since = Some(Instant::now());
    }

    pub fn pause(&mut self) {
        self.at = self.position();
        self.since = None;
        self.stop_ffmpeg();
        // the stopped ffmpeg's reader sees its output end: that is not the video running out (play would start it over)
        self.shared = Arc::new(Shared::default());
    }

    /// Go to `t` seconds, playing on if it was playing.
    pub fn seek(&mut self, t: f32) {
        let playing = self.playing();
        self.pause();
        self.shared.ended.store(false, Ordering::Relaxed);
        self.at = t.clamp(0.0, self.duration);
        if playing {
            self.play();
        }
    }

    fn stop_ffmpeg(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if self.since.is_some() && self.shared.ended.load(Ordering::Relaxed) {
            self.at = self.duration;
        }
    }

    /// Drawing its newest frame over `rect` (points), corners rounded by `corner`: uploaded first when a new one came. None
    /// until the first one has.
    pub fn paint(&mut self, rect: egui::Rect, corner: f32) -> Option<egui::Shape> {
        let frame = self.shared.frame.lock().unwrap().take();
        self.shown |= frame.is_some();
        self.shown.then(|| egui_wgpu::Callback::new_paint_callback(rect, Paint { frame: Mutex::new(frame), rect, corner, hd: self.hd }).into())
    }
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
        assert!(p.child.is_none());
        std::thread::sleep(std::time::Duration::from_millis(200)); // the stopped ffmpeg's output has ended by now
        p.play();
        assert!(p.position() >= at && p.position() < at + 0.1, "played again: on from where it was, {at} -> {}", p.position());
        p.pause();
        p.seek(2.5);
        assert_eq!(p.position(), 2.5);
        std::fs::remove_dir_all(&temp).unwrap();
    }
}
