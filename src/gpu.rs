//! wgpu setup and the photo renderer: the whole atlas lives in VRAM, every photo is drawn in one instanced call.

use crate::data::MapData;
use crate::stream::Streamer;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window).unwrap();
        // the laptop also has an integrated GPU: prefer the discrete one, it holds the atlas
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let adapter = adapters
            .into_iter()
            .filter(|a| a.is_surface_supported(&surface))
            .min_by_key(|a| match a.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                _ => 2,
            })
            .expect("no GPU that can draw to this window");
        eprintln!("GPU: {}", adapter.get_info().name);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::TEXTURE_COMPRESSION_BC, // the atlas is BC7
            required_limits: adapter.limits(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        }))
        .unwrap();
        let caps = surface.get_capabilities(&adapter);
        // non-sRGB target: the atlas holds sRGB bytes and they are passed through unchanged
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            color_space: wgpu::SurfaceColorSpace::Auto,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        Self { device, queue, surface, config }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub center: [f32; 2],
    pub viewport: [f32; 2],
    pub scale: f32,
    pub size_px: f32,
    /// How far the photos have slid from their old places (`from`) to their new ones: 0 ... 1, 1 when nothing moves.
    pub slide: f32,
    /// A match's copy beside the world (beside.rs): its longer side on screen.
    pub copy_px: f32,
}

pub struct MapRenderer {
    pipeline: wgpu::RenderPipeline,
    /// The same, drawing the matches' copies (map.wgsl's vs_copy).
    copy_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    /// The same with the pixels taken as they are (nearest), not blended with their neighbours.
    bind_group_nearest: wgpu::BindGroup,
    camera: wgpu::Buffer,
    flags: wgpu::Buffer,
    /// Per photo: x, y and shape (where it is going, while the photos slide to another view).
    points: wgpu::Buffer,
    /// Per photo: x, y where it was before the view changed.
    from: wgpu::Buffer,
    /// The matches' copies beside the world: x, y, the photo's number (its bits), unused; and how many there are now.
    copies: wgpu::Buffer,
    n_copies: u32,
    n: u32,
}

impl MapRenderer {
    pub fn new(gpu: &Gpu, data: &MapData, streamer: &Streamer) -> Self {
        let device = &gpu.device;
        let n = data.n();
        let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points"),
            contents: bytemuck::cast_slice(&points_with_shapes(data)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let from = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("from"),
            contents: bytemuck::cast_slice(&data.points),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let copies = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copies"),
            size: (n.max(1) * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let flags = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flags"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // pixels: one texture array of pages, owned by the streamer; slots sit at exact texel spots, so no mipmaps
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("map"),
            source: wgpu::ShaderSource::Wgsl(include_str!("map.wgsl").into()),
        });
        let entry = |binding, visibility, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None };
        let storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("map"),
            entries: &[
                entry(0, vf, wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }),
                entry(1, wgpu::ShaderStages::VERTEX, storage),
                entry(2, vf, storage),
                entry(3, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                }),
                entry(4, wgpu::ShaderStages::FRAGMENT, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(5, wgpu::ShaderStages::FRAGMENT, storage),
                entry(6, wgpu::ShaderStages::FRAGMENT, storage),
                entry(7, wgpu::ShaderStages::VERTEX, storage),
                entry(8, wgpu::ShaderStages::VERTEX, storage),
            ],
        });
        let bind_group_with = |sampler: &wgpu::Sampler| device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("map"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: points.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: flags.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&streamer.view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: streamer.table.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: streamer.first_pages.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: from.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: copies.as_entire_binding() },
            ],
        });
        let (bind_group, bind_group_nearest) = (bind_group_with(&sampler), bind_group_with(&nearest));
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("map"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |vertex: &str| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(vertex),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(vertex),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), // antialiased edges (map.wgsl)
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let (pipeline, copy_pipeline) = (pipeline("vs_main"), pipeline("vs_copy"));
        Self { pipeline, copy_pipeline, bind_group, bind_group_nearest, camera, flags, points, from, copies, n_copies: 0, n: n as u32 }
    }

    /// The matches' copies beside the world: photo `photos[k]` at `at[k]`.
    pub fn set_copies(&mut self, queue: &wgpu::Queue, photos: &[usize], at: &[crate::data::Point]) {
        let copies: Vec<[f32; 4]> = photos.iter().zip(at).map(|(&i, p)| [p.x, p.y, f32::from_bits(i as u32), 0.0]).collect();
        if !copies.is_empty() {
            queue.write_buffer(&self.copies, 0, bytemuck::cast_slice(&copies));
        }
        self.n_copies = copies.len() as u32;
    }

    /// The photos move from `from` to where `data` has them now (they slide there as `CameraUniform::slide` goes from 0 to 1).
    pub fn move_points(&self, queue: &wgpu::Queue, data: &MapData, from: &[crate::data::Point]) {
        queue.write_buffer(&self.points, 0, bytemuck::cast_slice(&points_with_shapes(data)));
        queue.write_buffer(&self.from, 0, bytemuck::cast_slice(from));
    }

    /// The whole flags table (per photo: how red the search makes it, 0..255), re-sent when anything in it changed.
    pub fn upload_flags(&self, queue: &wgpu::Queue, flags: &[u32]) {
        queue.write_buffer(&self.flags, 0, bytemuck::cast_slice(flags));
    }

    /// `smooth`: pixels blended with their neighbours when a photo is shown bigger or smaller than it is stored; else each screen
    /// pixel takes the nearest one.
    pub fn draw(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, cam: CameraUniform, smooth: bool) {
        queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(&cam));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("map"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(BACKGROUND), store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, if smooth { &self.bind_group } else { &self.bind_group_nearest }, &[]);
        pass.draw(0..6, 0..self.n);
        if self.n_copies > 0 {
            pass.set_pipeline(&self.copy_pipeline);
            pass.draw(0..6, 0..self.n_copies);
        }
    }
}

/// Per photo: x, y, and its shape (1, 0.67 for a 3:2 photo).
fn points_with_shapes(data: &MapData) -> Vec<[f32; 4]> {
    (0..data.n())
        .map(|i| {
            let (p, shape) = (data.points[i], data.shape(i));
            [p.x, p.y, shape.x, shape.y]
        })
        .collect()
}

/// The viewer's background (photo.rs fills transparent pixels with the same colour).
const BACKGROUND: wgpu::Color = wgpu::Color { r: 0.047, g: 0.051, b: 0.063, a: 1.0 };

/// Just the background, when no map is open.
pub fn clear(encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(BACKGROUND), store: wgpu::StoreOp::Store },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}

#[cfg(test)]
mod tests {
    /// map.wgsl is only compiled when a window opens: parse and validate it here, so a slip shows in the tests.
    #[test]
    fn shader_is_valid() {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("map.wgsl")).expect("map.wgsl does not parse");
        wgpu::naga::valid::Validator::new(Default::default(), wgpu::naga::valid::Capabilities::all()).validate(&module).expect("map.wgsl is not valid");
    }
}
