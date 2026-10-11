//! Where every level of every photo lives on the GPU: one BC7 texture array of PAGE x PAGE pages.
//! Level L holds photos in slots of 2^L px (1 px ... 1024 px), in a grid of PAGE / 2^L slots per row.
//! Levels below STREAMED are always loaded (slot = photo id, baked by make/bake.rs); the rest are streamed into
//! a fixed budget of pages, slots handed out by stream.rs. The shader repeats the same address maths.

pub const PAGE: u32 = 4096;
pub const LEVELS: usize = 11;
/// First streamed level (128 px); levels below it are always loaded.
pub const STREAMED: usize = 7;
/// Pages per streamed level (128, 256, 512, 1024 px): 2048, 1024, 512, 256 slots.
const STREAM_PAGES: [u32; LEVELS - STREAMED] = [2, 4, 8, 16];

pub fn slot_size(level: usize) -> u32 {
    1 << level
}

pub fn per_row(level: usize) -> u32 {
    PAGE >> level
}

pub fn per_page(level: usize) -> u32 {
    per_row(level) * per_row(level)
}

pub struct Layout {
    pub first_page: [u32; LEVELS],
    pub pages: [u32; LEVELS],
    pub total_pages: u32,
}

impl Layout {
    pub fn new(photos: usize) -> Self {
        let mut first_page = [0; LEVELS];
        let mut pages = [0; LEVELS];
        let mut next = 0;
        for level in 0..LEVELS {
            first_page[level] = next;
            pages[level] = if level < STREAMED { (photos as u32).div_ceil(per_page(level)) } else { STREAM_PAGES[level - STREAMED] };
            next += pages[level];
        }
        Self { first_page, pages, total_pages: next }
    }

    pub fn slots(&self, level: usize) -> u32 {
        self.pages[level] * per_page(level)
    }

    /// Page (array layer) and top-left pixel of a slot.
    pub fn place(&self, level: usize, slot: u32) -> (u32, u32, u32) {
        let local = slot % per_page(level);
        let s = slot_size(level);
        (self.first_page[level] + slot / per_page(level), local % per_row(level) * s, local / per_row(level) * s)
    }
}
