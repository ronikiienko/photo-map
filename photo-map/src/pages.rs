//! Where every level of every photo lives on the GPU: one BC7 texture array of PAGE x PAGE pages.
//! Level L holds photos in slots of 2^L px (1 px ... 4096 px, 4k is the most that is kept), in a grid of PAGE / 2^L slots per row.
//! Levels 0 ... 5 (1 px ... 32 px) are always loaded for every photo (slot = photo id, baked by bake.rs). Levels from STREAMED
//! (64 px) up are streamed for the photos on screen into a fixed budget of pages, slots handed out by stream.rs.
//! The shader repeats the same address maths.

pub const PAGE: u32 = 4096;
pub const LEVELS: usize = 13;
/// Levels always loaded: 1 ... 32 px (about 1.4 KB a photo).
pub const RESIDENT: usize = 6;
/// First streamed level (64 px).
pub const STREAMED: usize = 6;
/// Pages per streamed level (64, 128, 256, 512, 1024, 2048, 4096 px): 8192, 2048, 1024, 512, 256, 8, 3 slots
/// (a 4k picture is a whole page: 16.8 MB).
const STREAM_PAGES: [u32; LEVELS - STREAMED] = [2, 2, 4, 8, 16, 2, 3];

pub fn slot_size(level: usize) -> u32 {
    1 << level
}

pub fn per_row(level: usize) -> u32 {
    PAGE >> level
}

pub fn per_page(level: usize) -> u32 {
    per_row(level) * per_row(level)
}

/// Bytes of the always-loaded levels for `photos` photos (BC7: a byte per pixel, whole pages): the size of pages_bc7.bin.
pub fn resident_bytes(photos: usize) -> usize {
    (0..RESIDENT).map(|l| photos.div_ceil(per_page(l) as usize)).sum::<usize>() * (PAGE as usize).pow(2)
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
            pages[level] = match level {
                l if l < RESIDENT => (photos as u32).div_ceil(per_page(l)),
                l => STREAM_PAGES[l - STREAMED],
            };
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's table entry (map.wgsl ENTRY) holds the header and a slot for every level; 4k is the top level.
    #[test]
    fn table_entry_fits_every_level() {
        assert_eq!(crate::stream::ENTRY, 17);
        assert_eq!(slot_size(LEVELS - 1), 4096);
        assert_eq!(per_page(LEVELS - 1), 1);
    }

    /// Memory grows only with the always-loaded levels (about 1.4 KB a photo); the streamed pools are 37 pages of 16.8 MB, whatever the count.
    #[test]
    fn memory_is_bounded() {
        for photos in [1_000, 88_133, 1_000_000] {
            let layout = Layout::new(photos);
            let resident: u32 = layout.pages[..RESIDENT].iter().sum();
            assert_eq!(layout.total_pages, resident + 37, "{photos} photos");
            assert_eq!(resident_bytes(photos), resident as usize * 4096 * 4096);
        }
        assert_eq!(Layout::new(1000).total_pages, 6 + 37);
        let layout = Layout::new(1000);
        assert_eq!((layout.slots(6), layout.slots(11), layout.slots(12)), (8192, 8, 3));
    }
}
