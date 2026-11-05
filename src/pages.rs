//! Where every level of every photo lives on the GPU: one BC7 texture array of PAGE x PAGE pages.
//! Level L holds photos in slots of 2^L px (1 px ... 4096 px, 4k is the most that is kept), in a grid of PAGE / 2^L slots per row.
//! Levels 0 ... 5 (1 px ... 32 px) are baked for every photo by bake.rs (pages.bin, slot = photo id). Levels 0 ... 4 (1 px ...
//! 16 px) are always loaded; from RESIDENT (32 px) up, levels are streamed for the photos on screen into a fixed budget of pages,
//! slots handed out by stream.rs: 32 px copied from the baked file, from BAKED (64 px) up decoded from the originals.
//! The shader repeats the same address maths.

pub const PAGE: u32 = 4096;
pub const LEVELS: usize = 13;
/// Levels always loaded: 1 ... 16 px (about 350 bytes a photo). The first streamed level is this one (32 px).
pub const RESIDENT: usize = 5;
/// Levels in pages.bin: 1 ... 32 px (about 1.4 KB a photo). From here on (64 px) levels are decoded from the originals.
pub const BAKED: usize = 6;
/// Pages per streamed level (32, 64, 128, 256, 512, 1024, 2048, 4096 px): 32768, 8192, 2048, 1024, 512, 256, 8, 3 slots
/// (a 4k picture is a whole page: 16.8 MB).
const STREAM_PAGES: [u32; LEVELS - RESIDENT] = [2, 2, 2, 4, 8, 16, 2, 3];

pub fn slot_size(level: usize) -> u32 {
    1 << level
}

pub fn per_row(level: usize) -> u32 {
    PAGE >> level
}

pub fn per_page(level: usize) -> u32 {
    per_row(level) * per_row(level)
}

/// Pages of a baked level for `photos` photos.
fn baked_pages(level: usize, photos: usize) -> usize {
    photos.div_ceil(per_page(level) as usize)
}

/// Bytes of the baked levels for `photos` photos (BC7: a byte per pixel, whole pages): the size of pages.bin.
pub fn baked_bytes(photos: usize) -> usize {
    (0..BAKED).map(|l| baked_pages(l, photos)).sum::<usize>() * (PAGE as usize).pow(2)
}

/// Where photo `id`'s baked picture of `level` starts in pages.bin (byte), and the bytes from one row of its BC7 blocks to
/// the next: a page is PAGE / 4 blocks of 16 bytes a row.
pub fn baked_at(level: usize, photos: usize, id: usize) -> (usize, usize) {
    let page = (0..level).map(|l| baked_pages(l, photos)).sum::<usize>() + id / per_page(level) as usize;
    let local = id % per_page(level) as usize;
    let (x, y) = (local % per_row(level) as usize * slot_size(level) as usize, local / per_row(level) as usize * slot_size(level) as usize);
    let row = PAGE as usize / 4 * 16;
    (page * (PAGE as usize).pow(2) + y / 4 * row + x / 4 * 16, row)
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
                l => STREAM_PAGES[l - RESIDENT],
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

    /// Memory grows only with the always-loaded levels (about 350 bytes a photo); the streamed pools are 39 pages of 16.8 MB,
    /// whatever the count.
    #[test]
    fn memory_is_bounded() {
        for photos in [1_000, 88_133, 1_000_000] {
            let layout = Layout::new(photos);
            let resident: u32 = layout.pages[..RESIDENT].iter().sum();
            assert_eq!(layout.total_pages, resident + 39, "{photos} photos");
            assert_eq!(baked_bytes(photos), (resident as usize + photos.div_ceil(16384)) * 4096 * 4096);
        }
        assert_eq!(Layout::new(1000).total_pages, 5 + 39);
        assert_eq!(Layout::new(1_800_000).total_pages, 39 + 39); // 32 px of every photo would be 110 more
        let layout = Layout::new(1000);
        assert_eq!((layout.slots(5), layout.slots(6), layout.slots(11), layout.slots(12)), (32768, 8192, 8, 3));
    }

    /// A baked picture's place in the file: after the smaller levels' pages, on its page by its slot.
    #[test]
    fn baked_pictures_are_found() {
        let photos = 20_000; // level 5: 16384 a page, so two pages
        let before = (1 + 1 + 1 + 1 + 2) * 4096 * 4096; // levels 0 ... 4, then level 5's first page
        assert_eq!(baked_at(5, photos, 0), (before - 4096 * 4096, 16384));
        assert_eq!(baked_at(5, photos, 1), (before - 4096 * 4096 + 8 * 16, 16384)); // the next slot: 32 px = 8 blocks on
        assert_eq!(baked_at(5, photos, 128).0, before - 4096 * 4096 + 8 * 16384); // the next row of slots: 8 block rows down
        assert_eq!(baked_at(5, photos, 16384).0, before);
    }
}
