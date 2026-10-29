//! What a map takes on disk, by kind of file, for the Maps page: how much, and what each part is for.

use std::path::Path;

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct Sizes {
    /// The photos copied into the map.
    pub photos: u64,
    /// Photos that are links to the originals on the same disk: they take no space of their own.
    pub linked: u64,
    /// Photos on another disk that the map links to by path (symbolic links): they take no space here either.
    pub elsewhere: u64,
    /// Of those, how many can't be reached now (the disk is not plugged in, or the photo was moved).
    pub unreachable: usize,
    pub pictures: u64,
    pub vectors: u64,
    pub captions: u64,
    pub layout: u64,
    pub other: u64,
}

/// One row of the breakdown.
pub struct Part {
    pub name: &'static str,
    pub bytes: u64,
    /// What it is, in a sentence or two.
    pub about: &'static str,
}

impl Sizes {
    /// What the map adds to the disk (linked photos are not counted: they are the originals, here or on another disk).
    pub fn total(&self) -> u64 {
        self.photos + self.pictures + self.vectors + self.captions + self.layout + self.other
    }

    /// The parts that take space, biggest first.
    pub fn parts(&self) -> Vec<Part> {
        let mut parts = vec![
            Part { name: "Photos", bytes: self.photos, about: "The pictures themselves, at full size, so you can zoom in to see them sharp. Photos on the same disk as the map are linked instead of copied, and the map points at photos on another disk where they are: see below." },
            Part { name: "Small pictures", bytes: self.pictures, about: "Tiny versions of every photo (1 to 32 pixels wide), compressed and kept ready. They are what you see when the map is zoomed out, and they are why the whole map is always visible at once." },
            Part { name: "Photo vectors", bytes: self.vectors, about: "For each photo, 768 numbers that describe what it shows, made by the AI model. Photos with similar numbers look alike: this is what the layout and the search are built on." },
            Part { name: "Caption vectors", bytes: self.captions, about: "The same 768 numbers for each photo's caption (the text), so a search can match the words as well as the picture." },
            Part { name: "Layout and names", bytes: self.layout, about: "Where each photo sits on the map, the named regions you see when zooming, and each photo's file name and size." },
            Part { name: "Other", bytes: self.other, about: "The cover picture shown on this page, and leftovers." },
        ];
        parts.retain(|p| p.bytes > 0);
        parts.sort_by_key(|p| std::cmp::Reverse(p.bytes));
        parts
    }
}

/// Add up a map's files. Walks the folder once; the originals are one stat per photo, so a big map takes a moment.
pub fn measure(map: &Path) -> Sizes {
    let mut sizes = Sizes::default();
    let Ok(entries) = std::fs::read_dir(map) else { return sizes };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        let name = entry.file_name().to_string_lossy().into_owned();
        if meta.is_dir() {
            if name == "originals" {
                for photo in std::fs::read_dir(entry.path()).into_iter().flatten().flatten() {
                    // DirEntry::metadata doesn't follow a link by path: the link itself
                    let Ok(m) = photo.metadata() else { continue };
                    if m.is_symlink() {
                        match std::fs::metadata(photo.path()) {
                            Ok(target) => sizes.elsewhere += target.len(),
                            Err(_) => sizes.unreachable += 1,
                        }
                    } else if links(&m) > 1 {
                        sizes.linked += m.len();
                    } else {
                        sizes.photos += m.len();
                    }
                }
            }
            continue;
        }
        let bytes = meta.len();
        match name.as_str() {
            "pages_bc7.bin" => sizes.pictures += bytes,
            "embeddings.bin" => sizes.vectors += bytes,
            "captions.bin" => sizes.captions += bytes,
            "layout.bin" | "photos.json" | "manifest.json" => sizes.layout += bytes,
            n if n.starts_with("regions") || n.starts_with("region_names") => sizes.layout += bytes,
            _ => sizes.other += bytes,
        }
    }
    sizes
}

#[cfg(unix)]
fn links(m: &std::fs::Metadata) -> u64 {
    std::os::unix::fs::MetadataExt::nlink(m)
}

#[cfg(not(unix))]
fn links(_: &std::fs::Metadata) -> u64 {
    1
}

/// 1.2 GB, 340 MB, 12 KB.
pub fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} KB", b / 1e3)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_up_the_files() {
        let dir = std::env::temp_dir().join(format!("photo-map-storage-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("originals")).unwrap();
        std::fs::write(dir.join("originals/a.jpg"), vec![0u8; 1000]).unwrap();
        std::fs::hard_link(dir.join("originals/a.jpg"), std::env::temp_dir().join(format!("photo-map-storage-test-{}.jpg", std::process::id()))).unwrap();
        std::fs::write(dir.join("originals/b.jpg"), vec![0u8; 500]).unwrap();
        let away = std::env::temp_dir().join(format!("photo-map-storage-test-{}-away.jpg", std::process::id()));
        std::fs::write(&away, vec![0u8; 2000]).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&away, dir.join("originals/c.jpg")).unwrap();
            std::os::unix::fs::symlink(dir.join("gone.jpg"), dir.join("originals/d.jpg")).unwrap();
        }
        std::fs::write(dir.join("pages_bc7.bin"), vec![0u8; 300]).unwrap();
        std::fs::write(dir.join("embeddings.bin"), vec![0u8; 200]).unwrap();
        std::fs::write(dir.join("layout.bin"), vec![0u8; 10]).unwrap();
        std::fs::write(dir.join("region_names.json"), vec![0u8; 5]).unwrap();
        std::fs::write(dir.join("cover.png"), vec![0u8; 7]).unwrap();
        let s = measure(&dir);
        assert_eq!((s.photos, s.linked, s.pictures, s.vectors, s.layout, s.other), (500, 1000, 300, 200, 15, 7));
        assert_eq!(s.total(), 1022);
        #[cfg(unix)]
        assert_eq!((s.elsewhere, s.unreachable), (2000, 1));
        std::fs::remove_file(&away).unwrap();
        assert_eq!(human(1_234_000_000), "1.2 GB");
        std::fs::remove_dir_all(&dir).unwrap();
        std::fs::remove_file(std::env::temp_dir().join(format!("photo-map-storage-test-{}.jpg", std::process::id()))).unwrap();
    }
}
