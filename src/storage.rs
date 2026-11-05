//! What a map takes on disk, by kind of file, for the Maps page: how much, and what each part is for.

use std::path::Path;

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct Sizes {
    pub pictures: u64,
    /// The videos' posters and small frames (frames/).
    pub frames: u64,
    pub vectors: u64,
    /// The videos' sound: a vector for each 10 s of it, and where each starts.
    pub sounds: u64,
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
    /// What the map takes on disk (the photos are not in it: it reads them where they are).
    pub fn total(&self) -> u64 {
        self.pictures + self.frames + self.vectors + self.sounds + self.layout + self.other
    }

    /// The parts that take space, biggest first.
    pub fn parts(&self) -> Vec<Part> {
        let mut parts = vec![
            Part { name: "Small pictures", bytes: self.pictures, about: "Tiny versions of every photo (1 to 32 pixels wide), compressed and kept ready. They are what you see when the map is zoomed out, and they are why the whole map is always visible at once." },
            Part { name: "Video frames", bytes: self.frames, about: "For each video, its poster (the picture it has on the map) and its keyframes, small (256 pixels): a search shows a video by the frame that matched best." },
            Part { name: "Photo vectors", bytes: self.vectors, about: "For each photo (and each frame of a video), 768 numbers that describe what it shows, made by the AI model. Photos with similar numbers look alike: this is what the layout and the search are built on." },
            Part { name: "Sound vectors", bytes: self.sounds, about: "For every 10 seconds of a video's sound, 768 numbers that describe what is said or heard in it, made by the sound model: this is what searching by sound is built on." },
            Part { name: "Layout and names", bytes: self.layout, about: "Where each photo sits on the map, the named regions you see when zooming, and each photo's file, size and what it says about itself." },
            Part { name: "Other", bytes: self.other, about: "The cover picture shown on this page, and leftovers." },
        ];
        parts.retain(|p| p.bytes > 0);
        parts.sort_by_key(|p| std::cmp::Reverse(p.bytes));
        parts
    }
}

/// Add up a map's files. Walks the folder once (the frames are one stat each).
pub fn measure(map: &Path) -> Sizes {
    let mut sizes = Sizes::default();
    let Ok(entries) = std::fs::read_dir(map) else { return sizes };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        let name = entry.file_name().to_string_lossy().into_owned();
        if meta.is_dir() {
            if name == "frames" {
                for video in std::fs::read_dir(entry.path()).into_iter().flatten().flatten() {
                    for f in std::fs::read_dir(video.path()).into_iter().flatten().flatten() {
                        sizes.frames += f.metadata().map_or(0, |m| m.len());
                    }
                }
            }
            continue;
        }
        let bytes = meta.len();
        match name.as_str() {
            "pages.bin" => sizes.pictures += bytes,
            "look.bin" => sizes.vectors += bytes,
            "sound.bin" => sizes.sounds += bytes,
            "layout.bin" | "tree.bin" | "items.json" | "regions.json" | "manifest.json" => sizes.layout += bytes,
            _ => sizes.other += bytes,
        }
    }
    sizes
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
        std::fs::create_dir_all(dir.join("frames/3")).unwrap();
        std::fs::write(dir.join("pages.bin"), vec![0u8; 300]).unwrap();
        std::fs::write(dir.join("look.bin"), vec![0u8; 200]).unwrap();
        std::fs::write(dir.join("layout.bin"), vec![0u8; 10]).unwrap();
        std::fs::write(dir.join("items.json"), vec![0u8; 3]).unwrap();
        std::fs::write(dir.join("regions.json"), vec![0u8; 5]).unwrap();
        std::fs::write(dir.join("cover.png"), vec![0u8; 7]).unwrap();
        std::fs::write(dir.join("sound.bin"), vec![0u8; 30]).unwrap();
        // a video's poster and frames
        std::fs::write(dir.join("frames/3/poster.jpg"), vec![0u8; 60]).unwrap();
        std::fs::write(dir.join("frames/3/0.jpg"), vec![0u8; 20]).unwrap();
        let s = measure(&dir);
        assert_eq!((s.pictures, s.frames, s.vectors, s.sounds, s.layout, s.other), (300, 80, 200, 30, 18, 7));
        assert_eq!(s.total(), 635);
        assert_eq!(human(1_234_000_000), "1.2 GB");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
