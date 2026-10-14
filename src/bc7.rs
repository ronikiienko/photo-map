//! BC7 texture compression (the small pictures the map is drawn with), in Rust: `rusty_dds`' encoder.

use rusty_dds::{DecodeContent, Dds, EncodeLayout};

/// Compress `rgba` (width x height pixels, both multiples of 4) into BC7 blocks: 16 bytes for each 4 x 4 pixels, rows of blocks
/// one after the other.
pub fn compress(rgba: &[u8], width: u32, height: u32) -> Vec<u8> {
    let dds = Dds::encode_from_rgba8(rgba, EncodeLayout::flat_2d(DecodeContent::Bc7, width, height)).expect("BC7 compression of whole blocks");
    dds.data
}

/// The pixels BC7 blocks stand for (for tests and measurements).
#[cfg(test)]
pub fn decompress(blocks: &[u8], width: u32, height: u32) -> Vec<u8> {
    use rusty_dds::SubresourceId;
    let mut dds = Dds::encode_from_rgba8(&vec![0u8; (width * height * 4) as usize], EncodeLayout::flat_2d(DecodeContent::Bc7, width, height)).unwrap();
    dds.data.copy_from_slice(blocks);
    dds.decode_rgba8(SubresourceId::new(0, 0, 0)).unwrap().pixels
}
