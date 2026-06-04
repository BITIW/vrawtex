use crate::lanczos::{self, ResizeStats};
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use std::error::Error;

const MIPCHAIN_META_KIND: &str = "vrawtex.mipchain";
const MIPCHAIN_META_VERSION: u8 = 1;
const MIPCHAIN_PAD: u32 = 1;

#[derive(Clone, Debug)]
pub struct MipChainSpec {
    additional_levels: Option<usize>,
    explicit_heights: Vec<u32>,
}

impl MipChainSpec {
    pub fn from_cli(
        mipchain: Option<usize>,
        explicit_heights: Vec<u32>,
    ) -> Result<Option<Self>, Box<dyn Error>> {
        let Some(levels) = mipchain else {
            if explicit_heights.is_empty() {
                return Ok(None);
            }
            return Err("--size requires --mipchain".into());
        };

        let additional_levels = (levels != 0).then_some(levels);
        if let Some(levels) = additional_levels
            && !explicit_heights.is_empty()
            && explicit_heights.len() != levels
        {
            return Err(format!(
                "--mipchain {levels} requires exactly {levels} values in --size, got {}",
                explicit_heights.len()
            )
            .into());
        }

        Ok(Some(Self {
            additional_levels,
            explicit_heights,
        }))
    }

    pub fn additional_levels(&self) -> Option<usize> {
        self.additional_levels
    }

    pub fn has_explicit_heights(&self) -> bool {
        !self.explicit_heights.is_empty()
    }

    pub fn target_heights(&self, original_height: u32) -> Result<Vec<u32>, Box<dyn Error>> {
        if !self.explicit_heights.is_empty() {
            let mut previous = original_height;
            for (index, &height) in self.explicit_heights.iter().enumerate() {
                if height == 0 {
                    return Err(
                        format!("--size value #{} must be greater than 0", index + 1).into(),
                    );
                }
                if height >= previous {
                    return Err(format!(
                        "--size values must strictly decrease and stay below mip0 height {}; value #{} is {}",
                        original_height,
                        index + 1,
                        height
                    )
                    .into());
                }
                previous = height;
            }
            return Ok(self.explicit_heights.clone());
        }

        let mut heights = Vec::new();
        let mut height = original_height;
        while height > 1
            && self
                .additional_levels
                .is_none_or(|limit| heights.len() < limit)
        {
            height = (height / 2).max(1);
            heights.push(height);
        }

        if let Some(requested) = self.additional_levels
            && heights.len() != requested
        {
            return Err(format!(
                "requested {requested} mip level(s), but mip0 height {} only supports {} distinct level(s) down to 1",
                original_height,
                heights.len()
            )
            .into());
        }

        Ok(heights)
    }

    pub fn target_dimensions(
        &self,
        original_width: u32,
        original_height: u32,
    ) -> Result<Vec<(u32, u32)>, Box<dyn Error>> {
        if !self.explicit_heights.is_empty() {
            return Ok(self
                .target_heights(original_height)?
                .into_iter()
                .map(|height| {
                    let width = ((original_width as u64 * height as u64
                        + original_height as u64 / 2)
                        / original_height as u64)
                        .max(1)
                        .min(original_width as u64) as u32;
                    (width, height)
                })
                .collect());
        }

        let mut dimensions = Vec::new();
        let (mut width, mut height) = (original_width, original_height);
        while (width > 1 || height > 1)
            && self
                .additional_levels
                .is_none_or(|limit| dimensions.len() < limit)
        {
            width = (width / 2).max(1);
            height = (height / 2).max(1);
            dimensions.push((width, height));
        }

        if let Some(requested) = self.additional_levels
            && dimensions.len() != requested
        {
            return Err(format!(
                "requested {requested} mip level(s), but mip0 {}x{} only supports {} distinct level(s) down to 1x1",
                original_width,
                original_height,
                dimensions.len()
            )
            .into());
        }

        Ok(dimensions)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MipLevelRect {
    pub level: u16,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MipChainMeta {
    pub kind: String,
    pub version: u8,
    pub pad: u16,
    pub levels: Vec<MipLevelRect>,
}

pub struct MipChainAtlas {
    pub image: RgbaImage,
    pub meta: MipChainMeta,
    pub meta_bytes: Vec<u8>,
    pub resize_stats: ResizeStats,
}

#[derive(Clone, Copy)]
struct Placement {
    level: u16,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

struct Layout {
    width: u32,
    height: u32,
    placements: Vec<Placement>,
}

fn checked_outer(value: u32, pad: u32) -> Result<u32, Box<dyn Error>> {
    value
        .checked_add(pad.checked_mul(2).ok_or("mipchain padding overflow")?)
        .ok_or_else(|| "mipchain dimension overflow".into())
}

fn layout_right(levels: &[(u32, u32)], pad: u32) -> Result<Layout, Box<dyn Error>> {
    let (base_w, base_h) = levels[0];
    let base_outer_w = checked_outer(base_w, pad)?;
    let base_outer_h = checked_outer(base_h, pad)?;

    let mut lower_max_w = 0u32;
    let mut lower_total_h = 0u32;
    for &(w, h) in &levels[1..] {
        lower_max_w = lower_max_w.max(checked_outer(w, pad)?);
        lower_total_h = lower_total_h
            .checked_add(checked_outer(h, pad)?)
            .ok_or("mipchain height overflow")?;
    }

    let width = base_outer_w
        .checked_add(lower_max_w)
        .ok_or("mipchain width overflow")?;
    let height = base_outer_h.max(lower_total_h);
    let mut placements = Vec::with_capacity(levels.len());
    placements.push(Placement {
        level: 0,
        x: pad,
        y: pad,
        w: base_w,
        h: base_h,
    });

    let mut y = 0u32;
    for (level_idx, &(w, h)) in levels.iter().enumerate().skip(1) {
        placements.push(Placement {
            level: level_idx as u16,
            x: base_outer_w + pad,
            y: y + pad,
            w,
            h,
        });
        y = y
            .checked_add(checked_outer(h, pad)?)
            .ok_or("mipchain placement overflow")?;
    }

    Ok(Layout {
        width,
        height,
        placements,
    })
}

fn layout_bottom(levels: &[(u32, u32)], pad: u32) -> Result<Layout, Box<dyn Error>> {
    let (base_w, base_h) = levels[0];
    let base_outer_w = checked_outer(base_w, pad)?;
    let base_outer_h = checked_outer(base_h, pad)?;

    let mut lower_total_w = 0u32;
    let mut lower_max_h = 0u32;
    for &(w, h) in &levels[1..] {
        lower_total_w = lower_total_w
            .checked_add(checked_outer(w, pad)?)
            .ok_or("mipchain width overflow")?;
        lower_max_h = lower_max_h.max(checked_outer(h, pad)?);
    }

    let width = base_outer_w.max(lower_total_w);
    let height = base_outer_h
        .checked_add(lower_max_h)
        .ok_or("mipchain height overflow")?;
    let mut placements = Vec::with_capacity(levels.len());
    placements.push(Placement {
        level: 0,
        x: pad,
        y: pad,
        w: base_w,
        h: base_h,
    });

    let mut x = 0u32;
    for (level_idx, &(w, h)) in levels.iter().enumerate().skip(1) {
        placements.push(Placement {
            level: level_idx as u16,
            x: x + pad,
            y: base_outer_h + pad,
            w,
            h,
        });
        x = x
            .checked_add(checked_outer(w, pad)?)
            .ok_or("mipchain placement overflow")?;
    }

    Ok(Layout {
        width,
        height,
        placements,
    })
}

fn layout_score(layout: &Layout) -> (u32, u64, u32) {
    (
        layout.width.max(layout.height),
        layout.width as u64 * layout.height as u64,
        layout.width.saturating_add(layout.height),
    )
}

fn blit_with_padding(atlas: &mut RgbaImage, level: &RgbaImage, placement: Placement, pad: u32) {
    let atlas_w = atlas.width() as usize;
    let src_w = placement.w as usize;
    let src_h = placement.h as usize;
    let x = placement.x as usize;
    let y = placement.y as usize;
    let pad = pad as usize;
    let src = level.as_raw();
    let dst: &mut [u8] = atlas.as_flat_samples_mut().samples;
    let pixel_index = |px: usize, py: usize| (py * atlas_w + px) * 4;

    for row in 0..src_h {
        let src_start = row * src_w * 4;
        let dst_start = pixel_index(x, y + row);
        dst[dst_start..dst_start + src_w * 4]
            .copy_from_slice(&src[src_start..src_start + src_w * 4]);
    }

    for offset in 1..=pad {
        let top_src = pixel_index(x, y);
        let bottom_src = pixel_index(x, y + src_h - 1);
        let top_dst = pixel_index(x, y - offset);
        let bottom_dst = pixel_index(x, y + src_h - 1 + offset);
        dst.copy_within(top_src..top_src + src_w * 4, top_dst);
        dst.copy_within(bottom_src..bottom_src + src_w * 4, bottom_dst);

        for row in 0..src_h {
            let left_src = pixel_index(x, y + row);
            let right_src = pixel_index(x + src_w - 1, y + row);
            let left = [
                dst[left_src],
                dst[left_src + 1],
                dst[left_src + 2],
                dst[left_src + 3],
            ];
            let right = [
                dst[right_src],
                dst[right_src + 1],
                dst[right_src + 2],
                dst[right_src + 3],
            ];
            let left_dst = pixel_index(x - offset, y + row);
            let right_dst = pixel_index(x + src_w - 1 + offset, y + row);
            dst[left_dst..left_dst + 4].copy_from_slice(&left);
            dst[right_dst..right_dst + 4].copy_from_slice(&right);
        }
    }

    for dy in 1..=pad {
        for dx in 1..=pad {
            for corner_y in [y - dy, y + src_h - 1 + dy] {
                let source_y = if corner_y < y { y } else { y + src_h - 1 };
                for corner_x in [x - dx, x + src_w - 1 + dx] {
                    let source_x = if corner_x < x { x } else { x + src_w - 1 };
                    let source = pixel_index(source_x, source_y);
                    let color = [
                        dst[source],
                        dst[source + 1],
                        dst[source + 2],
                        dst[source + 3],
                    ];
                    let target = pixel_index(corner_x, corner_y);
                    dst[target..target + 4].copy_from_slice(&color);
                }
            }
        }
    }
}

fn meta_from_layout(layout: &Layout, pad: u32) -> Result<MipChainMeta, Box<dyn Error>> {
    let pad = u16::try_from(pad).map_err(|_| "mipchain pad does not fit u16")?;
    let levels = layout
        .placements
        .iter()
        .map(|p| {
            Ok(MipLevelRect {
                level: p.level,
                x: u16::try_from(p.x).map_err(|_| "mipchain x does not fit u16")?,
                y: u16::try_from(p.y).map_err(|_| "mipchain y does not fit u16")?,
                w: u16::try_from(p.w).map_err(|_| "mipchain width does not fit u16")?,
                h: u16::try_from(p.h).map_err(|_| "mipchain height does not fit u16")?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    Ok(MipChainMeta {
        kind: MIPCHAIN_META_KIND.to_string(),
        version: MIPCHAIN_META_VERSION,
        pad,
        levels,
    })
}

pub fn encode_meta(meta: &MipChainMeta) -> Result<Vec<u8>, Box<dyn Error>> {
    Ok(rmp_serde::to_vec_named(meta)?)
}

pub fn decode_meta(bytes: &[u8]) -> Result<MipChainMeta, Box<dyn Error>> {
    let meta: MipChainMeta = rmp_serde::from_slice(bytes)?;
    if meta.kind != MIPCHAIN_META_KIND {
        return Err("not vrawtex mipchain metadata".into());
    }
    if meta.version != MIPCHAIN_META_VERSION {
        return Err(format!("unsupported mipchain metadata version: {}", meta.version).into());
    }
    if meta.levels.is_empty() {
        return Err("mipchain metadata contains no levels".into());
    }
    Ok(meta)
}

pub fn build_single_atlas(
    src: &RgbaImage,
    max_side: u32,
    spec: &MipChainSpec,
) -> Result<MipChainAtlas, Box<dyn Error>> {
    let mut level_sizes = vec![src.dimensions()];
    level_sizes.extend(spec.target_dimensions(src.width(), src.height())?);

    let right = layout_right(&level_sizes, MIPCHAIN_PAD)?;
    let bottom = layout_bottom(&level_sizes, MIPCHAIN_PAD)?;
    let layout = if layout_score(&right) <= layout_score(&bottom) {
        right
    } else {
        bottom
    };

    if layout.width > max_side || layout.height > max_side {
        return Err(format!(
            "mipchain atlas {}x{} exceeds supported max side {}",
            layout.width, layout.height, max_side
        )
        .into());
    }

    let mut image = RgbaImage::from_pixel(layout.width, layout.height, Rgba([0, 0, 0, 0]));
    blit_with_padding(&mut image, src, layout.placements[0], MIPCHAIN_PAD);

    let mut previous: Option<RgbaImage> = None;
    let mut resize_stats = ResizeStats {
        taps_total: 0,
        ops_total: 0,
    };
    for (level_idx, &(dst_w, dst_h)) in level_sizes.iter().enumerate().skip(1) {
        let source = previous.as_ref().unwrap_or(src);
        let (next, stats) = lanczos::resize_lanczos_rgba_percent(source, dst_w, dst_h, 100);
        resize_stats.taps_total = resize_stats.taps_total.saturating_add(stats.taps_total);
        resize_stats.ops_total = resize_stats.ops_total.saturating_add(stats.ops_total);
        blit_with_padding(
            &mut image,
            &next,
            layout.placements[level_idx],
            MIPCHAIN_PAD,
        );
        previous = Some(next);
    }

    let meta = meta_from_layout(&layout, MIPCHAIN_PAD)?;
    let meta_bytes = encode_meta(&meta)?;
    Ok(MipChainAtlas {
        image,
        meta,
        meta_bytes,
        resize_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mipchain_atlas_contains_full_chain_and_roundtrips_meta() {
        let src = RgbaImage::from_pixel(8, 4, Rgba([10, 20, 30, 255]));
        let spec = MipChainSpec::from_cli(Some(0), Vec::new())
            .unwrap()
            .unwrap();
        let built = build_single_atlas(&src, 64, &spec).unwrap();
        let dims: Vec<(u16, u16)> = built.meta.levels.iter().map(|m| (m.w, m.h)).collect();

        assert_eq!(dims, vec![(8, 4), (4, 2), (2, 1), (1, 1)]);
        assert!(built.image.width() <= 64);
        assert!(built.image.height() <= 64);
        assert_eq!(decode_meta(&built.meta_bytes).unwrap().levels.len(), 4);
    }

    #[test]
    fn explicit_heights_preserve_aspect_ratio() {
        let spec = MipChainSpec::from_cli(Some(3), vec![50, 25, 10])
            .unwrap()
            .unwrap();
        assert_eq!(
            spec.target_dimensions(200, 100).unwrap(),
            vec![(100, 50), (50, 25), (20, 10)]
        );
    }
}
