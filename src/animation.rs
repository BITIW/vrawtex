use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;

pub(crate) const MAGIC: [u8; 8] = *b"VRAWANM\0";
const CONTAINER_VERSION: u16 = 2;
const HEADER_VERSION: u16 = 2;
const MIN_CONTAINER_VERSION: u16 = 1;
const MIN_HEADER_VERSION: u16 = 1;
const PREAMBLE_LEN: usize = 8 + 2 + 8;
const KIND: &str = "vrawtex.animation";
const VIEWER_PREFETCH_TARGET: Duration = Duration::from_secs(2);
const VIEWER_PREFETCH_MAX_BYTES: usize = 256 * 1024 * 1024;
const VIEWER_REFERENCE_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;
const RECT_MAX_AREA_BPS: u64 = 6_000;
const MOTION_SEARCH_RADIUS: i32 = 8;
const MOTION_MIN_GAIN_BPS: u64 = 1_000;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnimationFrameCoding {
    /// A full-size keyframe or same-coordinate wrapping delta.
    #[default]
    Full,
    /// An exact copy of the reference frame with no embedded blob.
    Copy,
    /// A wrapping-delta rectangle applied over the reference frame.
    Rect,
    /// A full-size residual predicted from a globally shifted reference frame.
    Motion,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimationRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationFrameHeader {
    /// Zero-based frame index.
    pub index: u32,
    /// Frame duration numerator in milliseconds.
    pub duration_num_ms: u32,
    /// Frame duration denominator in milliseconds.
    pub duration_den_ms: u32,
    /// Earlier reconstructed frame used by this delta, or `None` for a keyframe.
    pub reference: Option<u32>,
    /// Spatial interpretation of the embedded VRAWTEX blob.
    #[serde(default)]
    pub coding: AnimationFrameCoding,
    /// Changed rectangle for `rect` coding.
    #[serde(default)]
    pub rect: Option<AnimationRect>,
    /// Signed global reference displacement for `motion` coding.
    #[serde(default)]
    pub motion: Option<[i16; 2]>,
    /// Absolute byte offset of the embedded VRAWTEX blob.
    pub offset: u64,
    /// Embedded VRAWTEX blob length in bytes.
    pub len: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationHeader {
    /// Schema identifier, currently `vrawtex.animation`.
    pub kind: String,
    /// MessagePack header schema version.
    pub version: u16,
    /// Common frame width.
    pub width: u32,
    /// Common frame height.
    pub height: u32,
    /// VRAWTEX pixel-format identifier used by every frame blob.
    pub pixfmt: u16,
    /// Stored channel count used by every frame blob.
    pub channels: u8,
    /// Number of playback loops; zero means infinite.
    pub loop_count: u32,
    /// Absolute offset where embedded frame blobs begin.
    pub blob_section_offset: u64,
    /// Timing, reference, and byte range for each frame.
    pub frames: Vec<AnimationFrameHeader>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnimationPixels {
    /// Interleaved RGBA8; RGB containers receive synthetic alpha 255.
    U8(Vec<u8>),
    /// Interleaved RGBA16; RGB containers receive synthetic alpha 65,535.
    U16(Vec<u16>),
}

#[derive(Clone, Debug)]
pub struct DecodedAnimationFrame {
    /// Frame duration numerator in milliseconds.
    pub duration_num_ms: u32,
    /// Frame duration denominator in milliseconds.
    pub duration_den_ms: u32,
    /// Fully reconstructed interleaved pixels.
    pub pixels: AnimationPixels,
}

#[derive(Clone, Debug)]
pub struct DecodedAnimation {
    /// Parsed container metadata.
    pub header: AnimationHeader,
    /// Fully reconstructed frames in playback order.
    pub frames: Vec<DecodedAnimationFrame>,
}

pub(crate) fn is_animation(data: &[u8]) -> bool {
    data.starts_with(&MAGIC)
}

fn u8_delta(current: &RgbaImage, reference: &RgbaImage) -> RgbaImage {
    let bytes = current
        .as_raw()
        .iter()
        .zip(reference.as_raw())
        .map(|(current, reference)| current.wrapping_sub(*reference))
        .collect();
    RgbaImage::from_raw(current.width(), current.height(), bytes).expect("delta dimensions")
}

fn to_u16(image: &RgbaImage) -> Rgba16Image {
    Rgba16Image::from_raw(
        image.width(),
        image.height(),
        image
            .as_raw()
            .iter()
            .map(|sample| *sample as u16 * 257)
            .collect(),
    )
    .expect("U16 conversion dimensions")
}

fn u16_delta(current: &Rgba16Image, reference: &Rgba16Image) -> Rgba16Image {
    let samples = current
        .as_raw()
        .iter()
        .zip(reference.as_raw())
        .map(|(current, reference)| current.wrapping_sub(*reference))
        .collect();
    Rgba16Image::from_raw(current.width(), current.height(), samples).expect("delta dimensions")
}

struct FramePlan8 {
    pixels: Option<RgbaImage>,
    reference: Option<usize>,
    coding: AnimationFrameCoding,
    rect: Option<AnimationRect>,
    motion: Option<[i16; 2]>,
}

struct FramePlan16 {
    pixels: Option<Rgba16Image>,
    reference: Option<usize>,
    coding: AnimationFrameCoding,
    rect: Option<AnimationRect>,
    motion: Option<[i16; 2]>,
}

fn estimate_u8_candidate(
    image: &RgbaImage,
    pixel_format: EncodePixelFormat,
) -> Result<usize, Box<dyn Error>> {
    let (transform, _, choices) = choose_color_transform_and_predictors(
        image.as_raw(),
        image.width() as usize,
        image.height() as usize,
    )?;
    let mut size = choices
        .iter()
        .find(|choice| choice.transform == transform)
        .ok_or("missing animation transform estimate")?
        .total_size;
    if pixel_format.has_alpha() {
        match detect_alpha_mode(image.as_raw()) {
            AlphaMode::Normal => {
                size = size.saturating_add(
                    choose_predictor_for_channel_sample(
                        image.as_raw(),
                        image.width() as usize,
                        image.height() as usize,
                        3,
                    )?
                    .chosen_size,
                );
            }
            AlphaMode::Mask1Bit => {
                size = size
                    .saturating_add((image.width() as usize * image.height() as usize).div_ceil(8));
            }
            AlphaMode::Opaque255 | AlphaMode::Transparent0 => {}
        }
    }
    Ok(size)
}

fn changed_rect_u8(
    current: &RgbaImage,
    reference: &RgbaImage,
    channels: usize,
) -> Option<AnimationRect> {
    let (width, height) = current.dimensions();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for (index, (current, reference)) in current
        .as_raw()
        .chunks_exact(4)
        .zip(reference.as_raw().chunks_exact(4))
        .enumerate()
    {
        if current[..channels] == reference[..channels] {
            continue;
        }
        let x = index as u32 % width;
        let y = index as u32 / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        found = true;
    }
    found.then(|| AnimationRect {
        x: min_x,
        y: min_y,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    })
}

fn changed_rect_u16(
    current: &Rgba16Image,
    reference: &Rgba16Image,
    channels: usize,
) -> Option<AnimationRect> {
    let (width, height) = current.dimensions();
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;
    for (index, (current, reference)) in current
        .as_raw()
        .chunks_exact(4)
        .zip(reference.as_raw().chunks_exact(4))
        .enumerate()
    {
        if current[..channels] == reference[..channels] {
            continue;
        }
        let x = index as u32 % width;
        let y = index as u32 / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
        found = true;
    }
    found.then(|| AnimationRect {
        x: min_x,
        y: min_y,
        width: max_x - min_x + 1,
        height: max_y - min_y + 1,
    })
}

fn rect_is_sparse(rect: AnimationRect, width: u32, height: u32) -> bool {
    let rect_area = rect.width as u64 * rect.height as u64;
    let full_area = width as u64 * height as u64;
    rect_area.saturating_mul(10_000) <= full_area.saturating_mul(RECT_MAX_AREA_BPS)
}

fn rect_delta_u8(current: &RgbaImage, reference: &RgbaImage, rect: AnimationRect) -> RgbaImage {
    RgbaImage::from_fn(rect.width, rect.height, |x, y| {
        let current = current.get_pixel(rect.x + x, rect.y + y).0;
        let reference = reference.get_pixel(rect.x + x, rect.y + y).0;
        Rgba([
            current[0].wrapping_sub(reference[0]),
            current[1].wrapping_sub(reference[1]),
            current[2].wrapping_sub(reference[2]),
            current[3].wrapping_sub(reference[3]),
        ])
    })
}

fn rect_delta_u16(
    current: &Rgba16Image,
    reference: &Rgba16Image,
    rect: AnimationRect,
) -> Rgba16Image {
    Rgba16Image::from_fn(rect.width, rect.height, |x, y| {
        let current = current.get_pixel(rect.x + x, rect.y + y).0;
        let reference = reference.get_pixel(rect.x + x, rect.y + y).0;
        Rgba([
            current[0].wrapping_sub(reference[0]),
            current[1].wrapping_sub(reference[1]),
            current[2].wrapping_sub(reference[2]),
            current[3].wrapping_sub(reference[3]),
        ])
    })
}

fn motion_score_u8(
    current: &RgbaImage,
    reference: &RgbaImage,
    channels: usize,
    dx: i32,
    dy: i32,
    target_samples: usize,
    abort_at: u64,
) -> u64 {
    let (width, height) = current.dimensions();
    let pixels = width as usize * height as usize;
    let stride = (pixels / target_samples.max(1)).max(1);
    let current = current.as_raw();
    let reference = reference.as_raw();
    let mut score = 0u64;
    for pixel in (0..pixels).step_by(stride) {
        let x = (pixel % width as usize) as i32;
        let y = (pixel / width as usize) as i32;
        let source_x = x + dx;
        let source_y = y + dy;
        let current_offset = pixel * 4;
        if source_x < 0 || source_y < 0 || source_x >= width as i32 || source_y >= height as i32 {
            for channel in 0..channels {
                score = score.saturating_add(current[current_offset + channel] as u64);
            }
        } else {
            let source_offset = (source_y as usize * width as usize + source_x as usize) * 4;
            for channel in 0..channels {
                score = score.saturating_add(
                    current[current_offset + channel].abs_diff(reference[source_offset + channel])
                        as u64,
                );
            }
        }
        if score >= abort_at {
            break;
        }
    }
    score
}

fn motion_score_u16(
    current: &Rgba16Image,
    reference: &Rgba16Image,
    channels: usize,
    dx: i32,
    dy: i32,
    target_samples: usize,
    abort_at: u64,
) -> u64 {
    let (width, height) = current.dimensions();
    let pixels = width as usize * height as usize;
    let stride = (pixels / target_samples.max(1)).max(1);
    let current = current.as_raw();
    let reference = reference.as_raw();
    let mut score = 0u64;
    for pixel in (0..pixels).step_by(stride) {
        let x = (pixel % width as usize) as i32;
        let y = (pixel / width as usize) as i32;
        let source_x = x + dx;
        let source_y = y + dy;
        let current_offset = pixel * 4;
        if source_x < 0 || source_y < 0 || source_x >= width as i32 || source_y >= height as i32 {
            for channel in 0..channels {
                score = score.saturating_add(current[current_offset + channel] as u64);
            }
        } else {
            let source_offset = (source_y as usize * width as usize + source_x as usize) * 4;
            for channel in 0..channels {
                score = score.saturating_add(
                    current[current_offset + channel].abs_diff(reference[source_offset + channel])
                        as u64,
                );
            }
        }
        if score >= abort_at {
            break;
        }
    }
    score
}

fn find_motion<F>(mut score: F) -> Option<[i16; 2]>
where
    F: FnMut(i32, i32, usize, u64) -> u64,
{
    let zero_score = score(0, 0, 4096, u64::MAX);
    if zero_score == 0 {
        return None;
    }
    let mut coarse = (0, 0, u64::MAX);
    for dy in (-MOTION_SEARCH_RADIUS..=MOTION_SEARCH_RADIUS).step_by(2) {
        for dx in (-MOTION_SEARCH_RADIUS..=MOTION_SEARCH_RADIUS).step_by(2) {
            let candidate = score(dx, dy, 1024, coarse.2);
            if candidate < coarse.2 {
                coarse = (dx, dy, candidate);
            }
        }
    }
    let mut best = (coarse.0, coarse.1, u64::MAX);
    for dy in (coarse.1 - 1).max(-MOTION_SEARCH_RADIUS)..=(coarse.1 + 1).min(MOTION_SEARCH_RADIUS) {
        for dx in
            (coarse.0 - 1).max(-MOTION_SEARCH_RADIUS)..=(coarse.0 + 1).min(MOTION_SEARCH_RADIUS)
        {
            let candidate = score(dx, dy, 4096, best.2);
            if candidate < best.2 {
                best = (dx, dy, candidate);
            }
        }
    }
    let enough_gain =
        best.2.saturating_mul(10_000) <= zero_score.saturating_mul(10_000 - MOTION_MIN_GAIN_BPS);
    (enough_gain && (best.0, best.1) != (0, 0)).then_some([best.0 as i16, best.1 as i16])
}

fn motion_delta_u8(
    current: &RgbaImage,
    reference: &RgbaImage,
    motion: [i16; 2],
    channels: usize,
) -> RgbaImage {
    let (width, height) = current.dimensions();
    RgbaImage::from_fn(width, height, |x, y| {
        let source_x = x as i32 + motion[0] as i32;
        let source_y = y as i32 + motion[1] as i32;
        let predicted = if source_x >= 0
            && source_y >= 0
            && source_x < width as i32
            && source_y < height as i32
        {
            reference.get_pixel(source_x as u32, source_y as u32).0
        } else {
            [0; 4]
        };
        let current = current.get_pixel(x, y).0;
        let mut out = [255u8; 4];
        for channel in 0..channels {
            out[channel] = current[channel].wrapping_sub(predicted[channel]);
        }
        Rgba(out)
    })
}

fn motion_delta_u16(
    current: &Rgba16Image,
    reference: &Rgba16Image,
    motion: [i16; 2],
    channels: usize,
) -> Rgba16Image {
    let (width, height) = current.dimensions();
    Rgba16Image::from_fn(width, height, |x, y| {
        let source_x = x as i32 + motion[0] as i32;
        let source_y = y as i32 + motion[1] as i32;
        let predicted = if source_x >= 0
            && source_y >= 0
            && source_x < width as i32
            && source_y < height as i32
        {
            reference.get_pixel(source_x as u32, source_y as u32).0
        } else {
            [0; 4]
        };
        let current = current.get_pixel(x, y).0;
        let mut out = [u16::MAX; 4];
        for channel in 0..channels {
            out[channel] = current[channel].wrapping_sub(predicted[channel]);
        }
        Rgba(out)
    })
}

fn choose_frame_plan_u8(
    current: &RgbaImage,
    first: Option<&RgbaImage>,
    previous: Option<&RgbaImage>,
    index: usize,
    pixel_format: EncodePixelFormat,
) -> Result<FramePlan8, Box<dyn Error>> {
    let Some(previous) = previous else {
        return Ok(FramePlan8 {
            pixels: Some(current.clone()),
            reference: None,
            coding: AnimationFrameCoding::Full,
            rect: None,
            motion: None,
        });
    };
    let channels = pixel_format.channels() as usize;
    let Some(rect) = changed_rect_u8(current, previous, channels) else {
        return Ok(FramePlan8 {
            pixels: None,
            reference: Some(index - 1),
            coding: AnimationFrameCoding::Copy,
            rect: None,
            motion: None,
        });
    };

    let mut best_score = estimate_u8_candidate(current, pixel_format)?;
    let mut best = FramePlan8 {
        pixels: Some(current.clone()),
        reference: None,
        coding: AnimationFrameCoding::Full,
        rect: None,
        motion: None,
    };
    let mut consider = |pixels: RgbaImage,
                        reference: usize,
                        coding: AnimationFrameCoding,
                        rect: Option<AnimationRect>,
                        motion: Option<[i16; 2]>|
     -> Result<(), Box<dyn Error>> {
        let score = estimate_u8_candidate(&pixels, pixel_format)?;
        if score < best_score {
            best_score = score;
            best = FramePlan8 {
                pixels: Some(pixels),
                reference: Some(reference),
                coding,
                rect,
                motion,
            };
        }
        Ok(())
    };

    if index > 1 {
        consider(
            u8_delta(current, first.ok_or("missing first animation frame")?),
            0,
            AnimationFrameCoding::Full,
            None,
            None,
        )?;
    }
    consider(
        u8_delta(current, previous),
        index - 1,
        AnimationFrameCoding::Full,
        None,
        None,
    )?;
    if rect_is_sparse(rect, current.width(), current.height()) {
        consider(
            rect_delta_u8(current, previous, rect),
            index - 1,
            AnimationFrameCoding::Rect,
            Some(rect),
            None,
        )?;
    }
    let motion = find_motion(|dx, dy, samples, abort| {
        motion_score_u8(current, previous, channels, dx, dy, samples, abort)
    });
    if let Some(motion) = motion {
        consider(
            motion_delta_u8(current, previous, motion, channels),
            index - 1,
            AnimationFrameCoding::Motion,
            None,
            Some(motion),
        )?;
    }
    Ok(best)
}

fn choose_frame_plan_u16(
    current: &Rgba16Image,
    first: Option<&Rgba16Image>,
    previous: Option<&Rgba16Image>,
    index: usize,
    pixel_format: EncodePixelFormat,
) -> Result<FramePlan16, Box<dyn Error>> {
    let Some(previous) = previous else {
        return Ok(FramePlan16 {
            pixels: Some(current.clone()),
            reference: None,
            coding: AnimationFrameCoding::Full,
            rect: None,
            motion: None,
        });
    };
    let channels = pixel_format.channels() as usize;
    let Some(rect) = changed_rect_u16(current, previous, channels) else {
        return Ok(FramePlan16 {
            pixels: None,
            reference: Some(index - 1),
            coding: AnimationFrameCoding::Copy,
            rect: None,
            motion: None,
        });
    };

    let mut best_score = u16_codec::estimate_candidate_size(current, pixel_format)?;
    let mut best = FramePlan16 {
        pixels: Some(current.clone()),
        reference: None,
        coding: AnimationFrameCoding::Full,
        rect: None,
        motion: None,
    };
    let mut consider = |pixels: Rgba16Image,
                        reference: usize,
                        coding: AnimationFrameCoding,
                        rect: Option<AnimationRect>,
                        motion: Option<[i16; 2]>|
     -> Result<(), Box<dyn Error>> {
        let score = u16_codec::estimate_candidate_size(&pixels, pixel_format)?;
        if score < best_score {
            best_score = score;
            best = FramePlan16 {
                pixels: Some(pixels),
                reference: Some(reference),
                coding,
                rect,
                motion,
            };
        }
        Ok(())
    };

    if index > 1 {
        consider(
            u16_delta(current, first.ok_or("missing first animation frame")?),
            0,
            AnimationFrameCoding::Full,
            None,
            None,
        )?;
    }
    consider(
        u16_delta(current, previous),
        index - 1,
        AnimationFrameCoding::Full,
        None,
        None,
    )?;
    if rect_is_sparse(rect, current.width(), current.height()) {
        consider(
            rect_delta_u16(current, previous, rect),
            index - 1,
            AnimationFrameCoding::Rect,
            Some(rect),
            None,
        )?;
    }
    let motion = find_motion(|dx, dy, samples, abort| {
        motion_score_u16(current, previous, channels, dx, dy, samples, abort)
    });
    if let Some(motion) = motion {
        consider(
            motion_delta_u16(current, previous, motion, channels),
            index - 1,
            AnimationFrameCoding::Motion,
            None,
            Some(motion),
        )?;
    }
    Ok(best)
}

fn frame_header(
    index: usize,
    duration_num_ms: u32,
    duration_den_ms: u32,
    reference: Option<usize>,
    coding: AnimationFrameCoding,
    rect: Option<AnimationRect>,
    motion: Option<[i16; 2]>,
) -> AnimationFrameHeader {
    AnimationFrameHeader {
        index: index as u32,
        duration_num_ms,
        duration_den_ms: duration_den_ms.max(1),
        reference: reference.map(|value| value as u32),
        coding,
        rect,
        motion,
        offset: 0,
        len: 0,
    }
}

fn stabilize_header(
    mut header: AnimationHeader,
    blob_lengths: &[usize],
) -> Result<(AnimationHeader, Vec<u8>), Box<dyn Error>> {
    let mut encoded = rmp_serde::to_vec(&header)?;
    for _ in 0..8 {
        let blob_section_offset = (PREAMBLE_LEN + encoded.len()) as u64;
        let mut cursor = blob_section_offset;
        for (frame, length) in header.frames.iter_mut().zip(blob_lengths) {
            frame.offset = cursor;
            frame.len = *length as u64;
            cursor = cursor
                .checked_add(*length as u64)
                .ok_or("animation size overflow")?;
        }
        header.blob_section_offset = blob_section_offset;
        let next = rmp_serde::to_vec(&header)?;
        if next.len() == encoded.len() {
            return Ok((header, next));
        }
        encoded = next;
    }
    Err("failed to stabilize animation header layout".into())
}

fn finish_animation_container(
    header: AnimationHeader,
    pixel_format: EncodePixelFormat,
    blobs: Vec<Vec<u8>>,
    verbose: bool,
    start: Instant,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let lengths = blobs.iter().map(Vec::len).collect::<Vec<_>>();
    let (header, header_bytes) = stabilize_header(header, &lengths)?;
    let total_blob_bytes = lengths
        .iter()
        .try_fold(0usize, |sum, length| sum.checked_add(*length))
        .ok_or("animation output size overflow")?;
    let mut out = Vec::with_capacity(PREAMBLE_LEN + header_bytes.len() + total_blob_bytes);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for blob in blobs {
        out.extend_from_slice(&blob);
    }
    if verbose {
        let keyframes = header
            .frames
            .iter()
            .filter(|frame| frame.reference.is_none())
            .count();
        let full_deltas = header
            .frames
            .iter()
            .filter(|frame| frame.reference.is_some() && frame.coding == AnimationFrameCoding::Full)
            .count();
        let copies = header
            .frames
            .iter()
            .filter(|frame| frame.coding == AnimationFrameCoding::Copy)
            .count();
        let rects = header
            .frames
            .iter()
            .filter(|frame| frame.coding == AnimationFrameCoding::Rect)
            .count();
        let motions = header
            .frames
            .iter()
            .filter(|frame| frame.coding == AnimationFrameCoding::Motion)
            .count();
        println!(
            "[vrawtex] Animation: frames={}, image={}x{}, {}, coding=key/full/copy/rect/motion:{}/{}/{}/{}/{}, blobs={} bytes, total={} bytes, encode={}",
            header.frames.len(),
            header.width,
            header.height,
            pixel_format.as_str(),
            keyframes,
            full_deltas,
            copies,
            rects,
            motions,
            total_blob_bytes,
            out.len(),
            format_duration_ns(start.elapsed())
        );
    }
    Ok(out)
}

pub(crate) fn encode_animation(
    animation: &image_input::LoadedAnimation,
    pixel_format: EncodePixelFormat,
    profile: CompressionProfile,
    verbose: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let first = animation
        .frames
        .first()
        .ok_or("animation contains no frames")?;
    let (width, height) = first.image.dimensions();
    if animation
        .frames
        .iter()
        .any(|frame| frame.image.dimensions() != (width, height))
    {
        return Err("animation frames have inconsistent dimensions".into());
    }
    if animation.frames.len() > u32::MAX as usize {
        return Err("animation contains too many frames".into());
    }

    let start = Instant::now();
    let mut blobs = Vec::with_capacity(animation.frames.len());
    let mut headers = Vec::with_capacity(animation.frames.len());
    if pixel_format.is_16_bit() {
        let frames = animation
            .frames
            .iter()
            .map(|frame| to_u16(&frame.image))
            .collect::<Vec<_>>();
        for index in 0..frames.len() {
            let plan = choose_frame_plan_u16(
                &frames[index],
                frames.first(),
                index.checked_sub(1).map(|previous| &frames[previous]),
                index,
                pixel_format,
            )?;
            blobs.push(match plan.pixels.as_ref() {
                Some(pixels) => u16_codec::encode_rgba16_with_meta_to_vec(
                    pixels,
                    None,
                    pixel_format,
                    profile,
                    false,
                    None,
                    Instant::now(),
                )?,
                None => Vec::new(),
            });
            let source = &animation.frames[index];
            headers.push(frame_header(
                index,
                source.delay_num_ms,
                source.delay_den_ms,
                plan.reference,
                plan.coding,
                plan.rect,
                plan.motion,
            ));
        }
    } else {
        for index in 0..animation.frames.len() {
            let plan = choose_frame_plan_u8(
                &animation.frames[index].image,
                animation.frames.first().map(|frame| &frame.image),
                index
                    .checked_sub(1)
                    .map(|previous| &animation.frames[previous].image),
                index,
                pixel_format,
            )?;
            blobs.push(match plan.pixels.as_ref() {
                Some(pixels) => encode_rgba8_with_meta_to_vec(
                    pixels,
                    None,
                    pixel_format,
                    profile,
                    false,
                    None,
                    Instant::now(),
                )?,
                None => Vec::new(),
            });
            let source = &animation.frames[index];
            headers.push(frame_header(
                index,
                source.delay_num_ms,
                source.delay_den_ms,
                plan.reference,
                plan.coding,
                plan.rect,
                plan.motion,
            ));
        }
    }

    finish_animation_container(
        AnimationHeader {
            kind: KIND.to_owned(),
            version: HEADER_VERSION,
            width,
            height,
            pixfmt: pixel_format.pixfmt_bits(),
            channels: pixel_format.channels(),
            loop_count: animation.loop_count,
            blob_section_offset: 0,
            frames: headers,
        },
        pixel_format,
        blobs,
        verbose,
        start,
    )
}

pub(crate) fn encode_animation16(
    animation: &image_input::LoadedAnimation16,
    pixel_format: EncodePixelFormat,
    profile: CompressionProfile,
    verbose: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if !pixel_format.is_16_bit() {
        return Err("16-bit animation requires RGB16 or RGBA16 output".into());
    }
    let first = animation
        .frames
        .first()
        .ok_or("animation contains no frames")?;
    let (width, height) = first.image.dimensions();
    if animation
        .frames
        .iter()
        .any(|frame| frame.image.dimensions() != (width, height))
    {
        return Err("animation frames have inconsistent dimensions".into());
    }
    if animation.frames.len() > u32::MAX as usize {
        return Err("animation contains too many frames".into());
    }

    let start = Instant::now();
    let frames = animation
        .frames
        .iter()
        .map(|frame| &frame.image)
        .collect::<Vec<_>>();
    let mut blobs = Vec::with_capacity(frames.len());
    let mut headers = Vec::with_capacity(frames.len());
    for index in 0..frames.len() {
        let plan = choose_frame_plan_u16(
            frames[index],
            frames.first().copied(),
            index.checked_sub(1).map(|previous| frames[previous]),
            index,
            pixel_format,
        )?;
        blobs.push(match plan.pixels.as_ref() {
            Some(pixels) => u16_codec::encode_rgba16_with_meta_to_vec(
                pixels,
                None,
                pixel_format,
                profile,
                false,
                None,
                Instant::now(),
            )?,
            None => Vec::new(),
        });
        let source = &animation.frames[index];
        headers.push(frame_header(
            index,
            source.delay_num_ms,
            source.delay_den_ms,
            plan.reference,
            plan.coding,
            plan.rect,
            plan.motion,
        ));
    }
    finish_animation_container(
        AnimationHeader {
            kind: KIND.to_owned(),
            version: HEADER_VERSION,
            width,
            height,
            pixfmt: pixel_format.pixfmt_bits(),
            channels: pixel_format.channels(),
            loop_count: animation.loop_count,
            blob_section_offset: 0,
            frames: headers,
        },
        pixel_format,
        blobs,
        verbose,
        start,
    )
}

pub(crate) fn encode_streamed_animation(
    input: &Path,
    pixel_format: EncodePixelFormat,
    profile: CompressionProfile,
    verbose: bool,
) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
    let Some(mut reader) = image_input::open_streamed_animation(input, pixel_format.is_16_bit())?
    else {
        return Ok(None);
    };
    let (width, height) = reader.dimensions();
    let loop_count = reader.loop_count();
    let frame_count_hint = reader.frame_count_hint();
    let start = Instant::now();
    let mut blobs = Vec::with_capacity(frame_count_hint.unwrap_or(0));
    let mut headers = Vec::with_capacity(frame_count_hint.unwrap_or(0));

    if pixel_format.is_16_bit() {
        let mut first: Option<Rgba16Image> = None;
        let mut previous: Option<Rgba16Image> = None;
        while let Some(frame) = reader.next_frame()? {
            let image_input::StreamAnimationPixels::U16(current) = frame.pixels else {
                return Err("streamed animation bit depth mismatch".into());
            };
            let index = headers.len();
            let plan = choose_frame_plan_u16(
                &current,
                first.as_ref(),
                previous.as_ref(),
                index,
                pixel_format,
            )?;
            blobs.push(match plan.pixels.as_ref() {
                Some(pixels) => u16_codec::encode_rgba16_with_meta_to_vec(
                    pixels,
                    None,
                    pixel_format,
                    profile,
                    false,
                    None,
                    Instant::now(),
                )?,
                None => Vec::new(),
            });
            headers.push(frame_header(
                index,
                frame.delay_num_ms,
                frame.delay_den_ms,
                plan.reference,
                plan.coding,
                plan.rect,
                plan.motion,
            ));
            if first.is_none() {
                first = Some(current.clone());
            }
            previous = Some(current);
            print_stream_progress(verbose, headers.len(), frame_count_hint);
        }
    } else {
        let mut first: Option<RgbaImage> = None;
        let mut previous: Option<RgbaImage> = None;
        while let Some(frame) = reader.next_frame()? {
            let image_input::StreamAnimationPixels::U8(current) = frame.pixels else {
                return Err("streamed animation bit depth mismatch".into());
            };
            let index = headers.len();
            let plan = choose_frame_plan_u8(
                &current,
                first.as_ref(),
                previous.as_ref(),
                index,
                pixel_format,
            )?;
            blobs.push(match plan.pixels.as_ref() {
                Some(pixels) => encode_rgba8_with_meta_to_vec(
                    pixels,
                    None,
                    pixel_format,
                    profile,
                    false,
                    None,
                    Instant::now(),
                )?,
                None => Vec::new(),
            });
            headers.push(frame_header(
                index,
                frame.delay_num_ms,
                frame.delay_den_ms,
                plan.reference,
                plan.coding,
                plan.rect,
                plan.motion,
            ));
            if first.is_none() {
                first = Some(current.clone());
            }
            previous = Some(current);
            print_stream_progress(verbose, headers.len(), frame_count_hint);
        }
    }

    if headers.is_empty() {
        return Err("streamed animation contains no frames".into());
    }
    finish_animation_container(
        AnimationHeader {
            kind: KIND.to_owned(),
            version: HEADER_VERSION,
            width,
            height,
            pixfmt: pixel_format.pixfmt_bits(),
            channels: pixel_format.channels(),
            loop_count,
            blob_section_offset: 0,
            frames: headers,
        },
        pixel_format,
        blobs,
        verbose,
        start,
    )
    .map(Some)
}

fn print_stream_progress(verbose: bool, frames: usize, total: Option<usize>) {
    if verbose && (frames == 1 || frames.is_multiple_of(30) || total == Some(frames)) {
        match total {
            Some(total) => println!("[vrawtex] Animation stream: {frames}/{total} frames"),
            None => println!("[vrawtex] Animation stream: {frames} frames"),
        }
    }
}

pub fn parse_header(data: &[u8]) -> Result<AnimationHeader, Box<dyn Error>> {
    if data.len() < PREAMBLE_LEN || !is_animation(data) {
        return Err("not a VRAWTEX animation".into());
    }
    let version = u16::from_le_bytes([data[8], data[9]]);
    if !(MIN_CONTAINER_VERSION..=CONTAINER_VERSION).contains(&version) {
        return Err(format!("unsupported animation container version: {version}").into());
    }
    let header_len = u64::from_le_bytes(data[10..18].try_into()?) as usize;
    let header_end = PREAMBLE_LEN
        .checked_add(header_len)
        .ok_or("animation header length overflow")?;
    if header_end > data.len() {
        return Err("truncated animation header".into());
    }
    let header: AnimationHeader = rmp_serde::from_slice(&data[PREAMBLE_LEN..header_end])?;
    if header.kind != KIND
        || !(MIN_HEADER_VERSION..=HEADER_VERSION).contains(&header.version)
        || header.version != version
    {
        return Err("unsupported animation header".into());
    }
    if header.blob_section_offset != header_end as u64 || header.frames.is_empty() {
        return Err("invalid animation blob layout".into());
    }
    for (index, frame) in header.frames.iter().enumerate() {
        if frame.index as usize != index || frame.duration_den_ms == 0 {
            return Err("invalid animation frame metadata".into());
        }
        if index == 0 && frame.reference.is_some() {
            return Err("first animation frame cannot be a delta".into());
        }
        if frame
            .reference
            .is_some_and(|reference| reference as usize >= index)
        {
            return Err("animation frame reference must point backwards".into());
        }
        match frame.coding {
            AnimationFrameCoding::Full => {
                if frame.rect.is_some() || frame.motion.is_some() {
                    return Err("full animation frame has spatial side data".into());
                }
            }
            AnimationFrameCoding::Copy => {
                if frame.reference.is_none()
                    || frame.rect.is_some()
                    || frame.motion.is_some()
                    || frame.len != 0
                {
                    return Err("invalid copy animation frame".into());
                }
            }
            AnimationFrameCoding::Rect => {
                let rect = frame.rect.ok_or("rect animation frame has no rectangle")?;
                if frame.reference.is_none()
                    || frame.motion.is_some()
                    || rect.width == 0
                    || rect.height == 0
                    || rect
                        .x
                        .checked_add(rect.width)
                        .is_none_or(|x| x > header.width)
                    || rect
                        .y
                        .checked_add(rect.height)
                        .is_none_or(|y| y > header.height)
                {
                    return Err("invalid rect animation frame".into());
                }
            }
            AnimationFrameCoding::Motion => {
                if frame.reference.is_none() || frame.rect.is_some() || frame.motion.is_none() {
                    return Err("invalid motion animation frame".into());
                }
            }
        }
        if version == 1 && frame.coding != AnimationFrameCoding::Full {
            return Err("animation v1 only supports full frame blobs".into());
        }
        let end = frame
            .offset
            .checked_add(frame.len)
            .ok_or("animation blob overflow")?;
        if frame.offset < header.blob_section_offset || end > data.len() as u64 {
            return Err("animation frame blob exceeds file bounds".into());
        }
    }
    Ok(header)
}

fn interleave_u8(planes: &[Vec<u8>], channels: u8, pixels: usize) -> Vec<u8> {
    let mut rgba = vec![255u8; pixels * 4];
    for index in 0..pixels {
        for channel in 0..channels as usize {
            rgba[index * 4 + channel] = planes[channel][index];
        }
    }
    rgba
}

fn interleave_u16(planes: &[Vec<u16>], channels: u8, pixels: usize) -> Vec<u16> {
    let mut rgba = vec![u16::MAX; pixels * 4];
    for index in 0..pixels {
        for channel in 0..channels as usize {
            rgba[index * 4 + channel] = planes[channel][index];
        }
    }
    rgba
}

impl AnimationPixels {
    fn storage_bytes(&self) -> usize {
        match self {
            AnimationPixels::U8(bytes) => bytes.len(),
            AnimationPixels::U16(samples) => samples.len().saturating_mul(2),
        }
    }
}

fn decode_frame_pixels(
    data: &[u8],
    header: &AnimationHeader,
    frame: &AnimationFrameHeader,
    reference: Option<&AnimationPixels>,
    safety: DecodeSafety,
    pixels: usize,
) -> Result<AnimationPixels, Box<dyn Error>> {
    if frame.coding == AnimationFrameCoding::Copy {
        return reference
            .cloned()
            .ok_or_else(|| "copy animation reference frame is unavailable".into());
    }
    let start = frame.offset as usize;
    let end = start
        .checked_add(frame.len as usize)
        .ok_or("frame range overflow")?;
    let blob = &data[start..end];
    let parsed = parse_container(blob, safety)?;
    let (blob_width, blob_height) = frame
        .rect
        .map(|rect| (rect.width, rect.height))
        .unwrap_or((header.width, header.height));
    if parsed.width != blob_width
        || parsed.height != blob_height
        || parsed.pixfmt_bits != header.pixfmt
        || parsed.chans != header.channels
    {
        return Err("animation frame blob format mismatch".into());
    }

    let blob_pixels = (blob_width as usize)
        .checked_mul(blob_height as usize)
        .ok_or("animation frame blob pixel count overflow")?;

    if parsed.sample_bytes == 1 {
        let (planes, _, _, _) = decode_container_to_planes(&parsed, blob, safety, true)?;
        let mut decoded = Some(interleave_u8(&planes, parsed.chans, blob_pixels));
        let reference: &[u8] = match reference {
            Some(AnimationPixels::U8(reference)) => reference,
            Some(AnimationPixels::U16(_)) => {
                return Err("animation reference bit depth mismatch".into());
            }
            None => &[],
        };
        let mut current = match frame.coding {
            AnimationFrameCoding::Full => decoded.take().unwrap(),
            AnimationFrameCoding::Rect => reference.to_vec(),
            AnimationFrameCoding::Motion => decoded.take().unwrap(),
            AnimationFrameCoding::Copy => unreachable!(),
        };
        match frame.coding {
            AnimationFrameCoding::Full if frame.reference.is_some() => {
                if reference.len() != current.len() {
                    return Err("animation reference frame is unavailable".into());
                }
                for pixel in 0..pixels {
                    for channel in 0..parsed.chans as usize {
                        let sample = pixel * 4 + channel;
                        current[sample] = current[sample].wrapping_add(reference[sample]);
                    }
                }
            }
            AnimationFrameCoding::Rect => {
                let rect = frame.rect.ok_or("missing animation rectangle")?;
                let decoded = decoded
                    .as_ref()
                    .ok_or("missing animation rectangle payload")?;
                for y in 0..rect.height as usize {
                    for x in 0..rect.width as usize {
                        let source = (y * rect.width as usize + x) * 4;
                        let destination =
                            ((rect.y as usize + y) * header.width as usize + rect.x as usize + x)
                                * 4;
                        for channel in 0..parsed.chans as usize {
                            current[destination + channel] = current[destination + channel]
                                .wrapping_add(decoded[source + channel]);
                        }
                    }
                }
            }
            AnimationFrameCoding::Motion => {
                if reference.len() != current.len() {
                    return Err("animation reference frame is unavailable".into());
                }
                let motion = frame.motion.ok_or("missing animation motion vector")?;
                for y in 0..header.height as usize {
                    for x in 0..header.width as usize {
                        let source_x = x as i64 + motion[0] as i64;
                        let source_y = y as i64 + motion[1] as i64;
                        if source_x < 0
                            || source_y < 0
                            || source_x >= header.width as i64
                            || source_y >= header.height as i64
                        {
                            continue;
                        }
                        let destination = (y * header.width as usize + x) * 4;
                        let source =
                            (source_y as usize * header.width as usize + source_x as usize) * 4;
                        for channel in 0..parsed.chans as usize {
                            current[destination + channel] = current[destination + channel]
                                .wrapping_add(reference[source + channel]);
                        }
                    }
                }
            }
            AnimationFrameCoding::Full | AnimationFrameCoding::Copy => {}
        }
        Ok(AnimationPixels::U8(current))
    } else {
        let (planes, _, _, _) =
            u16_codec::decode_container_to_planes_u16(&parsed, blob, safety, true)?;
        let mut decoded = Some(interleave_u16(&planes, parsed.chans, blob_pixels));
        let reference: &[u16] = match reference {
            Some(AnimationPixels::U16(reference)) => reference,
            Some(AnimationPixels::U8(_)) => {
                return Err("animation reference bit depth mismatch".into());
            }
            None => &[],
        };
        let mut current = match frame.coding {
            AnimationFrameCoding::Full => decoded.take().unwrap(),
            AnimationFrameCoding::Rect => reference.to_vec(),
            AnimationFrameCoding::Motion => decoded.take().unwrap(),
            AnimationFrameCoding::Copy => unreachable!(),
        };
        match frame.coding {
            AnimationFrameCoding::Full if frame.reference.is_some() => {
                if reference.len() != current.len() {
                    return Err("animation reference frame is unavailable".into());
                }
                for pixel in 0..pixels {
                    for channel in 0..parsed.chans as usize {
                        let sample = pixel * 4 + channel;
                        current[sample] = current[sample].wrapping_add(reference[sample]);
                    }
                }
            }
            AnimationFrameCoding::Rect => {
                let rect = frame.rect.ok_or("missing animation rectangle")?;
                let decoded = decoded
                    .as_ref()
                    .ok_or("missing animation rectangle payload")?;
                for y in 0..rect.height as usize {
                    for x in 0..rect.width as usize {
                        let source = (y * rect.width as usize + x) * 4;
                        let destination =
                            ((rect.y as usize + y) * header.width as usize + rect.x as usize + x)
                                * 4;
                        for channel in 0..parsed.chans as usize {
                            current[destination + channel] = current[destination + channel]
                                .wrapping_add(decoded[source + channel]);
                        }
                    }
                }
            }
            AnimationFrameCoding::Motion => {
                if reference.len() != current.len() {
                    return Err("animation reference frame is unavailable".into());
                }
                let motion = frame.motion.ok_or("missing animation motion vector")?;
                for y in 0..header.height as usize {
                    for x in 0..header.width as usize {
                        let source_x = x as i64 + motion[0] as i64;
                        let source_y = y as i64 + motion[1] as i64;
                        if source_x < 0
                            || source_y < 0
                            || source_x >= header.width as i64
                            || source_y >= header.height as i64
                        {
                            continue;
                        }
                        let destination = (y * header.width as usize + x) * 4;
                        let source =
                            (source_y as usize * header.width as usize + source_x as usize) * 4;
                        for channel in 0..parsed.chans as usize {
                            current[destination + channel] = current[destination + channel]
                                .wrapping_add(reference[source + channel]);
                        }
                    }
                }
            }
            AnimationFrameCoding::Full | AnimationFrameCoding::Copy => {}
        }
        Ok(AnimationPixels::U16(current))
    }
}

struct StreamingAnimationDecoder<'a> {
    data: &'a [u8],
    header: &'a AnimationHeader,
    safety: DecodeSafety,
    pixels: usize,
    cache: HashMap<usize, Arc<AnimationPixels>>,
    cache_order: VecDeque<usize>,
    cache_bytes: usize,
    cache_limit_bytes: usize,
}

impl<'a> StreamingAnimationDecoder<'a> {
    fn new(
        data: &'a [u8],
        header: &'a AnimationHeader,
        safety: DecodeSafety,
        cache_limit_bytes: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let pixels = (header.width as usize)
            .checked_mul(header.height as usize)
            .ok_or("animation pixel count overflow")?;
        Ok(Self {
            data,
            header,
            safety,
            pixels,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            cache_bytes: 0,
            cache_limit_bytes: cache_limit_bytes.max(1),
        })
    }

    fn cached(&mut self, index: usize) -> Option<Arc<AnimationPixels>> {
        let pixels = self.cache.get(&index)?.clone();
        if let Some(position) = self.cache_order.iter().position(|cached| *cached == index) {
            self.cache_order.remove(position);
        }
        self.cache_order.push_back(index);
        Some(pixels)
    }

    fn insert_cache(&mut self, index: usize, pixels: Arc<AnimationPixels>) {
        if let Some(previous) = self.cache.insert(index, pixels.clone()) {
            self.cache_bytes = self.cache_bytes.saturating_sub(previous.storage_bytes());
        }
        if let Some(position) = self.cache_order.iter().position(|cached| *cached == index) {
            self.cache_order.remove(position);
        }
        self.cache_order.push_back(index);
        self.cache_bytes = self.cache_bytes.saturating_add(pixels.storage_bytes());

        while self.cache_bytes > self.cache_limit_bytes && self.cache.len() > 1 {
            let Some(oldest) = self.cache_order.pop_front() else {
                break;
            };
            if oldest == index {
                self.cache_order.push_back(oldest);
                break;
            }
            if let Some(removed) = self.cache.remove(&oldest) {
                self.cache_bytes = self.cache_bytes.saturating_sub(removed.storage_bytes());
            }
        }
    }

    fn decode(&mut self, index: usize) -> Result<Arc<AnimationPixels>, Box<dyn Error>> {
        if index >= self.header.frames.len() {
            return Err("animation frame index is out of bounds".into());
        }
        if let Some(pixels) = self.cached(index) {
            return Ok(pixels);
        }

        let mut chain = Vec::new();
        let mut cursor = index;
        let mut reference = loop {
            if let Some(pixels) = self.cached(cursor) {
                break Some(pixels);
            }
            chain.push(cursor);
            match self.header.frames[cursor].reference {
                Some(previous) => cursor = previous as usize,
                None => break None,
            }
        };

        for frame_index in chain.into_iter().rev() {
            if self.header.frames[frame_index].coding == AnimationFrameCoding::Copy {
                let pixels = reference
                    .clone()
                    .ok_or("copy animation reference frame is unavailable")?;
                self.insert_cache(frame_index, pixels.clone());
                reference = Some(pixels);
                continue;
            }
            let pixels = Arc::new(decode_frame_pixels(
                self.data,
                self.header,
                &self.header.frames[frame_index],
                reference.as_deref(),
                self.safety,
                self.pixels,
            )?);
            self.insert_cache(frame_index, pixels.clone());
            reference = Some(pixels);
        }
        reference.ok_or_else(|| "failed to reconstruct animation frame".into())
    }
}

pub fn decode_animation(
    data: &[u8],
    safety: DecodeSafety,
) -> Result<DecodedAnimation, Box<dyn Error>> {
    let header = parse_header(data)?;
    let pixels = (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or("animation pixel count overflow")?;
    let mut frames: Vec<DecodedAnimationFrame> = Vec::with_capacity(header.frames.len());
    for frame in &header.frames {
        let reference = frame
            .reference
            .map(|reference| &frames[reference as usize].pixels);
        let pixels_data = decode_frame_pixels(data, &header, frame, reference, safety, pixels)?;
        frames.push(DecodedAnimationFrame {
            duration_num_ms: frame.duration_num_ms,
            duration_den_ms: frame.duration_den_ms,
            pixels: pixels_data,
        });
    }
    Ok(DecodedAnimation { header, frames })
}

fn frames_directory(base: &Path) -> PathBuf {
    let name = base
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("animation");
    base.with_file_name(format!("{name}_frames"))
}

pub(crate) fn decode_to_files(
    data: &[u8],
    input: &Path,
    output: Option<PathBuf>,
    target: DecodeFormat,
    safety: DecodeSafety,
    dump_meta: Option<PathBuf>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let decoded = decode_animation(data, safety)?;
    let base = output.unwrap_or_else(|| default_decode_base_path(input));
    let directory = frames_directory(&base);
    fs::create_dir_all(&directory)?;
    for (index, frame) in decoded.frames.iter().enumerate() {
        match (&frame.pixels, target) {
            (AnimationPixels::U8(bytes), DecodeFormat::Png) => {
                let image =
                    RgbaImage::from_raw(decoded.header.width, decoded.header.height, bytes.clone())
                        .ok_or("invalid decoded RGBA8 animation frame")?;
                image.save(directory.join(format!("frame_{index:06}.png")))?;
            }
            (AnimationPixels::U16(samples), DecodeFormat::Png) => {
                let image = Rgba16Image::from_raw(
                    decoded.header.width,
                    decoded.header.height,
                    samples.clone(),
                )
                .ok_or("invalid decoded RGBA16 animation frame")?;
                image::DynamicImage::ImageRgba16(image)
                    .save(directory.join(format!("frame_{index:06}.png")))?;
            }
            (AnimationPixels::U8(bytes), DecodeFormat::Raw) => {
                fs::write(directory.join(format!("frame_{index:06}.rgba")), bytes)?;
            }
            (AnimationPixels::U16(samples), DecodeFormat::Raw) => {
                let mut bytes = Vec::with_capacity(samples.len() * 2);
                for sample in samples {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                fs::write(directory.join(format!("frame_{index:06}.rgba16le")), bytes)?;
            }
        }
    }
    let metadata_path = dump_meta.unwrap_or_else(|| directory.join("animation.json"));
    fs::write(&metadata_path, serde_json::to_vec_pretty(&decoded.header)?)?;
    println!(
        "Decoded animation {}x{}, {} frames -> {}",
        decoded.header.width,
        decoded.header.height,
        decoded.frames.len(),
        directory.display()
    );
    if verbose {
        println!(
            "[vrawtex] Animation decode: {} bytes, {}, metadata={}",
            data.len(),
            format_duration_ns(start.elapsed()),
            metadata_path.display()
        );
    }
    Ok(())
}

fn frame_duration(frame: &AnimationFrameHeader) -> Duration {
    let nanos = (frame.duration_num_ms as u128)
        .saturating_mul(1_000_000)
        .checked_div(frame.duration_den_ms.max(1) as u128)
        .unwrap_or(0)
        .clamp(1_000_000, u64::MAX as u128) as u64;
    Duration::from_nanos(nanos)
}

fn preview_rgba8<'a>(pixels: &'a AnimationPixels, scratch: &'a mut Vec<u8>) -> &'a [u8] {
    match pixels {
        AnimationPixels::U8(bytes) => bytes,
        AnimationPixels::U16(samples) => {
            scratch.clear();
            scratch.extend(samples.iter().map(|sample| (sample >> 8) as u8));
            scratch
        }
    }
}

fn viewer_frame_bytes(header: &AnimationHeader) -> Result<usize, Box<dyn Error>> {
    let sample_bytes = match header.pixfmt {
        0x0001 => 1usize,
        0x0002 => 2usize,
        pixfmt => return Err(format!("unsupported animation pixel format: 0x{pixfmt:04X}").into()),
    };
    (header.width as usize)
        .checked_mul(header.height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|samples| samples.checked_mul(sample_bytes))
        .ok_or_else(|| "animation viewer frame size overflow".into())
}

fn viewer_prefetch_capacity(
    header: &AnimationHeader,
) -> Result<(usize, Duration, usize), Box<dyn Error>> {
    let frame_bytes = viewer_frame_bytes(header)?;
    let byte_limited_frames = (VIEWER_PREFETCH_MAX_BYTES / frame_bytes.max(1)).max(1);
    let mut duration = Duration::ZERO;
    let mut frames = 0usize;
    for frame in &header.frames {
        if frames >= byte_limited_frames || (frames > 0 && duration >= VIEWER_PREFETCH_TARGET) {
            break;
        }
        duration = duration.saturating_add(frame_duration(frame));
        frames += 1;
    }
    Ok((frames.max(1), duration, frame_bytes))
}

struct ViewerFrame {
    index: usize,
    duration: Duration,
    pixels: Arc<AnimationPixels>,
}

enum ViewerMessage {
    Frame(ViewerFrame),
    Finished,
    Error(String),
}

fn stream_viewer_frames(
    data: &[u8],
    header: &AnimationHeader,
    safety: DecodeSafety,
    sender: mpsc::SyncSender<ViewerMessage>,
    stop: &AtomicBool,
) {
    let mut decoder = match StreamingAnimationDecoder::new(
        data,
        header,
        safety,
        VIEWER_REFERENCE_CACHE_MAX_BYTES,
    ) {
        Ok(decoder) => decoder,
        Err(error) => {
            let _ = sender.send(ViewerMessage::Error(error.to_string()));
            return;
        }
    };
    let mut completed_loops = 0u32;
    loop {
        for (index, frame) in header.frames.iter().enumerate() {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let pixels = match decoder.decode(index) {
                Ok(pixels) => pixels,
                Err(error) => {
                    let _ = sender.send(ViewerMessage::Error(error.to_string()));
                    return;
                }
            };
            if sender
                .send(ViewerMessage::Frame(ViewerFrame {
                    index,
                    duration: frame_duration(frame),
                    pixels,
                }))
                .is_err()
            {
                return;
            }
        }
        completed_loops = completed_loops.saturating_add(1);
        if header.loop_count != 0 && completed_loops >= header.loop_count {
            let _ = sender.send(ViewerMessage::Finished);
            return;
        }
    }
}

fn render_preview(
    rgba: &[u8],
    image_width: usize,
    image_height: usize,
    window_width: usize,
    window_height: usize,
    output: &mut [u32],
) {
    output.fill(0);
    let scale =
        (window_width as f64 / image_width as f64).min(window_height as f64 / image_height as f64);
    let draw_width = (image_width as f64 * scale).round().max(1.0) as usize;
    let draw_height = (image_height as f64 * scale).round().max(1.0) as usize;
    let offset_x = window_width.saturating_sub(draw_width) / 2;
    let offset_y = window_height.saturating_sub(draw_height) / 2;
    for y in 0..draw_height.min(window_height) {
        let source_y = (y * image_height / draw_height).min(image_height - 1);
        for x in 0..draw_width.min(window_width) {
            let source_x = (x * image_width / draw_width).min(image_width - 1);
            let source = (source_y * image_width + source_x) * 4;
            let alpha = rgba[source + 3] as u32;
            let red = rgba[source] as u32 * alpha / 255;
            let green = rgba[source + 1] as u32 * alpha / 255;
            let blue = rgba[source + 2] as u32 * alpha / 255;
            output[(offset_y + y) * window_width + offset_x + x] =
                (red << 16) | (green << 8) | blue;
        }
    }
}

pub(crate) fn open_animation(
    data: &[u8],
    input: &Path,
    safety: DecodeSafety,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let start = Instant::now();
    let header = parse_header(data)?;
    let image_width = header.width as usize;
    let image_height = header.height as usize;
    let (prefetch_frames, prefetch_duration, frame_bytes) = viewer_prefetch_capacity(&header)?;
    let scale = (MAX_WINDOW_WIDTH as f64 / image_width as f64)
        .min(MAX_WINDOW_HEIGHT as f64 / image_height as f64)
        .min(1.0);
    let initial_width = (image_width as f64 * scale).round().max(1.0) as usize;
    let initial_height = (image_height as f64 * scale).round().max(1.0) as usize;
    thread::scope(|scope| -> Result<(), Box<dyn Error>> {
        let (sender, receiver) = mpsc::sync_channel(prefetch_frames);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let worker_header = &header;
        scope
            .spawn(move || stream_viewer_frames(data, worker_header, safety, sender, &worker_stop));

        let mut current = match receiver.recv() {
            Ok(ViewerMessage::Frame(frame)) => frame,
            Ok(ViewerMessage::Error(error)) => return Err(error.into()),
            Ok(ViewerMessage::Finished) => {
                return Err("animation contains no playable frames".into());
            }
            Err(_) => return Err("animation viewer decoder stopped unexpectedly".into()),
        };
        let mut window = Window::new(
            &format!("vrawtex animation: {}", input.display()),
            initial_width,
            initial_height,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )?;
        let mut deadline = Instant::now() + current.duration;
        let mut framebuffer = Vec::new();
        let mut u16_preview = Vec::new();
        let mut previous_size = (0usize, 0usize);
        let mut dirty = true;
        println!(
            "Opened {}x{} animation ({} frames, streaming, ESC to close)",
            image_width,
            image_height,
            header.frames.len()
        );
        if verbose {
            println!(
                "[vrawtex] Viewer ready in {}, prefetch={} frames/{:.3} sec (up to {:.1} MiB), reference_cache_limit={} MiB, loop_count={}",
                format_duration_ns(start.elapsed()),
                prefetch_frames,
                prefetch_duration.as_secs_f64(),
                prefetch_frames.saturating_mul(frame_bytes) as f64 / (1024.0 * 1024.0),
                VIEWER_REFERENCE_CACHE_MAX_BYTES / (1024 * 1024),
                header.loop_count
            );
        }
        while window.is_open() && !window.is_key_down(Key::Escape) {
            let now = Instant::now();
            if now >= deadline {
                match receiver.try_recv() {
                    Ok(ViewerMessage::Frame(frame)) => {
                        current = frame;
                        deadline = now + current.duration;
                        dirty = true;
                    }
                    Ok(ViewerMessage::Finished) => break,
                    Ok(ViewerMessage::Error(error)) => return Err(error.into()),
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        return Err("animation viewer decoder stopped unexpectedly".into());
                    }
                }
            }
            let size = window.get_size();
            if size != previous_size {
                framebuffer.resize(size.0.saturating_mul(size.1), 0);
                previous_size = size;
                dirty = true;
            }
            if dirty && size.0 > 0 && size.1 > 0 {
                let preview = preview_rgba8(&current.pixels, &mut u16_preview);
                render_preview(
                    preview,
                    image_width,
                    image_height,
                    size.0,
                    size.1,
                    &mut framebuffer,
                );
                window.update_with_buffer(&framebuffer, size.0, size.1)?;
                dirty = false;
            } else {
                window.update();
            }
            thread::sleep(Duration::from_millis(1));
        }
        if verbose {
            println!("[vrawtex] Viewer stopped on frame {}", current.index);
        }
        stop.store(true, Ordering::Relaxed);
        drop(receiver);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> image_input::LoadedAnimation {
        let first = RgbaImage::from_fn(8, 4, |x, y| Rgba([(x * 9) as u8, (y * 20) as u8, 7, 255]));
        let mut second = first.clone();
        second.put_pixel(2, 1, Rgba([250, 10, 20, 64]));
        let mut third = second.clone();
        third.put_pixel(3, 1, Rgba([1, 2, 3, 0]));
        image_input::LoadedAnimation {
            frames: vec![
                image_input::AnimationInputFrame {
                    image: first,
                    delay_num_ms: 40,
                    delay_den_ms: 1,
                },
                image_input::AnimationInputFrame {
                    image: second,
                    delay_num_ms: 50,
                    delay_den_ms: 1,
                },
                image_input::AnimationInputFrame {
                    image: third,
                    delay_num_ms: 1000,
                    delay_den_ms: 24,
                },
            ],
            loop_count: 0,
        }
    }

    #[test]
    fn animation_frames_roundtrip_independent_blobs() {
        let source = fixture();
        let encoded = encode_animation(
            &source,
            EncodePixelFormat::Rgba8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(decoded.frames.len(), source.frames.len());
        for (decoded, source) in decoded.frames.iter().zip(&source.frames) {
            let AnimationPixels::U8(pixels) = &decoded.pixels else {
                panic!("expected U8 frame");
            };
            assert_eq!(pixels, source.image.as_raw());
            assert_eq!(decoded.duration_num_ms, source.delay_num_ms);
            assert_eq!(decoded.duration_den_ms, source.delay_den_ms);
        }
        for frame in &decoded.header.frames {
            assert!(frame.len > 0);
            assert!(frame.offset >= decoded.header.blob_section_offset);
        }
    }

    #[test]
    fn rgb_animation_keeps_synthetic_alpha_opaque() {
        let source = fixture();
        let encoded = encode_animation(
            &source,
            EncodePixelFormat::Rgb8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(decoded.header.channels, 3);
        for (decoded, source) in decoded.frames.iter().zip(&source.frames) {
            let AnimationPixels::U8(pixels) = &decoded.pixels else {
                panic!("expected U8 frame");
            };
            for (actual, expected) in pixels.chunks_exact(4).zip(source.image.pixels()) {
                assert_eq!(&actual[..3], &expected.0[..3]);
                assert_eq!(actual[3], 255);
            }
        }
    }

    #[test]
    fn rgba16_animation_roundtrip() {
        let frame0 = Rgba16Image::from_fn(5, 3, |x, y| {
            Rgba([
                (x * 101 + y) as u16,
                65535 - (y * 257 + x) as u16,
                0x1234 + x as u16,
                (x * 4097 + y * 17) as u16,
            ])
        });
        let mut frame1 = frame0.clone();
        frame1.put_pixel(2, 1, Rgba([1, 255, 256, 65534]));
        let source = image_input::LoadedAnimation16 {
            frames: vec![
                image_input::AnimationInputFrame16 {
                    image: frame0,
                    delay_num_ms: 40,
                    delay_den_ms: 1,
                },
                image_input::AnimationInputFrame16 {
                    image: frame1,
                    delay_num_ms: 1000,
                    delay_den_ms: 24,
                },
            ],
            loop_count: 2,
        };
        let encoded = encode_animation16(
            &source,
            EncodePixelFormat::Rgba16,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        for (decoded, source) in decoded.frames.iter().zip(&source.frames) {
            let AnimationPixels::U16(samples) = &decoded.pixels else {
                panic!("expected U16 frame");
            };
            assert_eq!(samples, source.image.as_raw());
        }
        assert_eq!(decoded.header.loop_count, 2);
    }

    #[test]
    fn streaming_decoder_reconstructs_frames_with_a_tiny_cache() {
        let source = fixture();
        let encoded = encode_animation(
            &source,
            EncodePixelFormat::Rgba8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let header = parse_header(&encoded).unwrap();
        let frame_bytes = (header.width * header.height * 4) as usize;
        let mut decoder =
            StreamingAnimationDecoder::new(&encoded, &header, DecodeSafety::Strict, frame_bytes)
                .unwrap();

        for (index, expected) in source.frames.iter().enumerate() {
            let actual = decoder.decode(index).unwrap();
            let AnimationPixels::U8(actual) = actual.as_ref() else {
                panic!("expected U8 frame");
            };
            assert_eq!(actual, expected.image.as_raw());
            assert!(decoder.cache_bytes <= frame_bytes || decoder.cache.len() == 1);
        }
        assert_eq!(decoder.cache.len(), 1);
    }

    #[test]
    fn viewer_prefetch_is_limited_by_time_and_memory() {
        let encoded = encode_animation(
            &fixture(),
            EncodePixelFormat::Rgba8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let mut header = parse_header(&encoded).unwrap();
        header.frames = (0..180)
            .map(|index| AnimationFrameHeader {
                index,
                duration_num_ms: 16,
                duration_den_ms: 1,
                reference: (index > 0).then(|| index - 1),
                coding: AnimationFrameCoding::Full,
                rect: None,
                motion: None,
                offset: 0,
                len: 0,
            })
            .collect();

        header.width = 64;
        header.height = 64;
        let (frames, duration, _) = viewer_prefetch_capacity(&header).unwrap();
        assert_eq!(frames, 125);
        assert_eq!(duration, VIEWER_PREFETCH_TARGET);

        header.width = 1920;
        header.height = 1080;
        let (frames, duration, frame_bytes) = viewer_prefetch_capacity(&header).unwrap();
        assert_eq!(frames, VIEWER_PREFETCH_MAX_BYTES / frame_bytes);
        assert!(frames * frame_bytes <= VIEWER_PREFETCH_MAX_BYTES);
        assert!(duration < VIEWER_PREFETCH_TARGET);
    }

    fn procedural_image(width: u32, height: u32) -> RgbaImage {
        RgbaImage::from_fn(width, height, |x, y| {
            let value = ((x as u64 * 1_103_515_245 + y as u64 * 12_345) >> 13) as u8;
            Rgba([
                value,
                value.rotate_left(3),
                value.wrapping_add((x ^ y) as u8),
                255,
            ])
        })
    }

    fn animation_from_images(images: Vec<RgbaImage>) -> image_input::LoadedAnimation {
        image_input::LoadedAnimation {
            frames: images
                .into_iter()
                .map(|image| image_input::AnimationInputFrame {
                    image,
                    delay_num_ms: 16,
                    delay_den_ms: 1,
                })
                .collect(),
            loop_count: 0,
        }
    }

    #[test]
    fn identical_animation_frame_uses_zero_blob_copy() {
        let first = procedural_image(32, 32);
        let source = animation_from_images(vec![first.clone(), first]);
        let encoded = encode_animation(
            &source,
            EncodePixelFormat::Rgb8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(decoded.header.frames[1].coding, AnimationFrameCoding::Copy);
        assert_eq!(decoded.header.frames[1].len, 0);
        assert_eq!(decoded.frames[0].pixels, decoded.frames[1].pixels);
    }

    #[test]
    fn localized_change_uses_rect_delta() {
        let first = procedural_image(64, 64);
        let mut second = first.clone();
        for y in 20..24 {
            for x in 30..35 {
                second.put_pixel(x, y, Rgba([250, 4, 90, 31]));
            }
        }
        let source = animation_from_images(vec![first, second.clone()]);
        let encoded = encode_animation(
            &source,
            EncodePixelFormat::Rgba8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(decoded.header.frames[1].coding, AnimationFrameCoding::Rect);
        assert_eq!(
            decoded.header.frames[1].rect,
            Some(AnimationRect {
                x: 30,
                y: 20,
                width: 5,
                height: 4,
            })
        );
        let AnimationPixels::U8(actual) = &decoded.frames[1].pixels else {
            panic!("expected U8 frame");
        };
        assert_eq!(actual, second.as_raw());
    }

    #[test]
    fn global_pan_uses_motion_delta() {
        let first = procedural_image(128, 128);
        let second = RgbaImage::from_fn(128, 128, |x, y| *first.get_pixel((x + 2) % 128, y));
        let source = animation_from_images(vec![first, second.clone()]);
        let encoded = encode_animation(
            &source,
            EncodePixelFormat::Rgb8,
            CompressionProfile::Fast,
            false,
        )
        .unwrap();
        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(
            decoded.header.frames[1].coding,
            AnimationFrameCoding::Motion
        );
        assert_eq!(decoded.header.frames[1].motion, Some([2, 0]));
        let AnimationPixels::U8(actual) = &decoded.frames[1].pixels else {
            panic!("expected U8 frame");
        };
        for (actual, expected) in actual.chunks_exact(4).zip(second.pixels()) {
            assert_eq!(&actual[..3], &expected.0[..3]);
            assert_eq!(actual[3], 255);
        }
    }

    #[test]
    fn legacy_v1_header_defaults_to_full_coding() {
        #[derive(Serialize)]
        struct LegacyFrame {
            index: u32,
            duration_num_ms: u32,
            duration_den_ms: u32,
            reference: Option<u32>,
            offset: u64,
            len: u64,
        }

        #[derive(Serialize)]
        struct LegacyHeader {
            kind: String,
            version: u16,
            width: u32,
            height: u32,
            pixfmt: u16,
            channels: u8,
            loop_count: u32,
            blob_section_offset: u64,
            frames: Vec<LegacyFrame>,
        }

        let image = procedural_image(8, 8);
        let blob = encode_rgba8_with_meta_to_vec(
            &image,
            None,
            EncodePixelFormat::Rgb8,
            CompressionProfile::Fast,
            false,
            None,
            Instant::now(),
        )
        .unwrap();
        let mut header = LegacyHeader {
            kind: KIND.to_owned(),
            version: 1,
            width: 8,
            height: 8,
            pixfmt: 1,
            channels: 3,
            loop_count: 1,
            blob_section_offset: 0,
            frames: vec![LegacyFrame {
                index: 0,
                duration_num_ms: 16,
                duration_den_ms: 1,
                reference: None,
                offset: 0,
                len: blob.len() as u64,
            }],
        };
        let mut header_bytes = rmp_serde::to_vec_named(&header).unwrap();
        for _ in 0..8 {
            let offset = (PREAMBLE_LEN + header_bytes.len()) as u64;
            header.blob_section_offset = offset;
            header.frames[0].offset = offset;
            let next = rmp_serde::to_vec_named(&header).unwrap();
            if next.len() == header_bytes.len() {
                header_bytes = next;
                break;
            }
            header_bytes = next;
        }
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&MAGIC);
        encoded.extend_from_slice(&1u16.to_le_bytes());
        encoded.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&header_bytes);
        encoded.extend_from_slice(&blob);

        let decoded = decode_animation(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(decoded.header.version, 1);
        assert_eq!(decoded.header.frames[0].coding, AnimationFrameCoding::Full);
    }
}
