use image::{ImageFormat, RgbaImage};
use rawloader::{RawImage, RawImageData};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Copy, Clone)]
struct RawCrop {
    top: usize,
    left: usize,
    width: usize,
    height: usize,
}

enum RawSamples<'a> {
    Integer(&'a [u16]),
    Float(&'a [f32]),
}

#[derive(Copy, Clone)]
enum TiffEndian {
    Little,
    Big,
}

#[derive(Clone)]
struct TiffEntry {
    tag: u16,
    type_id: u16,
    count: u32,
    inline: [u8; 4],
}

struct TiffIfd {
    entries: Vec<TiffEntry>,
    next_offset: Option<u32>,
}

struct EmbeddedJpegCandidate {
    offset: usize,
    byte_count: usize,
    pixels: u64,
    orientation: u16,
}

impl RawSamples<'_> {
    fn len(&self) -> usize {
        match self {
            RawSamples::Integer(v) => v.len(),
            RawSamples::Float(v) => v.len(),
        }
    }

    fn sensor_value(&self, idx: usize, channel: usize, normalize: &RawNormalize) -> f32 {
        match self {
            RawSamples::Integer(v) => normalize.integer(v[idx], channel),
            RawSamples::Float(v) => normalize.float(v[idx], channel),
        }
    }
}

struct RawNormalize {
    blacklevels: [f32; 4],
    whitelevels: [f32; 4],
    wb: [f32; 4],
    cam_to_xyz: Option<[[f32; 4]; 3]>,
}

impl RawNormalize {
    fn new(raw: &RawImage) -> Self {
        let mut wb = raw.wb_coeffs;
        if wb.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            wb = raw.neutralwb();
        }

        let green = wb[1];
        if green.is_finite() && green > 0.0 {
            for v in &mut wb {
                *v /= green;
            }
        }

        for v in &mut wb {
            if !v.is_finite() || *v <= 0.0 {
                *v = 1.0;
            }
        }

        let mut cam_to_xyz = raw.cam_to_xyz_normalized();
        let matrix_ok = cam_to_xyz.iter().flatten().all(|v| v.is_finite())
            && cam_to_xyz.iter().flatten().any(|v| v.abs() > 0.000001);

        if !matrix_ok {
            cam_to_xyz = [[0.0; 4]; 3];
        }

        Self {
            blacklevels: raw.blacklevels.map(|v| v as f32),
            whitelevels: raw.whitelevels.map(|v| v as f32),
            wb,
            cam_to_xyz: matrix_ok.then_some(cam_to_xyz),
        }
    }

    fn integer(&self, value: u16, channel: usize) -> f32 {
        let channel = channel.min(3);
        let black = self.blacklevels[channel];
        let white = self.whitelevels[channel].max(black + 1.0);
        (((value as f32 - black) / (white - black)).clamp(0.0, 1.0) * self.wb[channel])
            .clamp(0.0, 8.0)
    }

    fn float(&self, value: f32, channel: usize) -> f32 {
        if !value.is_finite() {
            return 0.0;
        }
        (value.max(0.0) * self.wb[channel.min(3)]).clamp(0.0, 8.0)
    }

    fn camera_rgb_to_srgb8(&self, rgb: [f32; 3]) -> [u8; 3] {
        let linear = if let Some(cam_to_xyz) = self.cam_to_xyz {
            let cam = [rgb[0], rgb[1], rgb[2], 0.0];
            let x = dot4(cam_to_xyz[0], cam);
            let y = dot4(cam_to_xyz[1], cam);
            let z = dot4(cam_to_xyz[2], cam);
            [
                3.2406 * x - 1.5372 * y - 0.4986 * z,
                -0.9689 * x + 1.8758 * y + 0.0415 * z,
                0.0557 * x - 0.2040 * y + 1.0570 * z,
            ]
        } else {
            rgb
        };

        [
            linear_to_srgb8(linear[0]),
            linear_to_srgb8(linear[1]),
            linear_to_srgb8(linear[2]),
        ]
    }
}

pub(crate) fn is_supported_input_ext(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "tga" | "tif" | "tiff" | "gif" | "dng"
    )
}

pub(crate) fn load_rgba8(path: &Path) -> Result<RgbaImage, Box<dyn Error>> {
    if is_dng_ext(path) {
        load_dng_rgba8(path)
    } else {
        Ok(load_image_no_limits(path)?.to_rgba8())
    }
}

fn is_dng_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("dng"))
}

fn load_dng_rgba8(path: &Path) -> Result<RgbaImage, Box<dyn Error>> {
    if has_image_magic(path)? {
        if let Ok(img) = load_image_by_magic(path) {
            return Ok(img.to_rgba8());
        }
    }

    let mut errors = Vec::new();

    match load_dng_embedded_jpeg(path) {
        Ok(rgba) => return Ok(rgba),
        Err(e) => errors.push(format!("embedded JPEG fallback failed: {e}")),
    }

    match load_dng_with_rawloader(path) {
        Ok(rgba) if !looks_like_broken_green_dng(&rgba) => return Ok(rgba),
        Ok(_) => errors.push("rawloader produced a suspicious green image".to_string()),
        Err(e) => errors.push(format!("rawloader failed: {e}")),
    }

    match load_dng_with_magick(path) {
        Ok(rgba) => return Ok(rgba),
        Err(e) => errors.push(format!("ImageMagick fallback failed: {e}")),
    }

    match load_dng_with_rawler(path) {
        Ok(rgba) if !looks_like_broken_green_dng(&rgba) => return Ok(rgba),
        Ok(_) => errors.push("rawler produced a suspicious green image".to_string()),
        Err(e) => errors.push(format!("rawler failed: {e}")),
    };

    Err(format!("failed to decode DNG: {}", errors.join("; ")).into())
}

fn load_dng_with_rawloader(path: &Path) -> Result<RgbaImage, Box<dyn Error>> {
    run_with_silent_decoder_panics("rawloader", || {
        let raw = rawloader::decode_file(path)?;
        let mut rgba = raw_to_rgba8(&raw)?;
        rgba = apply_orientation_flips(rgba, raw.orientation.to_flips());
        Ok(rgba)
    })
}

fn has_image_magic(path: &Path) -> Result<bool, Box<dyn Error>> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 12];
    let n = file.read(&mut magic)?;
    let magic = &magic[..n];
    Ok(magic.starts_with(&[0xFF, 0xD8, 0xFF])
        || magic.starts_with(b"\x89PNG\r\n\x1A\n")
        || magic.starts_with(b"GIF87a")
        || magic.starts_with(b"GIF89a")
        || magic.starts_with(b"BM"))
}

fn load_dng_with_rawler(path: &Path) -> Result<RgbaImage, Box<dyn Error>> {
    run_with_silent_decoder_panics("rawler", || {
        let raw = rawler::decode_file(path)?;
        validate_rawler_develop_safe(&raw)?;
        let develop = rawler::imgop::develop::RawDevelop::default();
        let decoded = develop
            .develop_intermediate(&raw)?
            .to_dynamic_image()
            .ok_or("rawler produced invalid image buffer")?;
        let rgba = apply_orientation_flips(decoded.to_rgba8(), raw.orientation.to_flips());
        Ok(rgba)
    })
}

fn run_with_silent_decoder_panics<T, F>(decoder: &str, f: F) -> Result<T, Box<dyn Error>>
where
    F: FnOnce() -> Result<T, Box<dyn Error>>,
{
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(previous_hook);

    match result {
        Ok(result) => result,
        Err(payload) => {
            Err(format!("{decoder} panicked: {}", panic_payload_message(&payload)).into())
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn validate_rawler_develop_safe(raw: &rawler::RawImage) -> Result<(), Box<dyn Error>> {
    match raw.photometric {
        rawler::rawimage::RawPhotometricInterpretation::BlackIsZero => {
            Err("rawler cannot develop BlackIsZero DNG yet".into())
        }
        rawler::rawimage::RawPhotometricInterpretation::LinearRaw => {
            let black_len = raw.blacklevel.as_vec().len();
            let white_len = raw.whitelevel.as_vec().len();
            if black_len == white_len {
                Ok(())
            } else {
                Err(format!(
                    "rawler cannot safely rescale LinearRaw DNG with blacklevel count {} and whitelevel count {}",
                    black_len, white_len
                )
                .into())
            }
        }
        rawler::rawimage::RawPhotometricInterpretation::Cfa(_) => Ok(()),
    }
}

fn load_dng_with_magick(path: &Path) -> Result<RgbaImage, Box<dyn Error>> {
    let tmp_path = std::env::temp_dir().join(format!(
        "vrawtex-dng-{}-{}.png",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let output_arg = format!("png32:{}", tmp_path.display());

    let output = Command::new("magick")
        .arg(path)
        .arg(&output_arg)
        .output()
        .map_err(|e| format!("could not run `magick`: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let _ = fs::remove_file(&tmp_path);
        return Err(format!(
            "`magick` exited with {}: {}{}",
            output.status,
            stderr.trim(),
            stdout.trim()
        )
        .into());
    }

    let rgba = denoise_dng_fallback_rgba(load_image_no_limits(&tmp_path)?.to_rgba8());
    let _ = fs::remove_file(&tmp_path);
    Ok(rgba)
}

fn load_image_by_magic(path: &Path) -> Result<image::DynamicImage, Box<dyn Error>> {
    load_image_no_limits(path)
}

fn load_image_no_limits(path: &Path) -> Result<image::DynamicImage, Box<dyn Error>> {
    let mut reader = image::ImageReader::open(path)?.with_guessed_format()?;
    reader.no_limits();
    Ok(reader.decode()?)
}

fn load_image_bytes_no_limits(
    bytes: &[u8],
    format: ImageFormat,
) -> Result<image::DynamicImage, Box<dyn Error>> {
    let cursor = Cursor::new(bytes);
    let mut reader = image::ImageReader::with_format(BufReader::new(cursor), format);
    reader.no_limits();
    Ok(reader.decode()?)
}

fn load_dng_embedded_jpeg(path: &Path) -> Result<RgbaImage, Box<dyn Error>> {
    let data = fs::read(path)?;
    let (endian, first_ifd_offset) = parse_tiff_header(&data)?;
    let mut queue = VecDeque::from([first_ifd_offset]);
    let mut visited = Vec::new();
    let mut best = None;
    let mut main_orientation = 1u16;

    while let Some(ifd_offset) = queue.pop_front() {
        if ifd_offset == 0 || visited.contains(&ifd_offset) {
            continue;
        }
        visited.push(ifd_offset);

        let ifd = read_tiff_ifd(&data, ifd_offset, endian)?;
        let orientation = tiff_entry_u16(&data, &ifd.entries, 0x0112, endian)
            .unwrap_or(main_orientation)
            .max(1);
        if ifd_offset == first_ifd_offset {
            main_orientation = orientation;
        }

        if let Some(offsets) = tiff_entry_u32s(&data, &ifd.entries, 0x014a, endian)? {
            queue.extend(offsets);
        }
        if let Some(next_offset) = ifd.next_offset {
            queue.push_back(next_offset);
        }

        let compression = tiff_entry_u16(&data, &ifd.entries, 0x0103, endian);
        let width = tiff_entry_u32(&data, &ifd.entries, 0x0100, endian)?;
        let height = tiff_entry_u32(&data, &ifd.entries, 0x0101, endian)?;
        let strip_offsets = tiff_entry_u32s(&data, &ifd.entries, 0x0111, endian)?;
        let strip_byte_counts = tiff_entry_u32s(&data, &ifd.entries, 0x0117, endian)?;

        let Some((width, height, strip_offsets, strip_byte_counts)) = width
            .zip(height)
            .zip(strip_offsets.zip(strip_byte_counts))
            .map(|((width, height), (strip_offsets, strip_byte_counts))| {
                (width, height, strip_offsets, strip_byte_counts)
            })
        else {
            continue;
        };

        if !matches!(compression, Some(6 | 7)) {
            continue;
        }

        for (&offset, &byte_count) in strip_offsets.iter().zip(&strip_byte_counts) {
            let offset = offset as usize;
            let byte_count = byte_count as usize;
            let Some(bytes) = checked_slice(&data, offset, byte_count) else {
                continue;
            };
            if !bytes.starts_with(&[0xFF, 0xD8]) {
                continue;
            }

            let pixels = width as u64 * height as u64;
            let candidate = EmbeddedJpegCandidate {
                offset,
                byte_count,
                pixels,
                orientation,
            };
            if best
                .as_ref()
                .is_none_or(|best: &EmbeddedJpegCandidate| candidate.pixels > best.pixels)
            {
                best = Some(candidate);
            }
        }
    }

    let candidate = best.ok_or("DNG has no embedded JPEG image")?;
    let jpeg = checked_slice(&data, candidate.offset, candidate.byte_count)
        .ok_or("embedded JPEG range is outside DNG file")?;
    let rgba = load_image_bytes_no_limits(jpeg, ImageFormat::Jpeg)?.to_rgba8();
    Ok(apply_tiff_orientation(rgba, candidate.orientation))
}

fn parse_tiff_header(data: &[u8]) -> Result<(TiffEndian, u32), Box<dyn Error>> {
    if data.len() < 8 {
        return Err("TIFF header is too short".into());
    }

    let endian = match &data[..2] {
        b"II" => TiffEndian::Little,
        b"MM" => TiffEndian::Big,
        _ => return Err("DNG is not a TIFF container".into()),
    };

    let magic = read_tiff_u16(data, 2, endian)?;
    if magic != 42 {
        return Err(format!("unsupported TIFF magic: {magic}").into());
    }

    Ok((endian, read_tiff_u32(data, 4, endian)?))
}

fn read_tiff_ifd(data: &[u8], offset: u32, endian: TiffEndian) -> Result<TiffIfd, Box<dyn Error>> {
    let offset = offset as usize;
    let count = read_tiff_u16(data, offset, endian)? as usize;
    let entries_start = offset.checked_add(2).ok_or("TIFF IFD offset overflow")?;
    let entries_bytes = count
        .checked_mul(12)
        .ok_or("TIFF IFD entry table size overflow")?;
    let next_offset_pos = entries_start
        .checked_add(entries_bytes)
        .ok_or("TIFF IFD next offset overflow")?;
    checked_slice(data, entries_start, entries_bytes).ok_or("TIFF IFD entries exceed file size")?;

    let mut entries = Vec::with_capacity(count);
    for idx in 0..count {
        let pos = entries_start + idx * 12;
        let mut inline = [0u8; 4];
        inline.copy_from_slice(checked_slice(data, pos + 8, 4).ok_or("TIFF entry is truncated")?);
        entries.push(TiffEntry {
            tag: read_tiff_u16(data, pos, endian)?,
            type_id: read_tiff_u16(data, pos + 2, endian)?,
            count: read_tiff_u32(data, pos + 4, endian)?,
            inline,
        });
    }

    let next_offset = read_tiff_u32(data, next_offset_pos, endian).ok();
    Ok(TiffIfd {
        entries,
        next_offset: next_offset.filter(|offset| *offset != 0),
    })
}

fn tiff_entry_u16(data: &[u8], entries: &[TiffEntry], tag: u16, endian: TiffEndian) -> Option<u16> {
    tiff_entry_u32(data, entries, tag, endian)
        .ok()
        .flatten()
        .and_then(|v| u16::try_from(v).ok())
}

fn tiff_entry_u32(
    data: &[u8],
    entries: &[TiffEntry],
    tag: u16,
    endian: TiffEndian,
) -> Result<Option<u32>, Box<dyn Error>> {
    Ok(tiff_entry_u32s(data, entries, tag, endian)?.and_then(|values| values.first().copied()))
}

fn tiff_entry_u32s(
    data: &[u8],
    entries: &[TiffEntry],
    tag: u16,
    endian: TiffEndian,
) -> Result<Option<Vec<u32>>, Box<dyn Error>> {
    let Some(entry) = entries.iter().find(|entry| entry.tag == tag) else {
        return Ok(None);
    };
    let bytes = tiff_entry_bytes(data, entry, endian)?;

    match entry.type_id {
        3 => Ok(Some(
            bytes
                .chunks_exact(2)
                .map(|chunk| read_tiff_u16(chunk, 0, endian).map(|v| v as u32))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        4 => Ok(Some(
            bytes
                .chunks_exact(4)
                .map(|chunk| read_tiff_u32(chunk, 0, endian))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        _ => Ok(None),
    }
}

fn tiff_entry_bytes(
    data: &[u8],
    entry: &TiffEntry,
    endian: TiffEndian,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let type_size = match entry.type_id {
        1 | 2 | 6 | 7 => 1usize,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => return Err(format!("unsupported TIFF field type: {}", entry.type_id).into()),
    };
    let len = (entry.count as usize)
        .checked_mul(type_size)
        .ok_or("TIFF field byte size overflow")?;

    if len <= 4 {
        Ok(entry.inline[..len].to_vec())
    } else {
        let offset = read_tiff_u32(&entry.inline, 0, endian)? as usize;
        Ok(checked_slice(data, offset, len)
            .ok_or("TIFF field data exceeds file size")?
            .to_vec())
    }
}

fn read_tiff_u16(data: &[u8], offset: usize, endian: TiffEndian) -> Result<u16, Box<dyn Error>> {
    let bytes = checked_slice(data, offset, 2).ok_or("unexpected EOF while reading TIFF u16")?;
    Ok(match endian {
        TiffEndian::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
        TiffEndian::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
    })
}

fn read_tiff_u32(data: &[u8], offset: usize, endian: TiffEndian) -> Result<u32, Box<dyn Error>> {
    let bytes = checked_slice(data, offset, 4).ok_or("unexpected EOF while reading TIFF u32")?;
    Ok(match endian {
        TiffEndian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        TiffEndian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
    })
}

fn checked_slice(data: &[u8], offset: usize, len: usize) -> Option<&[u8]> {
    data.get(offset..offset.checked_add(len)?)
}

fn apply_tiff_orientation(rgba: RgbaImage, orientation: u16) -> RgbaImage {
    let flips = match orientation {
        2 => (false, true, false),
        3 => (false, true, true),
        4 => (false, false, true),
        5 => (true, false, false),
        6 => (true, false, true),
        7 => (true, true, true),
        8 => (true, true, false),
        _ => (false, false, false),
    };
    apply_orientation_flips(rgba, flips)
}

fn raw_to_rgba8(raw: &RawImage) -> Result<RgbaImage, Box<dyn Error>> {
    let samples = match &raw.data {
        RawImageData::Integer(v) => RawSamples::Integer(v),
        RawImageData::Float(v) => RawSamples::Float(v),
    };

    if raw.width == 0 || raw.height == 0 {
        return Err("DNG has empty dimensions".into());
    }
    if raw.cpp == 0 {
        return Err("DNG has zero components per pixel".into());
    }

    let expected_len = raw
        .width
        .checked_mul(raw.height)
        .and_then(|pixels| pixels.checked_mul(raw.cpp))
        .ok_or("DNG dimensions overflow")?;
    if samples.len() < expected_len {
        return Err(format!(
            "DNG sample data is too short: expected at least {}, got {}",
            expected_len,
            samples.len()
        )
        .into());
    }

    let crop = crop_from_raw(raw)?;
    let normalize = RawNormalize::new(raw);

    match raw.cpp {
        1 if raw.is_monochrome() || !raw.cfa.is_valid() => {
            monochrome_raw_to_rgba8(raw, &samples, crop, &normalize)
        }
        1 => demosaic_raw_to_rgba8(raw, &samples, crop, &normalize),
        3 => camera_rgb_raw_to_rgba8(raw, &samples, crop, &normalize),
        _ => Err(format!("unsupported DNG components per pixel: {}", raw.cpp).into()),
    }
}

fn looks_like_broken_green_dng(rgba: &RgbaImage) -> bool {
    let raw = rgba.as_raw();
    if raw.len() < 4 {
        return false;
    }

    let pixels = raw.len() / 4;
    let step = (pixels / 65_536).max(1);
    let mut r_sum = 0u64;
    let mut g_sum = 0u64;
    let mut b_sum = 0u64;
    let mut count = 0u64;

    for idx in (0..pixels).step_by(step) {
        let base = idx * 4;
        r_sum += raw[base] as u64;
        g_sum += raw[base + 1] as u64;
        b_sum += raw[base + 2] as u64;
        count += 1;
    }

    if count == 0 {
        return false;
    }

    let r_mean = r_sum as f64 / count as f64;
    let g_mean = g_sum as f64 / count as f64;
    let b_mean = b_sum as f64 / count as f64;

    g_mean > 100.0 && g_mean > r_mean * 6.0 && g_mean > b_mean * 6.0
}

fn denoise_dng_fallback_rgba(rgba: RgbaImage) -> RgbaImage {
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;
    if width < 3 || height < 3 {
        return rgba;
    }

    let src = rgba.as_raw();
    let pixels = width * height;
    let mut y_plane = vec![0.0f32; pixels];
    let mut cb_plane = vec![0.0f32; pixels];
    let mut cr_plane = vec![0.0f32; pixels];

    y_plane
        .par_iter_mut()
        .zip(cb_plane.par_iter_mut())
        .zip(cr_plane.par_iter_mut())
        .enumerate()
        .for_each(|(idx, ((y_out, cb_out), cr_out))| {
            let base = idx * 4;
            let r = src[base] as f32;
            let g = src[base + 1] as f32;
            let b = src[base + 2] as f32;
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            *y_out = y;
            *cb_out = b - y;
            *cr_out = r - y;
        });

    let cb_median = median3x3_f32(&cb_plane, width, height);
    let cr_median = median3x3_f32(&cr_plane, width, height);
    let y_median = median3x3_f32(&y_plane, width, height);

    let cb_smooth = box_blur_f32(&cb_median, width, height, 5);
    let cr_smooth = box_blur_f32(&cr_median, width, height, 5);
    let y_smooth = box_blur_f32(&y_median, width, height, 2);

    let mut out = src.clone();
    out.par_chunks_mut(4).enumerate().for_each(|(idx, px)| {
        let y = y_plane[idx];
        let dark = ((170.0 - y) / 170.0).clamp(0.0, 1.0);
        let detail = ((y - y_smooth[idx]).abs() / 45.0).clamp(0.0, 1.0);
        let highlight = ((y - 175.0) / 80.0).clamp(0.0, 1.0);
        let detail_guard = 1.0 - detail * 0.45;
        let highlight_guard = 1.0 - highlight * 0.55;
        let chroma_strength = (0.70 + dark * 0.28) * highlight_guard;
        let luma_strength = dark.powf(0.85) * 0.70 * detail_guard;
        let dark_desaturate = dark.powf(1.25) * 0.18;

        let y = lerp(y, y_smooth[idx], luma_strength);
        let cb = lerp(
            lerp(cb_plane[idx], cb_smooth[idx], chroma_strength),
            0.0,
            dark_desaturate,
        );
        let cr = lerp(
            lerp(cr_plane[idx], cr_smooth[idx], chroma_strength),
            0.0,
            dark_desaturate,
        );

        let r = y + cr;
        let b = y + cb;
        let g = (y - 0.299 * r - 0.114 * b) / 0.587;

        px[0] = f32_to_u8(r);
        px[1] = f32_to_u8(g);
        px[2] = f32_to_u8(b);
    });

    RgbaImage::from_raw(width as u32, height as u32, out)
        .expect("denoise keeps RGBA buffer dimensions valid")
}

fn median3x3_f32(src: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; src.len()];
    dst.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(1);
        let y1 = (y + 1).min(height - 1);
        for x in 0..width {
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);
            let mut values = [0.0f32; 9];
            let mut count = 0usize;

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    values[count] = src[ny * width + nx];
                    count += 1;
                }
            }

            values[..count].sort_by(|a, b| a.total_cmp(b));
            row[x] = values[count / 2];
        }
    });
    dst
}

fn box_blur_f32(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut tmp = vec![0.0f32; src.len()];
    tmp.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let row_start = y * width;
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width - 1);
            let mut sum = 0.0f32;
            for nx in x0..=x1 {
                sum += src[row_start + nx];
            }
            row[x] = sum / (x1 - x0 + 1) as f32;
        }
    });

    let mut dst = vec![0.0f32; src.len()];
    dst.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(height - 1);
        for x in 0..width {
            let mut sum = 0.0f32;
            for ny in y0..=y1 {
                sum += tmp[ny * width + x];
            }
            row[x] = sum / (y1 - y0 + 1) as f32;
        }
    });

    dst
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn f32_to_u8(v: f32) -> u8 {
    (v + 0.5).clamp(0.0, 255.0) as u8
}

fn crop_from_raw(raw: &RawImage) -> Result<RawCrop, Box<dyn Error>> {
    let top = raw.crops[0];
    let right = raw.crops[1];
    let bottom = raw.crops[2];
    let left = raw.crops[3];

    let used_w = left
        .checked_add(right)
        .filter(|used| *used < raw.width)
        .ok_or("DNG crop removes the whole width")?;
    let used_h = top
        .checked_add(bottom)
        .filter(|used| *used < raw.height)
        .ok_or("DNG crop removes the whole height")?;

    Ok(RawCrop {
        top,
        left,
        width: raw.width - used_w,
        height: raw.height - used_h,
    })
}

fn camera_rgb_raw_to_rgba8(
    raw: &RawImage,
    samples: &RawSamples<'_>,
    crop: RawCrop,
    normalize: &RawNormalize,
) -> Result<RgbaImage, Box<dyn Error>> {
    let mut out = vec![0u8; rgba_len(crop)?];
    let stride = crop.width * 4;

    out.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        let sy = crop.top + y;
        for x in 0..crop.width {
            let sx = crop.left + x;
            let src_idx = (sy * raw.width + sx) * raw.cpp;
            let rgb = [
                samples.sensor_value(src_idx, 0, normalize),
                samples.sensor_value(src_idx + 1, 1, normalize),
                samples.sensor_value(src_idx + 2, 2, normalize),
            ];
            let rgb8 = normalize.camera_rgb_to_srgb8(rgb);
            let dst = x * 4;
            row[dst] = rgb8[0];
            row[dst + 1] = rgb8[1];
            row[dst + 2] = rgb8[2];
            row[dst + 3] = 255;
        }
    });

    rgba_from_vec(crop, out)
}

fn monochrome_raw_to_rgba8(
    raw: &RawImage,
    samples: &RawSamples<'_>,
    crop: RawCrop,
    normalize: &RawNormalize,
) -> Result<RgbaImage, Box<dyn Error>> {
    let mut out = vec![0u8; rgba_len(crop)?];
    let stride = crop.width * 4;

    out.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        let sy = crop.top + y;
        for x in 0..crop.width {
            let sx = crop.left + x;
            let src_idx = sy * raw.width + sx;
            let luma = linear_to_srgb8(samples.sensor_value(src_idx, 1, normalize));
            let dst = x * 4;
            row[dst] = luma;
            row[dst + 1] = luma;
            row[dst + 2] = luma;
            row[dst + 3] = 255;
        }
    });

    rgba_from_vec(crop, out)
}

fn demosaic_raw_to_rgba8(
    raw: &RawImage,
    samples: &RawSamples<'_>,
    crop: RawCrop,
    normalize: &RawNormalize,
) -> Result<RgbaImage, Box<dyn Error>> {
    let mut out = vec![0u8; rgba_len(crop)?];
    let stride = crop.width * 4;
    let max_radius = raw.cfa.width.max(raw.cfa.height).max(2);

    out.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        for x in 0..crop.width {
            let rgb = [
                averaged_cfa_channel(raw, samples, crop, normalize, x, y, 0, max_radius),
                averaged_cfa_channel(raw, samples, crop, normalize, x, y, 1, max_radius),
                averaged_cfa_channel(raw, samples, crop, normalize, x, y, 2, max_radius),
            ];
            let rgb8 = normalize.camera_rgb_to_srgb8(rgb);
            let dst = x * 4;
            row[dst] = rgb8[0];
            row[dst + 1] = rgb8[1];
            row[dst + 2] = rgb8[2];
            row[dst + 3] = 255;
        }
    });

    rgba_from_vec(crop, out)
}

fn averaged_cfa_channel(
    raw: &RawImage,
    samples: &RawSamples<'_>,
    crop: RawCrop,
    normalize: &RawNormalize,
    x: usize,
    y: usize,
    target_channel: usize,
    max_radius: usize,
) -> f32 {
    let sx = crop.left + x;
    let sy = crop.top + y;
    let color = raw.cfa.color_at(sy, sx);
    if cfa_color_matches_channel(color, target_channel) {
        return cfa_sensor_value(raw, samples, normalize, sx, sy, color);
    }

    for radius in 1..=max_radius {
        let x0 = x.saturating_sub(radius);
        let y0 = y.saturating_sub(radius);
        let x1 = (x + radius).min(crop.width - 1);
        let y1 = (y + radius).min(crop.height - 1);
        let mut sum = 0.0f32;
        let mut count = 0usize;

        for ny in y0..=y1 {
            for nx in x0..=x1 {
                if nx == x && ny == y {
                    continue;
                }
                let nsx = crop.left + nx;
                let nsy = crop.top + ny;
                let ncolor = raw.cfa.color_at(nsy, nsx);
                if cfa_color_matches_channel(ncolor, target_channel) {
                    sum += cfa_sensor_value(raw, samples, normalize, nsx, nsy, ncolor);
                    count += 1;
                }
            }
        }

        if count > 0 {
            return sum / count as f32;
        }
    }

    cfa_sensor_value(raw, samples, normalize, sx, sy, color)
}

fn cfa_sensor_value(
    raw: &RawImage,
    samples: &RawSamples<'_>,
    normalize: &RawNormalize,
    x: usize,
    y: usize,
    cfa_color: usize,
) -> f32 {
    samples.sensor_value(y * raw.width + x, cfa_color.min(3), normalize)
}

fn cfa_color_matches_channel(cfa_color: usize, channel: usize) -> bool {
    match cfa_color {
        0 | 2 => cfa_color == channel,
        1 | 3 => channel == 1,
        _ => false,
    }
}

fn rgba_len(crop: RawCrop) -> Result<usize, Box<dyn Error>> {
    crop.width
        .checked_mul(crop.height)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "DNG RGBA output size overflow".into())
}

fn rgba_from_vec(crop: RawCrop, out: Vec<u8>) -> Result<RgbaImage, Box<dyn Error>> {
    let width = u32::try_from(crop.width).map_err(|_| "DNG width does not fit u32")?;
    let height = u32::try_from(crop.height).map_err(|_| "DNG height does not fit u32")?;
    RgbaImage::from_raw(width, height, out).ok_or_else(|| "invalid DNG RGBA buffer size".into())
}

fn apply_orientation_flips(rgba: RgbaImage, flips: (bool, bool, bool)) -> RgbaImage {
    let (transpose, flip_x, flip_y) = flips;
    if !transpose && !flip_x && !flip_y {
        return rgba;
    }

    let src_w = rgba.width() as usize;
    let src_h = rgba.height() as usize;
    let dst_w = if transpose { src_h } else { src_w };
    let dst_h = if transpose { src_w } else { src_h };

    let src = rgba.as_raw();
    let mut out = vec![0u8; dst_w * dst_h * 4];

    for sy in 0..src_h {
        for sx in 0..src_w {
            let fx = if flip_x { src_w - 1 - sx } else { sx };
            let fy = if flip_y { src_h - 1 - sy } else { sy };
            let (dx, dy) = if transpose { (fy, fx) } else { (fx, fy) };
            let src_idx = (sy * src_w + sx) * 4;
            let dst_idx = (dy * dst_w + dx) * 4;
            out[dst_idx..dst_idx + 4].copy_from_slice(&src[src_idx..src_idx + 4]);
        }
    }

    RgbaImage::from_raw(dst_w as u32, dst_h as u32, out)
        .expect("orientation keeps RGBA buffer dimensions valid")
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn linear_to_srgb8(v: f32) -> u8 {
    let v = if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let srgb = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_extensions_include_dng_case_insensitive() {
        assert!(is_supported_input_ext(Path::new("photo.dng")));
        assert!(is_supported_input_ext(Path::new("photo.DNG")));
        assert!(is_supported_input_ext(Path::new("photo.png")));
        assert!(!is_supported_input_ext(Path::new("photo.txt")));
    }

    #[test]
    fn cfa_green_and_emerald_map_to_green_channel() {
        assert!(cfa_color_matches_channel(1, 1));
        assert!(cfa_color_matches_channel(3, 1));
        assert!(!cfa_color_matches_channel(3, 0));
        assert!(!cfa_color_matches_channel(3, 2));
    }
}
