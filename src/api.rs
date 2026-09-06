//! Stable high-level Rust API and a small C ABI for VRAWTEX.

use super::*;
use std::cell::RefCell;
use std::error::Error;
use std::ffi::{CStr, CString, c_char};
use std::path::Path;

/// Result type returned by the Rust API.
pub type ApiResult<T> = Result<T, Box<dyn Error>>;

/// Options shared by still-image and animation encoders.
#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Pixel format written to each VRAWTEX frame blob.
    pub pixel_format: EncodePixelFormat,
    /// Zstd speed/ratio preset.
    pub compression: CompressionProfile,
    /// Emit encoder statistics to stdout.
    pub verbose: bool,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            pixel_format: EncodePixelFormat::Rgba8,
            compression: CompressionProfile::Balance,
            verbose: false,
        }
    }
}

/// Interleaved decoded samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PixelData {
    /// Interleaved 8-bit channel samples.
    U8(Vec<u8>),
    /// Interleaved native Rust `u16` samples. On-disk RAW streams are U16LE.
    U16(Vec<u16>),
}

/// One decoded still image.
#[derive(Clone, Debug)]
pub struct DecodedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of interleaved channels in `pixels`.
    pub channels: u8,
    /// Pixel representation found in the container.
    pub pixel_format: EncodePixelFormat,
    /// Interleaved pixel samples.
    pub pixels: PixelData,
    /// Opaque VRAWTEX metadata bytes, when present.
    pub metadata: Option<Vec<u8>>,
}

/// One input frame for [`encode_animation_rgba8`].
#[derive(Clone, Debug)]
pub struct AnimationFrame8 {
    /// Interleaved RGBA8 pixels with `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Frame delay numerator in milliseconds.
    pub duration_num_ms: u32,
    /// Frame delay denominator in milliseconds; must not be zero.
    pub duration_den_ms: u32,
}

/// One input frame for [`encode_animation_rgba16`].
#[derive(Clone, Debug)]
pub struct AnimationFrame16 {
    /// Interleaved RGBA16 samples with `width * height * 4` elements.
    pub rgba: Vec<u16>,
    /// Frame delay numerator in milliseconds.
    pub duration_num_ms: u32,
    /// Frame delay denominator in milliseconds; must not be zero.
    pub duration_den_ms: u32,
}

/// Decoded still image or multi-frame animation.
#[derive(Clone, Debug)]
pub enum DecodedAsset {
    /// A normal `VRTX` still image.
    Image(DecodedImage),
    /// A `VRAWANM` animation.
    Animation(animation::DecodedAnimation),
}

fn checked_pixel_len(width: u32, height: u32, channels: usize) -> ApiResult<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| "pixel buffer size overflow".into())
}

/// Encode an interleaved RGBA8 buffer into one VRAWTEX still-image blob.
///
/// `options.pixel_format` may be RGBA8 or RGB8. RGB8 discards input alpha.
pub fn encode_rgba8(
    width: u32,
    height: u32,
    rgba: &[u8],
    metadata: Option<&[u8]>,
    options: &EncodeOptions,
) -> ApiResult<Vec<u8>> {
    if options.pixel_format.is_16_bit() {
        return Err("encode_rgba8 requires an 8-bit output pixel format".into());
    }
    let expected = checked_pixel_len(width, height, 4)?;
    if rgba.len() != expected {
        return Err(format!(
            "RGBA8 length mismatch: got {}, expected {expected}",
            rgba.len()
        )
        .into());
    }
    let image = RgbaImage::from_raw(width, height, rgba.to_vec())
        .ok_or("failed to construct RGBA8 image")?;
    encode_rgba8_with_meta_to_vec(
        &image,
        metadata,
        options.pixel_format,
        options.compression,
        options.verbose,
        None,
        Instant::now(),
    )
}

/// Encode an interleaved RGBA16 buffer into one VRAWTEX still-image blob.
///
/// `options.pixel_format` may be RGBA16 or RGB16. RGB16 discards input alpha.
pub fn encode_rgba16(
    width: u32,
    height: u32,
    rgba: &[u16],
    metadata: Option<&[u8]>,
    options: &EncodeOptions,
) -> ApiResult<Vec<u8>> {
    if !options.pixel_format.is_16_bit() {
        return Err("encode_rgba16 requires a 16-bit output pixel format".into());
    }
    let expected = checked_pixel_len(width, height, 4)?;
    if rgba.len() != expected {
        return Err(format!(
            "RGBA16 length mismatch: got {}, expected {expected}",
            rgba.len()
        )
        .into());
    }
    let image = Rgba16Image::from_raw(width, height, rgba.to_vec())
        .ok_or("failed to construct RGBA16 image")?;
    u16_codec::encode_rgba16_with_meta_to_vec(
        &image,
        metadata,
        options.pixel_format,
        options.compression,
        options.verbose,
        None,
        Instant::now(),
    )
}

/// Encode RGBA8 animation frames into a `VRAWANM` container.
///
/// VRAWANM v2 adaptively stores each frame as a keyframe, full delta, changed
/// rectangle, global-motion residual, or zero-blob reference copy.
pub fn encode_animation_rgba8(
    width: u32,
    height: u32,
    frames: &[AnimationFrame8],
    loop_count: u32,
    options: &EncodeOptions,
) -> ApiResult<Vec<u8>> {
    if options.pixel_format.is_16_bit() {
        return Err("encode_animation_rgba8 requires RGBA8 or RGB8 output".into());
    }
    if frames.is_empty() {
        return Err("animation must contain at least one frame".into());
    }
    let expected = checked_pixel_len(width, height, 4)?;
    let mut loaded = Vec::with_capacity(frames.len());
    for frame in frames {
        if frame.rgba.len() != expected || frame.duration_den_ms == 0 {
            return Err("invalid animation frame pixels or duration".into());
        }
        loaded.push(image_input::AnimationInputFrame {
            image: RgbaImage::from_raw(width, height, frame.rgba.clone())
                .ok_or("failed to construct animation frame")?,
            delay_num_ms: frame.duration_num_ms,
            delay_den_ms: frame.duration_den_ms,
        });
    }
    animation::encode_animation(
        &image_input::LoadedAnimation {
            frames: loaded,
            loop_count,
        },
        options.pixel_format,
        options.compression,
        options.verbose,
    )
}

/// Encode native RGBA16 animation frames into a `VRAWANM` container.
///
/// VRAWANM v2 adaptively stores each native-U16 frame as a keyframe, full
/// delta, changed rectangle, global-motion residual, or reference copy.
pub fn encode_animation_rgba16(
    width: u32,
    height: u32,
    frames: &[AnimationFrame16],
    loop_count: u32,
    options: &EncodeOptions,
) -> ApiResult<Vec<u8>> {
    if !options.pixel_format.is_16_bit() {
        return Err("encode_animation_rgba16 requires RGBA16 or RGB16 output".into());
    }
    if frames.is_empty() {
        return Err("animation must contain at least one frame".into());
    }
    let expected = checked_pixel_len(width, height, 4)?;
    let mut loaded = Vec::with_capacity(frames.len());
    for frame in frames {
        if frame.rgba.len() != expected || frame.duration_den_ms == 0 {
            return Err("invalid animation frame pixels or duration".into());
        }
        loaded.push(image_input::AnimationInputFrame16 {
            image: Rgba16Image::from_raw(width, height, frame.rgba.clone())
                .ok_or("failed to construct RGBA16 animation frame")?,
            delay_num_ms: frame.duration_num_ms,
            delay_den_ms: frame.duration_den_ms,
        });
    }
    animation::encode_animation16(
        &image_input::LoadedAnimation16 {
            frames: loaded,
            loop_count,
        },
        options.pixel_format,
        options.compression,
        options.verbose,
    )
}

fn format_from_container(parsed: &ParsedContainer) -> ApiResult<EncodePixelFormat> {
    match (parsed.sample_bytes, parsed.chans) {
        (1, 3) => Ok(EncodePixelFormat::Rgb8),
        (1, 4) => Ok(EncodePixelFormat::Rgba8),
        (2, 3) => Ok(EncodePixelFormat::Rgb16),
        (2, 4) => Ok(EncodePixelFormat::Rgba16),
        _ => Err("unsupported decoded pixel format".into()),
    }
}

/// Decode one normal VRAWTEX still-image blob into interleaved samples.
pub fn decode_image(data: &[u8], safety: DecodeSafety) -> ApiResult<DecodedImage> {
    if animation::is_animation(data) {
        return Err("decode_image received an animation; use decode_asset".into());
    }
    let parsed = parse_container(data, safety)?;
    let pixel_format = format_from_container(&parsed)?;
    let pixels_count = parsed.pixels as usize;
    let pixels = if parsed.sample_bytes == 1 {
        let (planes, _, _, _) = decode_container_to_planes(&parsed, data, safety, true)?;
        let mut interleaved = vec![0u8; pixels_count * parsed.chans as usize];
        for index in 0..pixels_count {
            for channel in 0..parsed.chans as usize {
                interleaved[index * parsed.chans as usize + channel] = planes[channel][index];
            }
        }
        PixelData::U8(interleaved)
    } else {
        let (planes, _, _, _) =
            u16_codec::decode_container_to_planes_u16(&parsed, data, safety, true)?;
        let mut interleaved = vec![0u16; pixels_count * parsed.chans as usize];
        for index in 0..pixels_count {
            for channel in 0..parsed.chans as usize {
                interleaved[index * parsed.chans as usize + channel] = planes[channel][index];
            }
        }
        PixelData::U16(interleaved)
    };
    Ok(DecodedImage {
        width: parsed.width,
        height: parsed.height,
        channels: parsed.chans,
        pixel_format,
        pixels,
        metadata: parsed.meta_raw,
    })
}

/// Decode either a still VRAWTEX blob or a VRAWTEX animation container.
pub fn decode_asset(data: &[u8], safety: DecodeSafety) -> ApiResult<DecodedAsset> {
    if animation::is_animation(data) {
        Ok(DecodedAsset::Animation(animation::decode_animation(
            data, safety,
        )?))
    } else {
        Ok(DecodedAsset::Image(decode_image(data, safety)?))
    }
}

/// Load and encode a supported still image or animation from a filesystem path.
pub fn encode_file_to_vec(input: &Path, options: &EncodeOptions) -> ApiResult<Vec<u8>> {
    if image_input::is_streamed_animation_ext(input)
        && let Some(encoded) = animation::encode_streamed_animation(
            input,
            options.pixel_format,
            options.compression,
            options.verbose,
        )?
    {
        return Ok(encoded);
    }
    if options.pixel_format.is_16_bit()
        && let Some(animation) = image_input::load_animation16(input)?
    {
        return animation::encode_animation16(
            &animation,
            options.pixel_format,
            options.compression,
            options.verbose,
        );
    }
    if !options.pixel_format.is_16_bit()
        && let Some(animation) = image_input::load_animation(input)?
    {
        return animation::encode_animation(
            &animation,
            options.pixel_format,
            options.compression,
            options.verbose,
        );
    }
    if options.pixel_format.is_16_bit() {
        let image = image_input::load_rgba16(input)?;
        u16_codec::encode_rgba16_with_meta_to_vec(
            &image,
            None,
            options.pixel_format,
            options.compression,
            options.verbose,
            fs::metadata(input).ok().map(|metadata| metadata.len()),
            Instant::now(),
        )
    } else {
        let image = image_input::load_rgba8(input)?;
        encode_rgba8_with_meta_to_vec(
            &image,
            None,
            options.pixel_format,
            options.compression,
            options.verbose,
            fs::metadata(input).ok().map(|metadata| metadata.len()),
            Instant::now(),
        )
    }
}

/// Encode a supported still image or animation and write the container to disk.
pub fn encode_file(input: &Path, output: &Path, options: &EncodeOptions) -> ApiResult<()> {
    fs::write(output, encode_file_to_vec(input, options)?)?;
    Ok(())
}

/// Parse animation metadata without decompressing frame blobs.
pub fn inspect_animation(data: &[u8]) -> ApiResult<animation::AnimationHeader> {
    animation::parse_header(data)
}

pub use super::animation::{
    AnimationFrameHeader, AnimationHeader, AnimationPixels, DecodedAnimation, DecodedAnimationFrame,
};

/// Owned byte buffer returned by C ABI memory functions.
#[repr(C)]
pub struct VrawtexBuffer {
    /// Allocated data pointer.
    pub data: *mut u8,
    /// Number of initialized bytes.
    pub len: usize,
    /// Allocation capacity required by [`vrawtex_buffer_free`].
    pub capacity: usize,
}

impl VrawtexBuffer {
    fn from_vec(mut bytes: Vec<u8>) -> Self {
        let buffer = Self {
            data: bytes.as_mut_ptr(),
            len: bytes.len(),
            capacity: bytes.capacity(),
        };
        std::mem::forget(bytes);
        buffer
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("no error").unwrap());
}

fn set_last_error(error: impl ToString) {
    let text = error.to_string().replace('\0', " ");
    LAST_ERROR.with(|slot| {
        *slot.borrow_mut() = CString::new(text).unwrap_or_else(|_| CString::new("error").unwrap());
    });
}

fn ffi_status(operation: impl FnOnce() -> ApiResult<()> + std::panic::UnwindSafe) -> i32 {
    match std::panic::catch_unwind(operation) {
        Ok(Ok(())) => 0,
        Ok(Err(error)) => {
            set_last_error(error);
            1
        }
        Err(_) => {
            set_last_error("panic crossed VRAWTEX API boundary");
            2
        }
    }
}

fn ffi_pixel_format(value: u32) -> ApiResult<EncodePixelFormat> {
    match value {
        0 => Ok(EncodePixelFormat::Rgba8),
        1 => Ok(EncodePixelFormat::Rgb8),
        2 => Ok(EncodePixelFormat::Rgba16),
        3 => Ok(EncodePixelFormat::Rgb16),
        _ => Err("invalid C API pixel format".into()),
    }
}

fn ffi_profile(value: u32) -> ApiResult<CompressionProfile> {
    match value {
        0 => Ok(CompressionProfile::Fast),
        1 => Ok(CompressionProfile::Balance),
        2 => Ok(CompressionProfile::Compact),
        3 => Ok(CompressionProfile::Ultra),
        _ => Err("invalid C API compression profile".into()),
    }
}

unsafe fn ffi_path<'a>(path: *const c_char) -> ApiResult<&'a Path> {
    if path.is_null() {
        return Err("null path pointer".into());
    }
    let path = unsafe { CStr::from_ptr(path) }.to_str()?;
    Ok(Path::new(path))
}

/// Return the calling thread's last C ABI error message.
///
/// The pointer remains valid until the next VRAWTEX C API call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn vrawtex_last_error_message() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Release a buffer returned by a VRAWTEX C ABI memory function.
///
/// # Safety
///
/// `buffer` must be an unmodified successful result from this library and must
/// not have been freed before.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrawtex_buffer_free(buffer: VrawtexBuffer) {
    if !buffer.data.is_null() {
        drop(unsafe { Vec::from_raw_parts(buffer.data, buffer.len, buffer.capacity) });
    }
}

/// Encode an image/animation file through the compiled C ABI.
///
/// Pixel formats: `0=RGBA8, 1=RGB8, 2=RGBA16, 3=RGB16`.
/// Profiles: `0=fast, 1=balance, 2=compact, 3=ultra`.
///
/// # Safety
///
/// `input` and `output` must point to valid NUL-terminated UTF-8 strings for
/// the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrawtex_encode_file(
    input: *const c_char,
    output: *const c_char,
    pixel_format: u32,
    profile: u32,
) -> i32 {
    ffi_status(|| {
        let input = unsafe { ffi_path(input)? };
        let output = unsafe { ffi_path(output)? };
        encode_file(
            input,
            output,
            &EncodeOptions {
                pixel_format: ffi_pixel_format(pixel_format)?,
                compression: ffi_profile(profile)?,
                verbose: false,
            },
        )
    })
}

/// Decode a VRAWTEX file through the compiled C ABI.
///
/// `target=0` writes PNG/frame PNGs, `target=1` writes RAW/frame RAW files.
/// `safety=0` is strict and `safety=1` is relaxed.
///
/// # Safety
///
/// `input` and `output` must point to valid NUL-terminated UTF-8 strings for
/// the duration of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrawtex_decode_file(
    input: *const c_char,
    output: *const c_char,
    target: u32,
    safety: u32,
) -> i32 {
    ffi_status(|| {
        let input = unsafe { ffi_path(input)? }.to_path_buf();
        let output = unsafe { ffi_path(output)? }.to_path_buf();
        let target = match target {
            0 => DecodeFormat::Png,
            1 => DecodeFormat::Raw,
            _ => return Err("invalid C API decode target".into()),
        };
        let safety = match safety {
            0 => DecodeSafety::Strict,
            1 => DecodeSafety::Relaxed,
            _ => return Err("invalid C API safety mode".into()),
        };
        decode_cmd(input, Some(output), target, safety, None, false)
    })
}

/// Encode an RGBA8 memory buffer through the compiled C ABI.
///
/// The returned buffer must be released with [`vrawtex_buffer_free`].
///
/// # Safety
///
/// `rgba` must be readable for `rgba_len` bytes. `output` must be valid and
/// aligned for one write and must not overlap the input allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrawtex_encode_rgba8(
    rgba: *const u8,
    rgba_len: usize,
    width: u32,
    height: u32,
    pixel_format: u32,
    profile: u32,
    output: *mut VrawtexBuffer,
) -> i32 {
    ffi_status(|| {
        if rgba.is_null() || output.is_null() {
            return Err("null memory API pointer".into());
        }
        let rgba = unsafe { std::slice::from_raw_parts(rgba, rgba_len) };
        let pixel_format = ffi_pixel_format(pixel_format)?;
        if pixel_format.is_16_bit() {
            return Err("vrawtex_encode_rgba8 requires RGBA8 or RGB8 output".into());
        }
        let options = EncodeOptions {
            pixel_format,
            compression: ffi_profile(profile)?,
            verbose: false,
        };
        let bytes = encode_rgba8(width, height, rgba, None, &options)?;
        unsafe { output.write(VrawtexBuffer::from_vec(bytes)) };
        Ok(())
    })
}

/// Encode an RGBA16 memory buffer through the compiled C ABI.
///
/// `rgba_len` is measured in `uint16_t` samples, not bytes. Pixel formats must
/// be `2=RGBA16` or `3=RGB16`. The returned buffer must be released with
/// [`vrawtex_buffer_free`].
///
/// # Safety
///
/// `rgba` must be readable for `rgba_len` native `u16` samples. `output` must
/// be valid and aligned for one write and must not overlap the input allocation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrawtex_encode_rgba16(
    rgba: *const u16,
    rgba_len: usize,
    width: u32,
    height: u32,
    pixel_format: u32,
    profile: u32,
    output: *mut VrawtexBuffer,
) -> i32 {
    ffi_status(|| {
        if rgba.is_null() || output.is_null() {
            return Err("null memory API pointer".into());
        }
        let rgba = unsafe { std::slice::from_raw_parts(rgba, rgba_len) };
        let pixel_format = ffi_pixel_format(pixel_format)?;
        if !pixel_format.is_16_bit() {
            return Err("vrawtex_encode_rgba16 requires RGBA16 or RGB16 output".into());
        }
        let bytes = encode_rgba16(
            width,
            height,
            rgba,
            None,
            &EncodeOptions {
                pixel_format,
                compression: ffi_profile(profile)?,
                verbose: false,
            },
        )?;
        unsafe { output.write(VrawtexBuffer::from_vec(bytes)) };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_memory_api_roundtrips_rgba16() {
        let source = vec![0, 1, 255, 256, 65535, 32768, 1234, 4321];
        let options = EncodeOptions {
            pixel_format: EncodePixelFormat::Rgba16,
            compression: CompressionProfile::Fast,
            verbose: false,
        };
        let encoded = encode_rgba16(2, 1, &source, None, &options).unwrap();
        let decoded = decode_image(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(decoded.pixels, PixelData::U16(source));
    }
}
