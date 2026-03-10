mod atlas;
mod lanczos;

use clap::{Parser, Subcommand, ValueEnum};
use image::{ColorType, GenericImageView, RgbaImage};
use minifb::{Key, Window, WindowOptions};
use serde::Serialize;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use walkdir::WalkDir;
use zstd::{bulk, stream::Encoder};

const MAX_WINDOW_WIDTH: usize = 1920;
const MAX_WINDOW_HEIGHT: usize = 1080;

const ZSTD_LEVEL: i32 = 10;
const AUTO_SELECT_ZSTD_LEVEL: i32 = 3;
const ZSTD_WORKERS_TOTAL: u32 = 8;
const CHUNK_TARGET: usize = 128 * 1024;
const PREDICTOR_SAMPLE_BYTES: usize = 256 * 1024;
const PREDICTOR_NONE_MIN_GAIN_BPS: u64 = 200; // 2.00%
const COLOR_TRANSFORM_MIN_GAIN_BPS: u64 = 50; // 0.50%

const FILE_MAGIC: [u8; 4] = *b"VRTX";
const FILE_VERSION: u8 = 2;
const FILE_VERSION_V1: u8 = 1;
const HEADER_V1_SIZE: usize = 4 + 1 + 4 + 8;
const STREAM_HEADER_V1_SIZE: usize = 8 + 8 + 1;

const MAX_STRICT_SIDE: u32 = atlas::MAX_ATLAS_SIDE;
const MAX_STRICT_PIXELS: u64 = (MAX_STRICT_SIDE as u64) * (MAX_STRICT_SIDE as u64);
const MAX_STRICT_RAW_PLANAR_BYTES: u64 = 2_000_000_000;
const MAX_STRICT_META_BYTES: usize = 16 * 1024 * 1024;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum AlphaMode {
    Normal = 0,
    Opaque255 = 1,
    Transparent0 = 2,
    Mask1Bit = 3,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum ColorTransform {
    None = 0,
    SubGreen = 1,
}

impl ColorTransform {
    fn from_u8(v: u8) -> Result<Self, Box<dyn Error>> {
        match v {
            0 => Ok(ColorTransform::None),
            1 => Ok(ColorTransform::SubGreen),
            _ => Err(format!("unsupported color transform id: {v}").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ColorTransform::None => "none",
            ColorTransform::SubGreen => "subgreen",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum DecodeSafety {
    Strict,
    Relaxed,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum ContainerFormat {
    V1,
    Legacy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Predictor {
    None = 0,
    Delta = 1,
    Paeth = 2,
    Up = 3,
}

impl Predictor {
    fn from_u8(v: u8) -> Result<Self, Box<dyn Error>> {
        match v {
            0 => Ok(Predictor::None),
            1 => Ok(Predictor::Delta),
            2 => Ok(Predictor::Paeth),
            3 => Ok(Predictor::Up),
            _ => Err(format!("unsupported predictor id: {v}").into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Predictor::None => "none",
            Predictor::Delta => "delta",
            Predictor::Paeth => "paeth",
            Predictor::Up => "up",
        }
    }

    fn uses_prev_row(self) -> bool {
        matches!(self, Predictor::Up | Predictor::Paeth)
    }

    fn needs_scratch(self) -> bool {
        self == Predictor::Paeth
    }

    fn auto_rank(self) -> u8 {
        match self {
            Predictor::Delta => 0,
            Predictor::Up => 1,
            Predictor::Paeth => 2,
            Predictor::None => 3,
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Encode {
            input,
            output,
            recursive,
        } => encode_cmd(input, output, recursive, cli.verbose),

        Command::Decode {
            input,
            output,
            to,
            safety,
            dump_meta,
        } => decode_cmd(input, output, to, safety, dump_meta, cli.verbose),

        Command::Open { input, safety } => open_cmd(input, safety, cli.verbose),

        Command::Inspect {
            input,
            safety,
            dump_meta,
        } => inspect_cmd(input, safety, dump_meta, cli.verbose),

        Command::Atlas {
            inputs,
            output,
            max_side,
            pad,
        } => {
            if inputs.is_empty() {
                return Err("atlas: need at least one INPUT (file or directory)".into());
            }
            atlas::atlas_cmd(inputs, output, max_side, pad, cli.verbose)
        }
    }
}

#[derive(Parser)]
#[command(
    name = "vrawtex",
    about = "vrawtex encoder/decoder/viewer (planar U8 + zstd)"
)]
struct Cli {
    /// Verbose stats
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode any image file or directory into .vrawtex
    Encode {
        /// Input image OR directory
        input: PathBuf,

        /// Output .vrawtex (only for single-file encode).
        /// If not set, defaults to input with .vrawtex
        output: Option<PathBuf>,

        /// If input is a directory: process it recursively into sibling VRAWTEXed/
        #[arg(short = 'r', long = "recursive")]
        recursive: bool,
    },

    /// Decode .vrawtex into RAW or PNG
    Decode {
        /// Input .vrawtex file
        input: PathBuf,

        /// Output base name (optional). If not set, uses input name without extension.
        output: Option<PathBuf>,

        /// Decode target: raw (planar) or png
        #[arg(short = 't', long = "to", value_enum, default_value_t = DecodeFormat::Png)]
        to: DecodeFormat,

        /// Safety mode for decode/open path.
        #[arg(long = "safety", value_enum, default_value_t = DecodeSafety::Strict)]
        safety: DecodeSafety,

        /// Optional JSON output path for parsed atlas metadata.
        #[arg(long = "dump-meta")]
        dump_meta: Option<PathBuf>,
    },

    /// Open .vrawtex in a window (viewer), without writing PNG/RAW
    Open {
        /// Input .vrawtex file
        input: PathBuf,

        /// Safety mode for decode/open path.
        #[arg(long = "safety", value_enum, default_value_t = DecodeSafety::Strict)]
        safety: DecodeSafety,
    },

    /// Inspect container headers/metadata without exporting image data
    Inspect {
        /// Input .vrawtex file
        input: PathBuf,

        /// Safety mode for decode/open path.
        #[arg(long = "safety", value_enum, default_value_t = DecodeSafety::Strict)]
        safety: DecodeSafety,

        /// Optional JSON output path for parsed atlas metadata.
        #[arg(long = "dump-meta")]
        dump_meta: Option<PathBuf>,
    },

    /// Build texture atlas .vrawtex from many images (and/or directories)
    Atlas {
        /// Input images and/or directories (directories scanned recursively for images)
        #[arg(value_name = "INPUTS")]
        inputs: Vec<PathBuf>,

        /// Output file (optional). Default: ./atlas.vrawtex
        #[arg(short = 'o', long = "output")]
        output: Option<PathBuf>,

        /// Max atlas side (default 16384)
        #[arg(long = "max-side")]
        max_side: Option<u32>,

        /// Padding pixels around each texture (default 2)
        #[arg(long = "pad")]
        pad: Option<u32>,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DecodeFormat {
    Raw,
    Png,
}

/// FLAGS layout:
/// 31..16 = PIXFMT (u16)
/// 15..8  = QVAL   (u8)  <-- feature byte
/// 7..0   = CHANS  (u8)
fn build_flags(pixfmt: u16, qval: u8, chans: u8) -> u32 {
    ((pixfmt as u32) << 16) | ((qval as u32) << 8) | (chans as u32)
}

fn parse_flags(flags: u32) -> (u16, u8, u8) {
    let pixfmt = (flags >> 16) as u16;
    let qval = ((flags >> 8) & 0xFF) as u8;
    let chans = (flags & 0xFF) as u8;
    (pixfmt, qval, chans)
}

/// DIMMSK = (width << 32) | height
fn build_dimmask(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | (height as u64)
}

fn parse_dimmask(dimmask: u64) -> (u32, u32) {
    let width = (dimmask >> 32) as u32;
    let height = (dimmask & 0xFFFF_FFFF) as u32;
    (width, height)
}

fn human_mb(bytes: u64) -> String {
    let mb = bytes as f64 / (1024.0 * 1024.0);
    format!("{:.1}MB", mb)
}

fn channel_name(idx: usize, chans: u8) -> String {
    if chans == 4 {
        match idx {
            0 => "R".to_string(),
            1 => "G".to_string(),
            2 => "B".to_string(),
            3 => "A".to_string(),
            _ => format!("C{}", idx),
        }
    } else {
        format!("C{}", idx)
    }
}

fn format_duration_ns(d: Duration) -> String {
    let secs = d.as_secs();
    let nanos = d.subsec_nanos();
    if secs == 0 {
        format!("0.{:09} sec", nanos)
    } else {
        format!("{secs}.{nanos:09} sec")
    }
}

struct EncChannel {
    name: &'static str,
    orig_size: u64,
    comp_size: u64,
    predictor: Predictor,
    data: Vec<u8>,
}

#[derive(Clone, Debug)]
struct PredictorEval {
    predictor: Predictor,
    size: usize,
}

#[derive(Clone, Debug)]
struct ChannelAutoChoice {
    chosen: Predictor,
    chosen_size: usize,
    evals: Vec<PredictorEval>,
}

#[derive(Clone, Debug)]
struct RgbTransformChoice {
    transform: ColorTransform,
    total_size: usize,
    channels: Vec<ChannelAutoChoice>,
}

/// SIMD-friendly: RGBA row -> 4 planar rows
fn deinterleave_rgba_row_to_planar(
    row_rgba: &[u8],
    dst_r: &mut [u8],
    dst_g: &mut [u8],
    dst_b: &mut [u8],
    dst_a: &mut [u8],
) {
    let len = dst_r.len();
    debug_assert_eq!(dst_g.len(), len);
    debug_assert_eq!(dst_b.len(), len);
    debug_assert_eq!(dst_a.len(), len);
    debug_assert_eq!(row_rgba.len(), len * 4);

    unsafe {
        let mut src = row_rgba.as_ptr();
        let mut pr = dst_r.as_mut_ptr();
        let mut pg = dst_g.as_mut_ptr();
        let mut pb = dst_b.as_mut_ptr();
        let mut pa = dst_a.as_mut_ptr();

        for _ in 0..len {
            *pr = *src;
            *pg = *src.add(1);
            *pb = *src.add(2);
            *pa = *src.add(3);

            src = src.add(4);
            pr = pr.add(1);
            pg = pg.add(1);
            pb = pb.add(1);
            pa = pa.add(1);
        }
    }
}

/// SIMD-friendly: 4 planar planes -> RGBA interleaved
fn interleave_planar_rgba(planes: &[Vec<u8>], out: &mut [u8]) {
    assert!(planes.len() >= 4);
    let n = planes[0].len();
    debug_assert_eq!(planes[1].len(), n);
    debug_assert_eq!(planes[2].len(), n);
    debug_assert_eq!(planes[3].len(), n);
    debug_assert_eq!(out.len(), n * 4);

    unsafe {
        let mut pr = planes[0].as_ptr();
        let mut pg = planes[1].as_ptr();
        let mut pb = planes[2].as_ptr();
        let mut pa = planes[3].as_ptr();
        let mut po = out.as_mut_ptr();

        for _ in 0..n {
            *po = *pr;
            po = po.add(1);
            pr = pr.add(1);

            *po = *pg;
            po = po.add(1);
            pg = pg.add(1);

            *po = *pb;
            po = po.add(1);
            pb = pb.add(1);

            *po = *pa;
            po = po.add(1);
            pa = pa.add(1);
        }
    }
}

fn apply_color_transform_inplace(
    row_r: &mut [u8],
    row_g: &mut [u8],
    row_b: &mut [u8],
    transform: ColorTransform,
) {
    debug_assert_eq!(row_r.len(), row_g.len());
    debug_assert_eq!(row_r.len(), row_b.len());

    match transform {
        ColorTransform::None => {}
        ColorTransform::SubGreen => {
            for i in 0..row_r.len() {
                let g = row_g[i];
                row_r[i] = row_r[i].wrapping_sub(g);
                row_b[i] = row_b[i].wrapping_sub(g);
            }
        }
    }
}

fn inverse_color_transform_rgb_planes_inplace(
    planes: &mut [Vec<u8>],
    transform: ColorTransform,
) -> Result<(), Box<dyn Error>> {
    if transform == ColorTransform::None {
        return Ok(());
    }
    if planes.len() < 3 {
        return Err("color transform requires at least RGB planes".into());
    }

    let (r_planes, gb_planes) = planes.split_at_mut(1);
    let row_r = &mut r_planes[0];
    let (g_planes, b_planes) = gb_planes.split_at_mut(1);
    let row_g = &g_planes[0];
    let row_b = &mut b_planes[0];

    if row_r.len() != row_g.len() || row_r.len() != row_b.len() {
        return Err("color transform plane size mismatch".into());
    }

    match transform {
        ColorTransform::None => {}
        ColorTransform::SubGreen => {
            for i in 0..row_r.len() {
                let g = row_g[i];
                row_r[i] = row_r[i].wrapping_add(g);
                row_b[i] = row_b[i].wrapping_add(g);
            }
        }
    }

    Ok(())
}

fn extract_transformed_channel_row(
    row_rgba: &[u8],
    width: usize,
    channel: usize,
    color_transform: ColorTransform,
    out: &mut [u8],
) {
    debug_assert_eq!(row_rgba.len(), width * 4);
    debug_assert_eq!(out.len(), width);

    match (color_transform, channel) {
        (ColorTransform::None, 0..=3) => {
            for x in 0..width {
                out[x] = row_rgba[x * 4 + channel];
            }
        }
        (ColorTransform::SubGreen, 0) => {
            for x in 0..width {
                let base = x * 4;
                let r = row_rgba[base];
                let g = row_rgba[base + 1];
                out[x] = r.wrapping_sub(g);
            }
        }
        (ColorTransform::SubGreen, 1) => {
            for x in 0..width {
                out[x] = row_rgba[x * 4 + 1];
            }
        }
        (ColorTransform::SubGreen, 2) => {
            for x in 0..width {
                let base = x * 4;
                let b = row_rgba[base + 2];
                let g = row_rgba[base + 1];
                out[x] = b.wrapping_sub(g);
            }
        }
        (ColorTransform::SubGreen, 3) => {
            for x in 0..width {
                out[x] = row_rgba[x * 4 + 3];
            }
        }
        _ => {}
    }
}

fn delta_encode_row(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    if src.is_empty() {
        return;
    }
    dst[0] = src[0];
    for i in 1..src.len() {
        dst[i] = src[i].wrapping_sub(src[i - 1]);
    }
}

fn delta_encode_row_inplace(row: &mut [u8]) {
    if row.is_empty() {
        return;
    }
    let mut prev = row[0];
    for i in 1..row.len() {
        let cur = row[i];
        row[i] = cur.wrapping_sub(prev);
        prev = cur;
    }
}

fn delta_decode_row_inplace(row: &mut [u8]) {
    if row.is_empty() {
        return;
    }
    let mut prev = row[0];
    for i in 1..row.len() {
        row[i] = row[i].wrapping_add(prev);
        prev = row[i];
    }
}

fn up_encode_row(src: &[u8], prev_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), prev_row.len());

    for i in 0..src.len() {
        dst[i] = src[i].wrapping_sub(prev_row[i]);
    }
}

fn up_encode_row_inplace(row: &mut [u8], prev_row: &[u8]) {
    debug_assert_eq!(row.len(), prev_row.len());

    for i in 0..row.len() {
        row[i] = row[i].wrapping_sub(prev_row[i]);
    }
}

fn up_decode_row_inplace(row: &mut [u8], prev_row: &[u8]) {
    debug_assert_eq!(row.len(), prev_row.len());

    for i in 0..row.len() {
        row[i] = row[i].wrapping_add(prev_row[i]);
    }
}

#[inline(always)]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).unsigned_abs();
    let pb = (p - b).unsigned_abs();
    let pc = (p - c).unsigned_abs();

    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

fn paeth_encode_row(src: &[u8], prev_row: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), prev_row.len());

    for i in 0..src.len() {
        let a = if i == 0 { 0 } else { src[i - 1] };
        let b = prev_row[i];
        let c = if i == 0 { 0 } else { prev_row[i - 1] };
        dst[i] = src[i].wrapping_sub(paeth_predictor(a, b, c));
    }
}

fn paeth_decode_row_inplace(row: &mut [u8], prev_row: &[u8]) {
    debug_assert_eq!(row.len(), prev_row.len());

    for i in 0..row.len() {
        let a = if i == 0 { 0 } else { row[i - 1] };
        let b = prev_row[i];
        let c = if i == 0 { 0 } else { prev_row[i - 1] };
        row[i] = row[i].wrapping_add(paeth_predictor(a, b, c));
    }
}

fn encode_row_with_predictor(src: &[u8], prev_row: &[u8], predictor: Predictor, dst: &mut [u8]) {
    debug_assert_eq!(src.len(), dst.len());
    if predictor.uses_prev_row() {
        debug_assert_eq!(src.len(), prev_row.len());
    }

    match predictor {
        Predictor::None => dst.copy_from_slice(src),
        Predictor::Delta => delta_encode_row(src, dst),
        Predictor::Up => up_encode_row(src, prev_row, dst),
        Predictor::Paeth => paeth_encode_row(src, prev_row, dst),
    }
}

fn append_predicted_row(
    row: &mut [u8],
    prev_row: &[u8],
    predictor: Predictor,
    paeth_scratch: &mut [u8],
    out: &mut Vec<u8>,
) {
    match predictor {
        Predictor::None => out.extend_from_slice(row),
        Predictor::Delta => {
            delta_encode_row_inplace(row);
            out.extend_from_slice(row);
        }
        Predictor::Up => {
            up_encode_row_inplace(row, prev_row);
            out.extend_from_slice(row);
        }
        Predictor::Paeth => {
            encode_row_with_predictor(row, prev_row, predictor, paeth_scratch);
            out.extend_from_slice(paeth_scratch);
        }
    }
}

fn decode_plane_with_predictor_inplace(
    plane: &mut [u8],
    width: usize,
    height: usize,
    predictor: Predictor,
) -> Result<(), Box<dyn Error>> {
    if predictor == Predictor::None || width == 0 || height == 0 {
        return Ok(());
    }

    let expected = width
        .checked_mul(height)
        .ok_or("predictor decode overflow")?;
    if plane.len() != expected {
        return Err(format!(
            "predictor decode plane size mismatch: got {}, expected {}",
            plane.len(),
            expected
        )
        .into());
    }

    match predictor {
        Predictor::None => {}
        Predictor::Delta => {
            for y in 0..height {
                let row = &mut plane[y * width..(y + 1) * width];
                delta_decode_row_inplace(row);
            }
        }
        Predictor::Up => {
            let zero_prev = vec![0u8; width];
            for y in 0..height {
                let row_start = y * width;
                let (head, tail) = plane.split_at_mut(row_start);
                let row = &mut tail[..width];
                let prev = if y == 0 {
                    zero_prev.as_slice()
                } else {
                    &head[row_start - width..row_start]
                };
                up_decode_row_inplace(row, prev);
            }
        }
        Predictor::Paeth => {
            let zero_prev = vec![0u8; width];
            for y in 0..height {
                let row_start = y * width;
                let (head, tail) = plane.split_at_mut(row_start);
                let row = &mut tail[..width];
                let prev = if y == 0 {
                    zero_prev.as_slice()
                } else {
                    &head[row_start - width..row_start]
                };
                paeth_decode_row_inplace(row, prev);
            }
        }
    }
    Ok(())
}

fn detect_alpha_mode(rgba_bytes: &[u8]) -> AlphaMode {
    if rgba_bytes.len() < 4 {
        return AlphaMode::Normal;
    }
    let first = rgba_bytes[3];
    let mut all_same = true;
    let mut binary = true;

    for i in (3..rgba_bytes.len()).step_by(4) {
        let a = rgba_bytes[i];
        if a != first {
            all_same = false;
        }
        if a != 0 && a != 255 {
            binary = false;
        }
        if !all_same && !binary {
            break;
        }
    }

    if all_same {
        if first == 255 {
            AlphaMode::Opaque255
        } else if first == 0 {
            AlphaMode::Transparent0
        } else {
            AlphaMode::Normal
        }
    } else if binary {
        AlphaMode::Mask1Bit
    } else {
        AlphaMode::Normal
    }
}

/// Feature byte bits:
/// bit0 = legacy_delta_or_predictor
/// bit1..2 = alpha_mode (0..3)
/// bit3 = has_meta
/// bit4..5 = color_transform (v2+)
fn feature_byte(
    has_predictor: bool,
    alpha_mode: AlphaMode,
    has_meta: bool,
    color_transform: ColorTransform,
) -> u8 {
    let d = if has_predictor { 1u8 } else { 0u8 };
    let am = (alpha_mode as u8) & 0x3;
    let m = if has_meta { 1u8 } else { 0u8 };
    let ct = ((color_transform as u8) & 0x3) << 4;
    d | (am << 1) | (m << 3) | ct
}

fn parse_feature_byte(
    version: u8,
    qval: u8,
) -> Result<(bool, AlphaMode, bool, ColorTransform), Box<dyn Error>> {
    let delta = (qval & 1) != 0;
    let am = (qval >> 1) & 0x3;
    let has_meta = ((qval >> 3) & 1) != 0;

    let alpha_mode = match am {
        0 => AlphaMode::Normal,
        1 => AlphaMode::Opaque255,
        2 => AlphaMode::Transparent0,
        3 => AlphaMode::Mask1Bit,
        _ => AlphaMode::Normal,
    };

    let color_transform = if version >= FILE_VERSION {
        ColorTransform::from_u8((qval >> 4) & 0x3)?
    } else {
        ColorTransform::None
    };

    Ok((delta, alpha_mode, has_meta, color_transform))
}

#[derive(Clone, Debug)]
struct ParsedContainer {
    format: ContainerFormat,
    version: u8,
    pixfmt_bits: u16,
    qval: u8,
    chans: u8,
    legacy_delta: bool,
    alpha_mode: AlphaMode,
    has_meta: bool,
    color_transform: ColorTransform,
    width: u32,
    height: u32,
    plane_size: u64,
    raw_planar_size: u64,
    packed_alpha_size: u64,
    store_alpha_stream: bool,
    expected_sizes: Vec<u64>,
    meta_raw: Option<Vec<u8>>,
    stream_offset: usize,
}

#[derive(Clone, Debug)]
struct StreamHeader {
    orig_size: u64,
    comp_size: u64,
    predictor: Predictor,
    data_offset: usize,
}

#[derive(Clone, Debug, Serialize)]
struct AtlasMetaDump {
    pad: u16,
    entries: Vec<atlas::AtlasRect>,
}

fn read_u32_le(data: &[u8], offset: usize, what: &str) -> Result<u32, Box<dyn Error>> {
    if offset + 4 > data.len() {
        return Err(format!("truncated file: {what}").into());
    }
    let mut arr = [0u8; 4];
    arr.copy_from_slice(&data[offset..offset + 4]);
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_le(data: &[u8], offset: usize, what: &str) -> Result<u64, Box<dyn Error>> {
    if offset + 8 > data.len() {
        return Err(format!("truncated file: {what}").into());
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&data[offset..offset + 8]);
    Ok(u64::from_le_bytes(arr))
}

fn parse_container_header(
    data: &[u8],
    safety: DecodeSafety,
) -> Result<(ContainerFormat, u8, u32, u64, usize), Box<dyn Error>> {
    if data.len() < 12 {
        return Err("file too small to be vrawtex (need at least 12 bytes)".into());
    }

    if data.len() >= HEADER_V1_SIZE && data[0..4] == FILE_MAGIC {
        let version = data[4];
        if !(FILE_VERSION_V1..=FILE_VERSION).contains(&version) {
            return Err(format!(
                "unsupported vrawtex version: got {}, supported {}..={}",
                version, FILE_VERSION_V1, FILE_VERSION
            )
            .into());
        }
        let flags = read_u32_le(data, 5, "flags")?;
        let dimmask = read_u64_le(data, 9, "dimmask")?;
        Ok((ContainerFormat::V1, version, flags, dimmask, HEADER_V1_SIZE))
    } else if safety == DecodeSafety::Relaxed {
        let flags = read_u32_le(data, 0, "legacy flags")?;
        let dimmask = read_u64_le(data, 4, "legacy dimmask")?;
        Ok((ContainerFormat::Legacy, 0, flags, dimmask, 12))
    } else {
        Err("missing vrawtex magic; strict mode rejects non-owned files".into())
    }
}

fn parse_container(data: &[u8], safety: DecodeSafety) -> Result<ParsedContainer, Box<dyn Error>> {
    let (format, version, flags, dimmask, mut offset) = parse_container_header(data, safety)?;
    let (pixfmt_bits, qval, chans) = parse_flags(flags);

    if chans == 0 {
        return Err("invalid header: channels == 0".into());
    }
    if chans < 3 || chans > 4 {
        return Err(format!("unsupported channel count in container: {chans}").into());
    }
    if pixfmt_bits != 0x0001 {
        return Err(format!("unsupported pixel format: pixfmt=0x{pixfmt_bits:04X}").into());
    }

    let (legacy_delta, alpha_mode, has_meta, color_transform) = parse_feature_byte(version, qval)?;
    let (width, height) = parse_dimmask(dimmask);
    if width == 0 || height == 0 {
        return Err("invalid dimensions: width/height must be > 0".into());
    }

    let pixels = (width as u64)
        .checked_mul(height as u64)
        .ok_or("width*height overflow")?;
    let plane_size = pixels;
    let raw_planar_size = plane_size
        .checked_mul(chans as u64)
        .ok_or("raw planar overflow")?;

    if plane_size > usize::MAX as u64 {
        return Err("image too large (usize overflow)".into());
    }

    if safety == DecodeSafety::Strict {
        if width > MAX_STRICT_SIDE || height > MAX_STRICT_SIDE {
            return Err(format!(
                "strict mode: dimensions {}x{} exceed max side {}",
                width, height, MAX_STRICT_SIDE
            )
            .into());
        }
        if pixels > MAX_STRICT_PIXELS {
            return Err(format!(
                "strict mode: pixel count {} exceeds limit {}",
                pixels, MAX_STRICT_PIXELS
            )
            .into());
        }
        if raw_planar_size > MAX_STRICT_RAW_PLANAR_BYTES {
            return Err(format!(
                "strict mode: raw planar size {} exceeds limit {}",
                raw_planar_size, MAX_STRICT_RAW_PLANAR_BYTES
            )
            .into());
        }
    }

    let meta_raw = if has_meta {
        let meta_len = read_u32_le(data, offset, "meta_len")? as usize;
        offset += 4;
        if safety == DecodeSafety::Strict && meta_len > MAX_STRICT_META_BYTES {
            return Err(format!(
                "strict mode: meta block too large ({} bytes > {} bytes)",
                meta_len, MAX_STRICT_META_BYTES
            )
            .into());
        }
        if offset + meta_len > data.len() {
            return Err("truncated file: meta bytes missing".into());
        }
        let bytes = data[offset..offset + meta_len].to_vec();
        offset += meta_len;
        Some(bytes)
    } else {
        None
    };

    let packed_alpha_size = (pixels + 7) / 8;
    let store_alpha_stream =
        !(alpha_mode == AlphaMode::Opaque255 || alpha_mode == AlphaMode::Transparent0);
    let stored_streams = if store_alpha_stream { 4usize } else { 3usize };
    let mut expected_sizes = vec![plane_size; stored_streams];
    if store_alpha_stream && alpha_mode == AlphaMode::Mask1Bit {
        expected_sizes[3] = packed_alpha_size;
    }

    Ok(ParsedContainer {
        format,
        version,
        pixfmt_bits,
        qval,
        chans,
        legacy_delta,
        alpha_mode,
        has_meta,
        color_transform,
        width,
        height,
        plane_size,
        raw_planar_size,
        packed_alpha_size,
        store_alpha_stream,
        expected_sizes,
        meta_raw,
        stream_offset: offset,
    })
}

fn legacy_stream_predictors(parsed: &ParsedContainer) -> Vec<Predictor> {
    let mut out = vec![
        if parsed.legacy_delta {
            Predictor::Delta
        } else {
            Predictor::None
        };
        parsed.expected_sizes.len()
    ];

    if parsed.store_alpha_stream && parsed.alpha_mode == AlphaMode::Mask1Bit {
        out[3] = Predictor::None;
    }
    out
}

fn read_stream_headers(
    data: &[u8],
    mut offset: usize,
    expected_sizes: &[u64],
    format: ContainerFormat,
    legacy_predictors: &[Predictor],
) -> Result<(Vec<StreamHeader>, usize), Box<dyn Error>> {
    let mut headers: Vec<StreamHeader> = Vec::with_capacity(expected_sizes.len());

    for i in 0..expected_sizes.len() {
        let hdr_size = match format {
            ContainerFormat::V1 => STREAM_HEADER_V1_SIZE,
            ContainerFormat::Legacy => 16,
        };
        if offset + hdr_size > data.len() {
            return Err("truncated file while reading stream header".into());
        }

        let orig_size = read_u64_le(data, offset, "stream orig_size")?;
        let comp_size = read_u64_le(data, offset + 8, "stream comp_size")?;
        let predictor = if format == ContainerFormat::V1 {
            Predictor::from_u8(data[offset + 16])?
        } else {
            legacy_predictors.get(i).copied().unwrap_or(Predictor::None)
        };
        offset += hdr_size;

        let expected = expected_sizes[i];
        if orig_size != expected {
            return Err(format!(
                "stream orig_size mismatch: got {orig_size}, expected {expected} (stream #{i})"
            )
            .into());
        }

        if comp_size > usize::MAX as u64 {
            return Err("stream comp_size too large for this build".into());
        }
        let cs = comp_size as usize;
        if offset + cs > data.len() {
            return Err("truncated file: comp_size goes past EOF".into());
        }

        headers.push(StreamHeader {
            orig_size,
            comp_size,
            predictor,
            data_offset: offset,
        });
        offset += cs;
    }

    Ok((headers, offset))
}

fn read_streams(
    data: &[u8],
    offset: usize,
    expected_sizes: &[u64],
    format: ContainerFormat,
    legacy_predictors: &[Predictor],
) -> Result<(Vec<Vec<u8>>, Vec<u64>, Vec<Predictor>, usize), Box<dyn Error>> {
    let (headers, final_offset) =
        read_stream_headers(data, offset, expected_sizes, format, legacy_predictors)?;

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(headers.len());
    let mut comp_sizes: Vec<u64> = Vec::with_capacity(headers.len());
    let mut predictors: Vec<Predictor> = Vec::with_capacity(headers.len());

    for hdr in &headers {
        if hdr.orig_size > usize::MAX as u64 {
            return Err("stream orig_size too large for this build".into());
        }
        let cs = hdr.comp_size as usize;
        let comp_slice = &data[hdr.data_offset..hdr.data_offset + cs];

        let decompressed = bulk::decompress(comp_slice, hdr.orig_size as usize)?;
        if decompressed.len() != hdr.orig_size as usize {
            return Err(format!(
                "decompressed size mismatch: expected {}, got {}",
                hdr.orig_size,
                decompressed.len()
            )
            .into());
        }

        planes.push(decompressed);
        comp_sizes.push(hdr.comp_size);
        predictors.push(hdr.predictor);
    }

    Ok((planes, comp_sizes, predictors, final_offset))
}

fn parse_atlas_meta(meta_raw: Option<&[u8]>) -> Option<atlas::AtlasMeta> {
    let bytes = meta_raw?;
    atlas::decode_meta(bytes).ok()
}

fn dump_atlas_meta(path: &Path, meta: &atlas::AtlasMeta) -> Result<(), Box<dyn Error>> {
    let dump = AtlasMetaDump {
        pad: meta.0,
        entries: atlas::meta_rects(meta),
    };
    let json = serde_json::to_vec_pretty(&dump)?;
    fs::write(path, json)?;
    Ok(())
}

fn predictor_candidates() -> [Predictor; 3] {
    // Auto-select stays limited to the cheap predictors to avoid slowing down
    // the full encode path on large atlases.
    [Predictor::Delta, Predictor::Up, Predictor::None]
}

fn color_transform_candidates() -> [ColorTransform; 2] {
    [ColorTransform::None, ColorTransform::SubGreen]
}

fn format_predictor_evals(evals: &[PredictorEval]) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(evals.len());
    for eval in evals {
        parts.push(format!("{}={}", eval.predictor.as_str(), eval.size));
    }
    parts.join(" ")
}

fn best_predictive_eval(evals: &[PredictorEval]) -> Option<&PredictorEval> {
    evals
        .iter()
        .filter(|eval| eval.predictor != Predictor::None)
        .min_by_key(|eval| (eval.size, eval.predictor.auto_rank()))
}

fn choose_channel_predictor_from_evals(
    evals: &[PredictorEval],
) -> Result<(Predictor, usize), Box<dyn Error>> {
    let best_predictive = best_predictive_eval(evals);
    let none_eval = evals.iter().find(|eval| eval.predictor == Predictor::None);

    match (best_predictive, none_eval) {
        (Some(best_pred), Some(none)) => {
            let lhs = (none.size as u64).saturating_mul(10_000);
            let rhs = (best_pred.size as u64)
                .saturating_mul(10_000u64.saturating_sub(PREDICTOR_NONE_MIN_GAIN_BPS));
            if lhs <= rhs {
                Ok((Predictor::None, none.size))
            } else {
                Ok((best_pred.predictor, best_pred.size))
            }
        }
        (Some(best_pred), None) => Ok((best_pred.predictor, best_pred.size)),
        (None, Some(none)) => Ok((Predictor::None, none.size)),
        (None, None) => Err("predictor auto-select produced no candidates".into()),
    }
}

fn collect_channel_sample(
    rgba_bytes: &[u8],
    width: usize,
    height: usize,
    channel: usize,
    color_transform: ColorTransform,
    predictor: Predictor,
    sample_limit_bytes: usize,
) -> Vec<u8> {
    if width == 0 || height == 0 || sample_limit_bytes == 0 {
        return Vec::new();
    }

    let max_bytes = sample_limit_bytes.min(width * height);
    let max_rows = (max_bytes / width).max(1);
    let rows_to_take = max_rows.min(height);
    let mut sample = Vec::with_capacity(rows_to_take.saturating_mul(width));
    let mut prev = vec![0u8; width];
    let mut row = vec![0u8; width];
    let mut transformed = vec![0u8; width];
    let stride = width * 4;
    let mut sampled_prev_y: Option<usize> = None;

    for i in 0..rows_to_take {
        let y = if rows_to_take == height {
            i
        } else {
            (i * height) / rows_to_take
        };
        if sampled_prev_y == Some(y) {
            continue;
        }
        sampled_prev_y = Some(y);
        let row_rgba = &rgba_bytes[y * stride..(y + 1) * stride];
        extract_transformed_channel_row(row_rgba, width, channel, color_transform, &mut row);

        encode_row_with_predictor(&row, &prev, predictor, &mut transformed);
        let remain = sample_limit_bytes - sample.len();
        if remain == 0 {
            break;
        }
        if remain >= width {
            sample.extend_from_slice(&transformed);
        } else {
            sample.extend_from_slice(&transformed[..remain]);
            break;
        }

        prev.copy_from_slice(&row);
        if sample.len() >= sample_limit_bytes {
            break;
        }
    }

    sample
}

fn choose_predictor_for_channel_sample(
    rgba_bytes: &[u8],
    width: usize,
    height: usize,
    channel: usize,
) -> Result<ChannelAutoChoice, Box<dyn Error>> {
    choose_predictor_for_channel_sample_with_transform(
        rgba_bytes,
        width,
        height,
        channel,
        ColorTransform::None,
    )
}

fn choose_predictor_for_channel_sample_with_transform(
    rgba_bytes: &[u8],
    width: usize,
    height: usize,
    channel: usize,
    color_transform: ColorTransform,
) -> Result<ChannelAutoChoice, Box<dyn Error>> {
    let mut evals: Vec<PredictorEval> = Vec::with_capacity(predictor_candidates().len());

    for candidate in predictor_candidates() {
        let sample = collect_channel_sample(
            rgba_bytes,
            width,
            height,
            channel,
            color_transform,
            candidate,
            PREDICTOR_SAMPLE_BYTES,
        );
        let comp = bulk::compress(&sample, AUTO_SELECT_ZSTD_LEVEL)?;
        evals.push(PredictorEval {
            predictor: candidate,
            size: comp.len(),
        });
    }

    let (chosen, chosen_size) = choose_channel_predictor_from_evals(&evals)?;
    Ok(ChannelAutoChoice {
        chosen,
        chosen_size,
        evals,
    })
}

fn choose_color_transform_and_predictors(
    rgba_bytes: &[u8],
    width: usize,
    height: usize,
) -> Result<(ColorTransform, [Predictor; 3], Vec<RgbTransformChoice>), Box<dyn Error>> {
    let mut decisions: Vec<RgbTransformChoice> =
        Vec::with_capacity(color_transform_candidates().len());

    for color_transform in color_transform_candidates() {
        let mut total_size = 0usize;
        let mut channels: Vec<ChannelAutoChoice> = Vec::with_capacity(3);

        for channel in 0..3 {
            let choice = choose_predictor_for_channel_sample_with_transform(
                rgba_bytes,
                width,
                height,
                channel,
                color_transform,
            )?;
            total_size = total_size.saturating_add(choice.chosen_size);
            channels.push(choice);
        }

        decisions.push(RgbTransformChoice {
            transform: color_transform,
            total_size,
            channels,
        });
    }

    let best_idx = decisions
        .iter()
        .enumerate()
        .min_by_key(|(_, decision)| {
            (
                decision.total_size,
                match decision.transform {
                    ColorTransform::None => 0u8,
                    ColorTransform::SubGreen => 1u8,
                },
            )
        })
        .map(|(idx, _)| idx)
        .ok_or("transform auto-select produced no candidates")?;

    let none_idx = decisions
        .iter()
        .position(|decision| decision.transform == ColorTransform::None)
        .ok_or("transform auto-select missing identity transform")?;

    let chosen_idx = if decisions[best_idx].transform != ColorTransform::None {
        let best_total = decisions[best_idx].total_size as u64;
        let none_total = decisions[none_idx].total_size as u64;
        let lhs = best_total.saturating_mul(10_000);
        let rhs = none_total.saturating_mul(10_000u64.saturating_sub(COLOR_TRANSFORM_MIN_GAIN_BPS));
        if lhs <= rhs { best_idx } else { none_idx }
    } else {
        best_idx
    };

    let chosen = &decisions[chosen_idx];
    if chosen.channels.len() != 3 {
        return Err("transform auto-select produced invalid RGB predictor set".into());
    }

    let predictors = [
        chosen.channels[0].chosen,
        chosen.channels[1].chosen,
        chosen.channels[2].chosen,
    ];

    Ok((chosen.transform, predictors, decisions))
}

fn encode_cmd(
    input: PathBuf,
    output: Option<PathBuf>,
    recursive: bool,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    if input.is_dir() {
        if !recursive {
            return Err("input is a directory; use -r/--recursive to process it".into());
        }
        encode_dir(&input, verbose)
    } else {
        let out_path = output.unwrap_or_else(|| default_encode_output_path(&input));
        encode_one(&input, &out_path, verbose)
    }
}

fn encode_dir(root: &Path, verbose: bool) -> Result<(), Box<dyn Error>> {
    if verbose {
        println!(
            "[vrawtex] Recursive encode of directory: {}",
            root.display()
        );
    }

    let root = root.canonicalize()?;
    let parent = root.parent().unwrap_or(&root);
    let target_root = parent.join("VRAWTEXed");
    fs::create_dir_all(&target_root)?;

    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for entry in WalkDir::new(&root).follow_links(true) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                if verbose {
                    eprintln!("[vrawtex] Walk error: {e}");
                }
                failed += 1;
                continue;
            }
        };
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        let is_image = matches!(
            ext.as_str(),
            "png" | "jpg" | "jpeg" | "bmp" | "tga" | "tif" | "tiff" | "gif"
        );

        if !is_image {
            skipped += 1;
            continue;
        }

        let rel = match path.strip_prefix(&root) {
            Ok(r) => r,
            Err(_) => {
                failed += 1;
                continue;
            }
        };

        let mut out_path = target_root.join(rel);
        out_path.set_extension("vrawtex");
        if let Some(parent_dir) = out_path.parent() {
            fs::create_dir_all(parent_dir)?;
        }

        match encode_one(path, &out_path, verbose) {
            Ok(_) => processed += 1,
            Err(e) => {
                failed += 1;
                if verbose {
                    eprintln!("[vrawtex] Failed to encode {}: {e}", path.display());
                }
            }
        }
    }

    println!(
        "[vrawtex] Done. Encoded {} image(s), skipped {} non-image file(s), failed {}. Output root: {}",
        processed,
        skipped,
        failed,
        target_root.display()
    );
    Ok(())
}

fn encode_one(input: &Path, out_path: &Path, verbose: bool) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();
    let original_size = fs::metadata(input).map(|m| m.len()).unwrap_or(0);

    let img = image::open(input)?;
    let (width, height) = img.dimensions();

    if verbose {
        println!(
            "[vrawtex] Encoding {} ({}x{}, RGBA8)",
            input.display(),
            width,
            height
        );
    }

    let rgba = img.to_rgba8();
    let bytes =
        encode_rgba8_with_meta_to_vec(&rgba, None, verbose, Some(original_size), start_total)?;
    fs::write(out_path, &bytes)?;

    println!(
        "Encoded {}x{} RGBA8 -> {}",
        width,
        height,
        out_path.display()
    );
    Ok(())
}

/// Делим workers_total на N стримов (3 или 4), “поровну”.
/// Гарантируем минимум 1 на стрим.
fn split_workers(workers_total: u32, streams: usize) -> Vec<u32> {
    let streams_u32 = streams as u32;
    if streams == 0 {
        return Vec::new();
    }
    let mut base = workers_total / streams_u32;
    let mut rem = workers_total % streams_u32;

    if base == 0 {
        base = 1;
        rem = 0;
    }

    let mut out = vec![base; streams];
    for i in 0..streams {
        if rem == 0 {
            break;
        }
        out[i] += 1;
        rem -= 1;
    }
    out
}

#[inline(always)]
fn worker_for_stream(workers_split: &[u32], stream_idx: usize) -> u32 {
    if workers_split.is_empty() {
        return 1;
    }
    workers_split[stream_idx.min(workers_split.len() - 1)]
}

/// Это ИМЕННО твой энкодер, но:
/// - берет уже готовую RGBA8 картинку
/// - опционально принимает meta-блок (msgpack bytes), который кладётся СРАЗУ ПОСЛЕ (flags+dimmask):
///   [flags u32][dimmask u64][meta_len u32][meta bytes][streams...]
pub(crate) fn encode_rgba8_with_meta_to_vec(
    rgba: &RgbaImage,
    meta: Option<&[u8]>,
    verbose: bool,
    original_size_opt: Option<u64>,
    start_total: Instant,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let (width, height) = rgba.dimensions();
    let rgba_bytes = rgba.as_raw();

    let width_usize = width as usize;
    let height_usize = height as usize;

    let pixels_u64 = (width as u64)
        .checked_mul(height as u64)
        .ok_or("width*height overflow")?;
    let plane_size_u64 = pixels_u64; // u8 per channel
    let raw_planar_size_u64 = plane_size_u64.checked_mul(4).ok_or("raw size overflow")?;

    if plane_size_u64 > usize::MAX as u64 {
        return Err("plane too large for this build (usize overflow)".into());
    }

    let alpha_mode = detect_alpha_mode(rgba_bytes);
    let has_meta = meta.is_some();
    let pixfmt_bits: u16 = 0x0001; // U8
    let chans: u8 = 4;

    // packed alpha bytes if mask:
    let packed_alpha_size_u64 = if alpha_mode == AlphaMode::Mask1Bit {
        (pixels_u64 + 7) / 8
    } else {
        0
    };
    if packed_alpha_size_u64 > usize::MAX as u64 {
        return Err("alpha mask too large for this build (usize overflow)".into());
    }
    let packed_alpha_size_usize = packed_alpha_size_u64 as usize;

    // streams actually stored:
    let store_alpha_stream =
        !(alpha_mode == AlphaMode::Opaque255 || alpha_mode == AlphaMode::Transparent0);
    let stored_streams = if store_alpha_stream { 4usize } else { 3usize };
    let ws = split_workers(ZSTD_WORKERS_TOTAL, stored_streams);

    let (color_transform, rgb_predictors, transform_choices) =
        choose_color_transform_and_predictors(rgba_bytes, width_usize, height_usize)?;
    let mut stream_predictors = vec![Predictor::None; stored_streams];
    stream_predictors[0] = rgb_predictors[0];
    stream_predictors[1] = rgb_predictors[1];
    stream_predictors[2] = rgb_predictors[2];

    let alpha_choice = if store_alpha_stream && alpha_mode == AlphaMode::Normal {
        Some(choose_predictor_for_channel_sample(
            rgba_bytes,
            width_usize,
            height_usize,
            3,
        )?)
    } else {
        None
    };
    if let Some(choice) = alpha_choice.as_ref() {
        stream_predictors[3] = choice.chosen;
    }
    let has_predictor = stream_predictors.iter().any(|&p| p != Predictor::None);

    if verbose {
        let mut transform_scores: Vec<String> = Vec::with_capacity(transform_choices.len());
        for choice in &transform_choices {
            transform_scores.push(format!(
                "{}={}",
                choice.transform.as_str(),
                choice.total_size
            ));
        }
        println!(
            "[vrawtex] Color transform sample: {} -> {}",
            transform_scores.join(" "),
            color_transform.as_str()
        );

        let chosen_transform = transform_choices
            .iter()
            .find(|choice| choice.transform == color_transform)
            .ok_or("missing chosen transform decision")?;
        for (channel, choice) in chosen_transform.channels.iter().enumerate() {
            println!(
                "[vrawtex] Predictor C{} [transform={}]: {} -> {}",
                channel,
                color_transform.as_str(),
                format_predictor_evals(&choice.evals),
                choice.chosen.as_str()
            );
        }
        if let Some(choice) = alpha_choice.as_ref() {
            println!(
                "[vrawtex] Predictor C3 [transform={}]: {} -> {}",
                ColorTransform::None.as_str(),
                format_predictor_evals(&choice.evals),
                choice.chosen.as_str()
            );
        }
    }

    let qval = feature_byte(has_predictor, alpha_mode, has_meta, color_transform);
    let flags = build_flags(pixfmt_bits, qval, chans);
    let dimmask = build_dimmask(width, height);

    // --- ZSTD streaming ---
    let mut enc_r = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc_r.multithread(worker_for_stream(&ws, 0))?;
    enc_r.set_pledged_src_size(Some(plane_size_u64))?;

    let mut enc_g = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc_g.multithread(worker_for_stream(&ws, 1))?;
    enc_g.set_pledged_src_size(Some(plane_size_u64))?;

    let mut enc_b = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc_b.multithread(worker_for_stream(&ws, 2))?;
    enc_b.set_pledged_src_size(Some(plane_size_u64))?;

    let mut enc_a_opt: Option<Encoder<'static, Vec<u8>>> = None;
    if store_alpha_stream {
        let pledged = if alpha_mode == AlphaMode::Mask1Bit {
            packed_alpha_size_u64
        } else {
            plane_size_u64
        };
        let mut enc_a = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
        enc_a.multithread(worker_for_stream(&ws, 3))?;
        enc_a.set_pledged_src_size(Some(pledged))?;
        enc_a_opt = Some(enc_a);
    }

    let mut buf_r = Vec::with_capacity(CHUNK_TARGET);
    let mut buf_g = Vec::with_capacity(CHUNK_TARGET);
    let mut buf_b = Vec::with_capacity(CHUNK_TARGET);

    // alpha buffers:
    let mut buf_a = Vec::with_capacity(CHUNK_TARGET);
    let mut buf_amask = Vec::with_capacity(CHUNK_TARGET);

    let mut row_r = vec![0u8; width_usize];
    let mut row_g = vec![0u8; width_usize];
    let mut row_b = vec![0u8; width_usize];
    let mut row_a = vec![0u8; width_usize];

    let use_prev_r = stream_predictors[0].uses_prev_row();
    let use_prev_g = stream_predictors[1].uses_prev_row();
    let use_prev_b = stream_predictors[2].uses_prev_row();
    let use_prev_a = store_alpha_stream
        && alpha_mode == AlphaMode::Normal
        && stream_predictors[3].uses_prev_row();

    let mut row_r_t = if stream_predictors[0].needs_scratch() {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut row_g_t = if stream_predictors[1].needs_scratch() {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut row_b_t = if stream_predictors[2].needs_scratch() {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut row_a_t = if store_alpha_stream
        && alpha_mode == AlphaMode::Normal
        && stream_predictors[3].needs_scratch()
    {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut prev_r = if use_prev_r {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut prev_g = if use_prev_g {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut prev_b = if use_prev_b {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };
    let mut prev_a = if use_prev_a {
        vec![0u8; width_usize]
    } else {
        Vec::new()
    };

    let stride = width_usize * 4;

    // mask packing state
    let mut mask_acc: u8 = 0;
    let mut mask_bits: u8 = 0;
    let mut packed_written: usize = 0;

    let start_enc = Instant::now();

    for y in 0..height_usize {
        let row_rgba = &rgba_bytes[y * stride..(y + 1) * stride];

        deinterleave_rgba_row_to_planar(row_rgba, &mut row_r, &mut row_g, &mut row_b, &mut row_a);
        apply_color_transform_inplace(&mut row_r, &mut row_g, &mut row_b, color_transform);

        append_predicted_row(
            &mut row_r,
            &prev_r,
            stream_predictors[0],
            &mut row_r_t,
            &mut buf_r,
        );
        append_predicted_row(
            &mut row_g,
            &prev_g,
            stream_predictors[1],
            &mut row_g_t,
            &mut buf_g,
        );
        append_predicted_row(
            &mut row_b,
            &prev_b,
            stream_predictors[2],
            &mut row_b_t,
            &mut buf_b,
        );

        if buf_r.len() >= CHUNK_TARGET {
            enc_r.write_all(&buf_r)?;
            buf_r.clear();
        }
        if buf_g.len() >= CHUNK_TARGET {
            enc_g.write_all(&buf_g)?;
            buf_g.clear();
        }
        if buf_b.len() >= CHUNK_TARGET {
            enc_b.write_all(&buf_b)?;
            buf_b.clear();
        }

        if store_alpha_stream {
            if alpha_mode == AlphaMode::Normal {
                append_predicted_row(
                    &mut row_a,
                    &prev_a,
                    stream_predictors[3],
                    &mut row_a_t,
                    &mut buf_a,
                );
                if buf_a.len() >= CHUNK_TARGET {
                    if let Some(enc_a) = enc_a_opt.as_mut() {
                        enc_a.write_all(&buf_a)?;
                    }
                    buf_a.clear();
                }
            } else if alpha_mode == AlphaMode::Mask1Bit {
                // pack 0/255 -> 1 bit/pixel (LSB-first)
                for &a in &row_a {
                    let bit = if a != 0 { 1u8 } else { 0u8 };
                    mask_acc |= bit << mask_bits;
                    mask_bits += 1;
                    if mask_bits == 8 {
                        buf_amask.push(mask_acc);
                        packed_written += 1;
                        mask_acc = 0;
                        mask_bits = 0;
                    }
                }

                if buf_amask.len() >= CHUNK_TARGET {
                    if let Some(enc_a) = enc_a_opt.as_mut() {
                        enc_a.write_all(&buf_amask)?;
                    }
                    buf_amask.clear();
                }
            }
        }

        if use_prev_r {
            prev_r.copy_from_slice(&row_r);
        }
        if use_prev_g {
            prev_g.copy_from_slice(&row_g);
        }
        if use_prev_b {
            prev_b.copy_from_slice(&row_b);
        }
        if use_prev_a {
            prev_a.copy_from_slice(&row_a);
        }
    }

    // flush RGB buffers
    if !buf_r.is_empty() {
        enc_r.write_all(&buf_r)?;
    }
    if !buf_g.is_empty() {
        enc_g.write_all(&buf_g)?;
    }
    if !buf_b.is_empty() {
        enc_b.write_all(&buf_b)?;
    }

    // flush alpha buffers
    if store_alpha_stream {
        if alpha_mode == AlphaMode::Normal {
            if !buf_a.is_empty() {
                if let Some(enc_a) = enc_a_opt.as_mut() {
                    enc_a.write_all(&buf_a)?;
                }
            }
        } else if alpha_mode == AlphaMode::Mask1Bit {
            // flush remaining partial byte
            if mask_bits != 0 {
                buf_amask.push(mask_acc);
                packed_written += 1;
            }
            if !buf_amask.is_empty() {
                if let Some(enc_a) = enc_a_opt.as_mut() {
                    enc_a.write_all(&buf_amask)?;
                }
            }
            if packed_written != packed_alpha_size_usize {
                return Err(format!(
                    "packed alpha size mismatch: got {}, expected {}",
                    packed_written, packed_alpha_size_usize
                )
                .into());
            }
        }
    }

    let writer_r = enc_r.finish()?;
    let writer_g = enc_g.finish()?;
    let writer_b = enc_b.finish()?;
    let writer_a = if let Some(enc_a) = enc_a_opt {
        enc_a.finish()?
    } else {
        Vec::new()
    };

    let elapsed_enc = start_enc.elapsed();
    let elapsed_total = start_total.elapsed();

    // Build channels vector in stored order: R,G,B,(A?)
    let mut channels: Vec<EncChannel> = Vec::with_capacity(stored_streams);

    channels.push(EncChannel {
        name: "R",
        orig_size: plane_size_u64,
        comp_size: writer_r.len() as u64,
        predictor: stream_predictors[0],
        data: writer_r,
    });
    channels.push(EncChannel {
        name: "G",
        orig_size: plane_size_u64,
        comp_size: writer_g.len() as u64,
        predictor: stream_predictors[1],
        data: writer_g,
    });
    channels.push(EncChannel {
        name: "B",
        orig_size: plane_size_u64,
        comp_size: writer_b.len() as u64,
        predictor: stream_predictors[2],
        data: writer_b,
    });

    if store_alpha_stream {
        let orig_a = if alpha_mode == AlphaMode::Mask1Bit {
            packed_alpha_size_u64
        } else {
            plane_size_u64
        };
        channels.push(EncChannel {
            name: "A",
            orig_size: orig_a,
            comp_size: writer_a.len() as u64,
            predictor: stream_predictors[3],
            data: writer_a,
        });
    }

    // Container assembly
    let mut total_comp_data: u64 = 0;
    for ch in &channels {
        total_comp_data = total_comp_data
            .checked_add(ch.comp_size)
            .ok_or("comp size overflow")?;
    }

    let meta_len_u32: u32 = match meta {
        Some(m) => {
            if m.len() > u32::MAX as usize {
                return Err("meta too large".into());
            }
            m.len() as u32
        }
        None => 0,
    };

    let header_size: u64 = HEADER_V1_SIZE as u64;
    let meta_overhead: u64 = if has_meta {
        4u64 + meta_len_u32 as u64
    } else {
        0
    };

    let per_stream_overhead: u64 = STREAM_HEADER_V1_SIZE as u64; // orig_size + comp_size + predictor
    let overhead = per_stream_overhead
        .checked_mul(channels.len() as u64)
        .ok_or("overhead overflow")?;

    let out_capacity = header_size
        .checked_add(meta_overhead)
        .and_then(|x| x.checked_add(overhead))
        .and_then(|x| x.checked_add(total_comp_data))
        .ok_or("out size overflow")?;

    if out_capacity > usize::MAX as u64 {
        return Err("output too large for this build".into());
    }

    let mut out = Vec::with_capacity(out_capacity as usize);
    out.extend_from_slice(&FILE_MAGIC);
    out.push(FILE_VERSION);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&dimmask.to_le_bytes());

    if has_meta {
        out.extend_from_slice(&meta_len_u32.to_le_bytes());
        if let Some(m) = meta {
            out.extend_from_slice(m);
        }
    }

    for ch in &channels {
        out.extend_from_slice(&ch.orig_size.to_le_bytes());
        out.extend_from_slice(&ch.comp_size.to_le_bytes());
        out.push(ch.predictor as u8);
        out.extend_from_slice(&ch.data);
    }

    if verbose {
        println!(
            "[vrawtex] Features: predictor={}, alpha_mode={:?}, has_meta={}, color_transform={}, chunk={} bytes, zstd_level={}, auto_select_zstd_level={}, workers_total={} (split={:?})",
            has_predictor,
            alpha_mode,
            has_meta,
            color_transform.as_str(),
            CHUNK_TARGET,
            ZSTD_LEVEL,
            AUTO_SELECT_ZSTD_LEVEL,
            ZSTD_WORKERS_TOTAL,
            ws
        );

        if has_meta {
            println!("[vrawtex] Meta: {} bytes", meta_len_u32);
        }

        println!("RAW planar size: {} bytes", raw_planar_size_u64);
        println!("Channel sizes (orig/comp):");

        for (i, ch) in channels.iter().enumerate() {
            let pct = (ch.comp_size as f64 / ch.orig_size as f64) * 100.0;
            println!(
                "  {}: {} -> {} ({:.1}%, pred={})",
                ch.name,
                ch.orig_size,
                ch.comp_size,
                pct,
                ch.predictor.as_str()
            );

            if i == 2 && !store_alpha_stream {
                let msg = match alpha_mode {
                    AlphaMode::Opaque255 => "A: ALL 255 (not stored)",
                    AlphaMode::Transparent0 => "A: ALL 0 (not stored)",
                    _ => "A: (not stored)",
                };
                println!("  {}", msg);
            }
        }

        let vrawtex_size = out.len() as u64;
        println!("Total vrawtex size: {} bytes", vrawtex_size);

        if let Some(original_size) = original_size_opt {
            if original_size > 0 {
                println!(
                    "Original size -> RAW Planar -> VRAWTEX: {} -> {} -> {}",
                    human_mb(original_size),
                    human_mb(raw_planar_size_u64),
                    human_mb(vrawtex_size)
                );
            }
        }

        let ratio = raw_planar_size_u64 as f64 / (out.len() as f64);
        println!("Compression ratio vs raw: {:.2}x smaller", ratio);

        println!(
            "Encoding time (compress): {}",
            format_duration_ns(elapsed_enc)
        );
        let secs = elapsed_enc.as_secs_f64();
        if secs > 0.0 {
            let speed_mb = raw_planar_size_u64 as f64 / secs / (1024.0 * 1024.0);
            println!("Speed: {:.1} MB/s", speed_mb);
        }
        println!(
            "Total encode time (full pipeline): {}",
            format_duration_ns(elapsed_total)
        );
    }

    Ok(out)
}

/// Удобный враппер для atlas.rs: кодировать и сразу писать файл
pub(crate) fn encode_rgba8_with_meta_to_file(
    rgba: &RgbaImage,
    meta: Option<&[u8]>,
    out_path: &Path,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();
    let bytes = encode_rgba8_with_meta_to_vec(rgba, meta, verbose, None, start_total)?;
    fs::write(out_path, &bytes)?;
    Ok(())
}

fn decode_container_to_planes(
    parsed: &ParsedContainer,
    data: &[u8],
    safety: DecodeSafety,
) -> Result<(Vec<Vec<u8>>, Vec<Predictor>, Vec<u64>, usize), Box<dyn Error>> {
    let legacy_predictors = if parsed.format == ContainerFormat::Legacy {
        legacy_stream_predictors(parsed)
    } else {
        Vec::new()
    };
    let (mut streams, comp_sizes, mut stream_predictors, end_offset) = read_streams(
        data,
        parsed.stream_offset,
        &parsed.expected_sizes,
        parsed.format,
        &legacy_predictors,
    )?;

    if safety == DecodeSafety::Strict && end_offset != data.len() {
        return Err(format!(
            "strict mode: trailing bytes after stream section ({} bytes)",
            data.len() - end_offset
        )
        .into());
    }

    if parsed.chans < 3 || streams.len() < 3 {
        return Err("invalid stream layout: expected at least RGB streams".into());
    }

    let plane_size_usize = parsed.plane_size as usize;
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(parsed.chans as usize);
    let mut logical_predictors: Vec<Predictor> = Vec::with_capacity(parsed.chans as usize);

    for _ in 0..3 {
        planes.push(streams.remove(0));
        logical_predictors.push(stream_predictors.remove(0));
    }

    if parsed.chans == 4 {
        let (a_plane, a_predictor) = if !parsed.store_alpha_stream {
            let v = match parsed.alpha_mode {
                AlphaMode::Opaque255 => 255u8,
                AlphaMode::Transparent0 => 0u8,
                _ => 255u8,
            };
            (vec![v; plane_size_usize], Predictor::None)
        } else if parsed.alpha_mode == AlphaMode::Mask1Bit {
            if streams.is_empty() {
                return Err("missing alpha mask stream".into());
            }
            let mask_predictor = stream_predictors.remove(0);
            if mask_predictor != Predictor::None {
                return Err("alpha mask stream must use predictor=none".into());
            }

            let mask = streams.remove(0);
            if mask.len() != parsed.packed_alpha_size as usize {
                return Err(format!(
                    "alpha mask size mismatch: got {}, expected {}",
                    mask.len(),
                    parsed.packed_alpha_size
                )
                .into());
            }
            let mut a = vec![0u8; plane_size_usize];
            for (i, out) in a.iter_mut().enumerate() {
                let byte = mask[i >> 3];
                let bit = (byte >> (i & 7)) & 1;
                *out = if bit != 0 { 255 } else { 0 };
            }
            (a, Predictor::None)
        } else {
            if streams.is_empty() {
                return Err("missing alpha stream".into());
            }
            (streams.remove(0), stream_predictors.remove(0))
        };
        planes.push(a_plane);
        logical_predictors.push(a_predictor);
    }

    let width_usize = parsed.width as usize;
    let height_usize = parsed.height as usize;

    decode_plane_with_predictor_inplace(
        &mut planes[0],
        width_usize,
        height_usize,
        logical_predictors[0],
    )?;
    decode_plane_with_predictor_inplace(
        &mut planes[1],
        width_usize,
        height_usize,
        logical_predictors[1],
    )?;
    decode_plane_with_predictor_inplace(
        &mut planes[2],
        width_usize,
        height_usize,
        logical_predictors[2],
    )?;
    if parsed.chans == 4 && parsed.alpha_mode == AlphaMode::Normal {
        decode_plane_with_predictor_inplace(
            &mut planes[3],
            width_usize,
            height_usize,
            logical_predictors[3],
        )?;
    }

    inverse_color_transform_rgb_planes_inplace(&mut planes, parsed.color_transform)?;

    Ok((planes, logical_predictors, comp_sizes, end_offset))
}

fn decode_cmd(
    input: PathBuf,
    output: Option<PathBuf>,
    to: DecodeFormat,
    safety: DecodeSafety,
    dump_meta: Option<PathBuf>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();
    let data = fs::read(&input)?;
    let file_size = data.len() as u64;

    let parsed = parse_container(&data, safety)?;
    let atlas_meta = parse_atlas_meta(parsed.meta_raw.as_deref());

    if verbose {
        println!(
            "[vrawtex] Decoding {} ({}x{}, {} channels, U8, format={:?}, version={})",
            input.display(),
            parsed.width,
            parsed.height,
            parsed.chans,
            parsed.format,
            parsed.version
        );
        println!(
            "[vrawtex] Features: legacy_delta={}, alpha_mode={:?}, has_meta={}, color_transform={}",
            parsed.legacy_delta,
            parsed.alpha_mode,
            parsed.has_meta,
            parsed.color_transform.as_str()
        );
    }

    let start_dec = Instant::now();
    let (planes, plane_predictors, comp_sizes, _end_offset) =
        decode_container_to_planes(&parsed, &data, safety)?;
    let elapsed_dec = start_dec.elapsed();
    let elapsed_total = start_total.elapsed();

    let base = output.unwrap_or_else(|| default_decode_base_path(&input));
    let plane_size_usize = parsed.plane_size as usize;
    let chans_usize = parsed.chans as usize;

    match to {
        DecodeFormat::Raw => {
            let mut raw_bytes: Vec<u8> = Vec::with_capacity(parsed.raw_planar_size as usize);
            for plane in planes.iter().take(chans_usize) {
                raw_bytes.extend_from_slice(plane);
            }
            let raw_path = with_ext(&base, "raw");
            fs::write(&raw_path, &raw_bytes)?;
            println!(
                "Decoded {}x{} ({} channels) -> {}",
                parsed.width,
                parsed.height,
                parsed.chans,
                raw_path.display()
            );

            if verbose {
                print_decode_stats(
                    file_size,
                    parsed.raw_planar_size,
                    &comp_sizes,
                    parsed.chans,
                    elapsed_dec,
                    elapsed_total,
                    "RAW",
                    &raw_path,
                );
                println!(
                    "[vrawtex] Predictors: R={}, G={}, B={}, A={}",
                    plane_predictors[0].as_str(),
                    plane_predictors[1].as_str(),
                    plane_predictors[2].as_str(),
                    if parsed.chans == 4 {
                        plane_predictors[3].as_str()
                    } else {
                        "n/a"
                    }
                );
            }
        }
        DecodeFormat::Png => {
            let mut interleaved = vec![0u8; plane_size_usize * chans_usize];

            if parsed.chans == 4 {
                interleave_planar_rgba(&planes, &mut interleaved);
            } else {
                for i in 0..plane_size_usize {
                    for c in 0..chans_usize {
                        interleaved[i * chans_usize + c] = planes[c][i];
                    }
                }
            }

            let png_path = with_ext(&base, "png");
            let color = match parsed.chans {
                1 => ColorType::L8,
                2 => ColorType::La8,
                3 => ColorType::Rgb8,
                4 => ColorType::Rgba8,
                _ => {
                    return Err(format!(
                        "unsupported channel count for PNG export: {}",
                        parsed.chans
                    )
                    .into());
                }
            };

            image::save_buffer(&png_path, &interleaved, parsed.width, parsed.height, color)?;
            println!(
                "Decoded {}x{} ({} channels) -> {}",
                parsed.width,
                parsed.height,
                parsed.chans,
                png_path.display()
            );

            if verbose {
                print_decode_stats(
                    file_size,
                    parsed.raw_planar_size,
                    &comp_sizes,
                    parsed.chans,
                    elapsed_dec,
                    elapsed_total,
                    "PNG",
                    &png_path,
                );
                println!(
                    "[vrawtex] Predictors: R={}, G={}, B={}, A={}",
                    plane_predictors[0].as_str(),
                    plane_predictors[1].as_str(),
                    plane_predictors[2].as_str(),
                    if parsed.chans == 4 {
                        plane_predictors[3].as_str()
                    } else {
                        "n/a"
                    }
                );
            }
        }
    }

    if let Some(meta) = atlas_meta.as_ref() {
        println!(
            "[vrawtex] Atlas meta: pad={}, entries={}",
            meta.0,
            meta.1.len()
        );
        let meta_path = dump_meta.unwrap_or_else(|| with_ext(&base, "atlas.json"));
        dump_atlas_meta(&meta_path, meta)?;
        println!("[vrawtex] Atlas meta JSON -> {}", meta_path.display());
    } else if let Some(path) = dump_meta {
        return Err(format!(
            "requested --dump-meta {}, but atlas metadata was not found",
            path.display()
        )
        .into());
    } else if parsed.has_meta {
        let meta_len = parsed.meta_raw.as_ref().map(|m| m.len()).unwrap_or(0);
        println!(
            "[vrawtex] Meta block present ({} bytes), atlas schema not recognized",
            meta_len
        );
    }

    Ok(())
}

fn render_viewport(
    planes: &[Vec<u8>],
    chans: u8,
    img_w: usize,
    img_h: usize,
    win_w: usize,
    win_h: usize,
    fb: &mut [u32],
) {
    fb.fill(0);
    if win_w == 0 || win_h == 0 || img_w == 0 || img_h == 0 {
        return;
    }

    let sx = win_w as f64 / img_w as f64;
    let sy = win_h as f64 / img_h as f64;
    let scale = sx.min(sy);

    let draw_w = ((img_w as f64 * scale).max(1.0).round() as usize).min(win_w);
    let draw_h = ((img_h as f64 * scale).max(1.0).round() as usize).min(win_h);
    let off_x = (win_w - draw_w) / 2;
    let off_y = (win_h - draw_h) / 2;

    let mut src_x_map = vec![0usize; draw_w];
    let mut src_y_map = vec![0usize; draw_h];

    for (dx, sx_out) in src_x_map.iter_mut().enumerate() {
        let src = ((dx as f64 + 0.5) / draw_w as f64 * img_w as f64).floor() as isize;
        *sx_out = src.clamp(0, img_w as isize - 1) as usize;
    }
    for (dy, sy_out) in src_y_map.iter_mut().enumerate() {
        let src = ((dy as f64 + 0.5) / draw_h as f64 * img_h as f64).floor() as isize;
        *sy_out = src.clamp(0, img_h as isize - 1) as usize;
    }

    for (dy, &src_y) in src_y_map.iter().enumerate() {
        let dst_row = (off_y + dy) * win_w + off_x;
        for (dx, &src_x) in src_x_map.iter().enumerate() {
            let src_idx = src_y * img_w + src_x;
            let sr = planes[0][src_idx] as u32;
            let sg = planes[1][src_idx] as u32;
            let sb = planes[2][src_idx] as u32;
            let sa = if chans == 4 {
                planes[3][src_idx] as u32
            } else {
                255
            };
            let out_r = (sr * sa + 127) / 255;
            let out_g = (sg * sa + 127) / 255;
            let out_b = (sb * sa + 127) / 255;
            fb[dst_row + dx] = (out_r << 16) | (out_g << 8) | out_b;
        }
    }
}

fn open_cmd(input: PathBuf, safety: DecodeSafety, verbose: bool) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();
    let data = fs::read(&input)?;
    let file_size = data.len() as u64;

    let parsed = parse_container(&data, safety)?;
    let atlas_meta = parse_atlas_meta(parsed.meta_raw.as_deref());

    if verbose {
        println!(
            "[vrawtex] Opening {} ({}x{}, {} channels, U8, format={:?}, version={})",
            input.display(),
            parsed.width,
            parsed.height,
            parsed.chans,
            parsed.format,
            parsed.version
        );
    }

    let start_dec = Instant::now();
    let (planes, plane_predictors, comp_sizes, _end_offset) =
        decode_container_to_planes(&parsed, &data, safety)?;
    let elapsed_dec = start_dec.elapsed();
    let elapsed_total = start_total.elapsed();

    if let Some(meta) = atlas_meta.as_ref() {
        println!(
            "[vrawtex] Atlas meta: pad={}, entries={}",
            meta.0,
            meta.1.len()
        );
    } else if parsed.has_meta {
        let meta_len = parsed.meta_raw.as_ref().map(|m| m.len()).unwrap_or(0);
        println!(
            "[vrawtex] Meta block present ({} bytes), atlas schema not recognized",
            meta_len
        );
    }

    let img_w = parsed.width as usize;
    let img_h = parsed.height as usize;

    let mut init_w = img_w;
    let mut init_h = img_h;
    if init_w > MAX_WINDOW_WIDTH || init_h > MAX_WINDOW_HEIGHT {
        let sx = MAX_WINDOW_WIDTH as f64 / init_w as f64;
        let sy = MAX_WINDOW_HEIGHT as f64 / init_h as f64;
        let scale = sx.min(sy);
        init_w = (init_w as f64 * scale).round() as usize;
        init_h = (init_h as f64 * scale).round() as usize;
    }

    let mut window = Window::new(
        &format!("vrawtex: {}", input.display()),
        init_w,
        init_h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )?;

    let mut fb: Vec<u32> = Vec::new();
    let mut cached_size: (usize, usize) = (0, 0);
    let mut dirty = true;

    println!(
        "Opened {}x{} ({} channels) from {} (ESC to close)",
        parsed.width,
        parsed.height,
        parsed.chans,
        input.display()
    );

    if verbose {
        print_decode_stats(
            file_size,
            parsed.raw_planar_size,
            &comp_sizes,
            parsed.chans,
            elapsed_dec,
            elapsed_total,
            "VIEW",
            &input,
        );
        println!(
            "[vrawtex] Predictors: R={}, G={}, B={}, A={}",
            plane_predictors[0].as_str(),
            plane_predictors[1].as_str(),
            plane_predictors[2].as_str(),
            if parsed.chans == 4 {
                plane_predictors[3].as_str()
            } else {
                "n/a"
            }
        );
    }

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let (win_w, win_h) = window.get_size();

        if win_w == 0 || win_h == 0 {
            window.update();
            thread::sleep(Duration::from_millis(16));
            continue;
        }

        if fb.len() != win_w * win_h {
            fb.resize(win_w * win_h, 0);
            dirty = true;
        }

        if cached_size != (win_w, win_h) {
            cached_size = (win_w, win_h);
            dirty = true;
        }

        if dirty {
            render_viewport(&planes, parsed.chans, img_w, img_h, win_w, win_h, &mut fb);
            dirty = false;
        }

        window.update_with_buffer(&fb, win_w, win_h)?;
        thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}

fn inspect_cmd(
    input: PathBuf,
    safety: DecodeSafety,
    dump_meta: Option<PathBuf>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let data = fs::read(&input)?;
    let parsed = parse_container(&data, safety)?;
    let legacy_predictors = if parsed.format == ContainerFormat::Legacy {
        legacy_stream_predictors(&parsed)
    } else {
        Vec::new()
    };

    let (headers, end_offset) = read_stream_headers(
        &data,
        parsed.stream_offset,
        &parsed.expected_sizes,
        parsed.format,
        &legacy_predictors,
    )?;

    if safety == DecodeSafety::Strict && end_offset != data.len() {
        return Err(format!(
            "strict mode: trailing bytes after stream section ({} bytes)",
            data.len() - end_offset
        )
        .into());
    }

    println!("File: {}", input.display());
    println!(
        "Format: {:?} (version={}) | Image: {}x{} | Channels: {}",
        parsed.format, parsed.version, parsed.width, parsed.height, parsed.chans
    );
    println!(
        "Header: pixfmt=0x{:04X} qval=0x{:02X}",
        parsed.pixfmt_bits, parsed.qval
    );
    println!(
        "Features: legacy_delta={} alpha_mode={:?} has_meta={} color_transform={}",
        parsed.legacy_delta,
        parsed.alpha_mode,
        parsed.has_meta,
        parsed.color_transform.as_str()
    );
    println!(
        "Sizes: file={} bytes, raw_planar={} bytes, streams={}",
        data.len(),
        parsed.raw_planar_size,
        headers.len()
    );
    println!("Streams:");
    for (i, hdr) in headers.iter().enumerate() {
        println!(
            "  {}: orig={} comp={} pred={}",
            channel_name(i, parsed.chans),
            hdr.orig_size,
            hdr.comp_size,
            hdr.predictor.as_str()
        );
    }

    let atlas_meta = parse_atlas_meta(parsed.meta_raw.as_deref());
    if let Some(meta) = atlas_meta.as_ref() {
        println!(
            "[vrawtex] Atlas meta: pad={} entries={}",
            meta.0,
            meta.1.len()
        );
        if verbose {
            for rect in atlas::meta_rects(meta).iter().take(10) {
                println!(
                    "  id={} x={} y={} w={} h={}",
                    rect.id, rect.x, rect.y, rect.w, rect.h
                );
            }
            if meta.1.len() > 10 {
                println!("  ... and {} more", meta.1.len() - 10);
            }
        }
    } else if parsed.has_meta {
        let meta_len = parsed.meta_raw.as_ref().map(|m| m.len()).unwrap_or(0);
        println!(
            "[vrawtex] Meta block present ({} bytes), atlas schema not recognized",
            meta_len
        );
    } else {
        println!("[vrawtex] Meta: none");
    }

    if let Some(path) = dump_meta {
        let meta = atlas_meta.ok_or("cannot dump meta: atlas metadata not found")?;
        dump_atlas_meta(&path, &meta)?;
        println!("[vrawtex] Atlas meta JSON -> {}", path.display());
    }

    Ok(())
}

fn print_decode_stats(
    file_size: u64,
    raw_planar_size: u64,
    comp_sizes: &[u64],
    chans: u8,
    elapsed_dec: Duration,
    elapsed_total: Duration,
    target_kind: &str,
    out_path: &Path,
) {
    println!("VRAWTEX size: {} bytes", file_size);
    println!("RAW planar size: {} bytes", raw_planar_size);
    println!("Channel sizes (orig/comp):");
    let plane_size = raw_planar_size / (chans as u64);

    for (i, &comp) in comp_sizes.iter().enumerate() {
        let pct = (comp as f64 / plane_size as f64) * 100.0;
        println!(
            "  {}: {} -> {} ({:.1}%)",
            channel_name(i, chans),
            plane_size,
            comp,
            pct
        );
    }

    println!("Decoded to {}: {}", target_kind, out_path.display());
    println!(
        "Decoding time (decompress+build): {}",
        format_duration_ns(elapsed_dec)
    );

    let secs = elapsed_dec.as_secs_f64();
    if secs > 0.0 {
        let speed_mb = raw_planar_size as f64 / secs / (1024.0 * 1024.0);
        println!("Speed: {:.1} MB/s", speed_mb);
    }

    println!(
        "Total decode time (full pipeline): {}",
        format_duration_ns(elapsed_total)
    );
}

fn default_encode_output_path(input: &Path) -> PathBuf {
    let mut out = input.to_path_buf();
    out.set_extension("vrawtex");
    out
}

fn default_decode_base_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default();
    let mut base = input.to_path_buf();
    base.set_file_name(stem);
    base
}

fn with_ext(base: &Path, ext: &str) -> PathBuf {
    let mut p = base.to_path_buf();
    p.set_extension(ext);
    p
}
