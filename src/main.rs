use clap::{Parser, Subcommand, ValueEnum};
use image::{ColorType, GenericImageView};
use minifb::{Key, Window, WindowOptions};
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

// Твои любимые ручки:
const ZSTD_LEVEL: i32 = 10;
const ZSTD_WORKERS: u32 = 6;
const CHUNK_TARGET: usize = 128 * 1024;

// Включаем delta всегда (можно потом вывести в CLI)
const DELTA_ENABLED: bool = true;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
enum AlphaMode {
    Normal = 0,
    Opaque255 = 1,
    Transparent0 = 2,
    Mask1Bit = 3,
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
        Command::Decode { input, output, to } => decode_cmd(input, output, to, cli.verbose),
        Command::Open { input } => open_cmd(input, cli.verbose),
    }
}

#[derive(Parser)]
#[command(name = "vrawtex", about = "vrawtex encoder/decoder/viewer (planar U8 + zstd)")]
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
    },

    /// Open .vrawtex in a window (viewer), without writing PNG/RAW
    Open {
        /// Input .vrawtex file
        input: PathBuf,
    },
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DecodeFormat {
    Raw,
    Png,
}

/// FLAGS layout:
/// 31..16 = PIXFMT (u16)
/// 15..8  = QVAL   (u8)  <-- we use this as FEATURE BYTE
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
    data: Vec<u8>,
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

/// Delta-encode a row in-place: row[i] = row[i] - row[i-1]
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

/// Delta-decode a row in-place: row[i] = row[i] + row[i-1]
fn delta_decode_row_inplace(row: &mut [u8]) {
    if row.is_empty() {
        return;
    }
    let mut prev = row[0];
    for i in 1..row.len() {
        let cur = row[i].wrapping_add(prev);
        row[i] = cur;
        prev = cur;
    }
}

fn delta_decode_plane_inplace(plane: &mut [u8], width: usize, height: usize) -> Result<(), Box<dyn Error>> {
    if width == 0 || height == 0 {
        return Ok(());
    }
    let expected = width
        .checked_mul(height)
        .ok_or("delta decode overflow")?;
    if plane.len() != expected {
        return Err(format!("delta decode plane size mismatch: got {}, expected {}", plane.len(), expected).into());
    }
    for y in 0..height {
        let row = &mut plane[y * width..(y + 1) * width];
        delta_decode_row_inplace(row);
    }
    Ok(())
}

fn detect_alpha_mode(rgba_bytes: &[u8]) -> AlphaMode {
    if rgba_bytes.len() < 4 {
        return AlphaMode::Normal;
    }
    let mut first = rgba_bytes[3];
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
            // константа “не 0/255” — оставляем как normal (можно потом расширить формат)
            AlphaMode::Normal
        }
    } else if binary {
        AlphaMode::Mask1Bit
    } else {
        AlphaMode::Normal
    }
}

fn feature_byte(delta: bool, alpha_mode: AlphaMode) -> u8 {
    let d = if delta { 1u8 } else { 0u8 };
    let am = (alpha_mode as u8) & 0x3;
    d | (am << 1)
}

fn parse_feature_byte(qval: u8) -> (bool, AlphaMode) {
    let delta = (qval & 1) != 0;
    let am = (qval >> 1) & 0x3;
    let alpha_mode = match am {
        0 => AlphaMode::Normal,
        1 => AlphaMode::Opaque255,
        2 => AlphaMode::Transparent0,
        3 => AlphaMode::Mask1Bit,
        _ => AlphaMode::Normal,
    };
    (delta, alpha_mode)
}

fn encode_cmd(input: PathBuf, output: Option<PathBuf>, recursive: bool, verbose: bool) -> Result<(), Box<dyn Error>> {
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
        println!("[vrawtex] Recursive encode of directory: {}", root.display());
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
        let ext = path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        let is_image = matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "bmp" | "tga" | "tif" | "tiff" | "gif");

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
    let rgba_bytes = rgba.into_raw();

    let width_usize = width as usize;
    let height_usize = height as usize;

    let pixels_u64 = (width as u64).checked_mul(height as u64).ok_or("width*height overflow")?;
    let plane_size_u64 = pixels_u64; // u8
    let raw_planar_size_u64 = plane_size_u64.checked_mul(4).ok_or("raw size overflow")?;

    if plane_size_u64 > usize::MAX as u64 {
        return Err("plane too large for this build (usize overflow)".into());
    }

    let alpha_mode = detect_alpha_mode(&rgba_bytes);
    let qval = feature_byte(DELTA_ENABLED, alpha_mode);

    let pixfmt_bits: u16 = 0x0001; // U8
    let chans: u8 = 4;
    let flags = build_flags(pixfmt_bits, qval, chans);
    let dimmask = build_dimmask(width, height);

    let plane_size_usize = plane_size_u64 as usize;

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
    let store_alpha_stream = !(alpha_mode == AlphaMode::Opaque255 || alpha_mode == AlphaMode::Transparent0);
    let stored_streams = if store_alpha_stream { 4usize } else { 3usize };

    // --- ZSTD streaming ---
    let mut enc_r = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc_r.multithread(ZSTD_WORKERS)?;
    enc_r.set_pledged_src_size(Some(plane_size_u64))?;

    let mut enc_g = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc_g.multithread(ZSTD_WORKERS)?;
    enc_g.set_pledged_src_size(Some(plane_size_u64))?;

    let mut enc_b = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
    enc_b.multithread(ZSTD_WORKERS)?;
    enc_b.set_pledged_src_size(Some(plane_size_u64))?;

    let mut enc_a_opt: Option<Encoder<'static, Vec<u8>>> = None;
    if store_alpha_stream {
        let pledged = if alpha_mode == AlphaMode::Mask1Bit {
            packed_alpha_size_u64
        } else {
            plane_size_u64
        };
        let mut enc_a = Encoder::new(Vec::new(), ZSTD_LEVEL)?;
        enc_a.multithread(ZSTD_WORKERS)?;
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

    let stride = width_usize * 4;

    // mask packing state
    let mut mask_acc: u8 = 0;
    let mut mask_bits: u8 = 0;
    let mut packed_written: usize = 0;

    let start_enc = Instant::now();

    for y in 0..height_usize {
        let row_rgba = &rgba_bytes[y * stride..(y + 1) * stride];

        deinterleave_rgba_row_to_planar(row_rgba, &mut row_r, &mut row_g, &mut row_b, &mut row_a);

        if DELTA_ENABLED {
            delta_encode_row_inplace(&mut row_r);
            delta_encode_row_inplace(&mut row_g);
            delta_encode_row_inplace(&mut row_b);
            if alpha_mode == AlphaMode::Normal {
                delta_encode_row_inplace(&mut row_a);
            }
        }

        buf_r.extend_from_slice(&row_r);
        buf_g.extend_from_slice(&row_g);
        buf_b.extend_from_slice(&row_b);

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
                buf_a.extend_from_slice(&row_a);
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
                mask_acc = 0;
                mask_bits = 0;
            }
            if !buf_amask.is_empty() {
                if let Some(enc_a) = enc_a_opt.as_mut() {
                    enc_a.write_all(&buf_amask)?;
                }
            }
            // sanity (best effort)
            if packed_written != packed_alpha_size_usize {
                // if mismatch, still write — but verbose can show
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

    // Build channels vector in stored order: R,G,B,(A?)
    let mut channels: Vec<EncChannel> = Vec::with_capacity(stored_streams);

    channels.push(EncChannel {
        name: "R",
        orig_size: plane_size_u64,
        comp_size: writer_r.len() as u64,
        data: writer_r,
    });
    channels.push(EncChannel {
        name: "G",
        orig_size: plane_size_u64,
        comp_size: writer_g.len() as u64,
        data: writer_g,
    });
    channels.push(EncChannel {
        name: "B",
        orig_size: plane_size_u64,
        comp_size: writer_b.len() as u64,
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
            data: writer_a,
        });
    }

    // Container assembly
    let mut total_comp_data: u64 = 0;
    for ch in &channels {
        total_comp_data = total_comp_data.checked_add(ch.comp_size).ok_or("comp size overflow")?;
    }

    let header_size: u64 = 4 + 8; // flags + dimmask
    let per_stream_overhead: u64 = 16; // orig_size + comp_size
    let overhead = per_stream_overhead
        .checked_mul(channels.len() as u64)
        .ok_or("overhead overflow")?;

    let out_capacity = header_size
        .checked_add(overhead)
        .and_then(|x| x.checked_add(total_comp_data))
        .ok_or("out size overflow")?;

    if out_capacity > usize::MAX as u64 {
        return Err("output too large for this build".into());
    }

    let mut out = Vec::with_capacity(out_capacity as usize);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&dimmask.to_le_bytes());

    for ch in &channels {
        out.extend_from_slice(&ch.orig_size.to_le_bytes());
        out.extend_from_slice(&ch.comp_size.to_le_bytes());
        out.extend_from_slice(&ch.data);
    }

    fs::write(out_path, &out)?;
    let vrawtex_size = out.len() as u64;
    let elapsed_total = start_total.elapsed();

    println!("Encoded {}x{} RGBA8 -> {}", width, height, out_path.display());

    if verbose {
        println!(
            "[vrawtex] Features: delta={}, alpha_mode={:?}, chunk={} bytes, zstd_level={}, workers={}",
            DELTA_ENABLED, alpha_mode, CHUNK_TARGET, ZSTD_LEVEL, ZSTD_WORKERS
        );

        println!("RAW planar size: {} bytes", raw_planar_size_u64);
        println!("Channel sizes (orig/comp):");

        // R/G/B always
        for (i, ch) in channels.iter().enumerate() {
            let pct = (ch.comp_size as f64 / ch.orig_size as f64) * 100.0;
            println!("  {}: {} -> {} ({:.1}%)", ch.name, ch.orig_size, ch.comp_size, pct);

            // if A omitted, we'll print separately below
            if i == 2 && !store_alpha_stream {
                let msg = match alpha_mode {
                    AlphaMode::Opaque255 => "A: ALL 255 (not stored)",
                    AlphaMode::Transparent0 => "A: ALL 0 (not stored)",
                    _ => "A: (not stored)",
                };
                println!("  {}", msg);
            }
        }

        println!("Total vrawtex size: {} bytes", vrawtex_size);
        if original_size > 0 {
            println!(
                "Original size -> RAW Planar -> VRAWTEX: {} -> {} -> {}",
                human_mb(original_size),
                human_mb(raw_planar_size_u64),
                human_mb(vrawtex_size)
            );
        }

        let ratio = raw_planar_size_u64 as f64 / vrawtex_size as f64;
        println!("Compression ratio vs raw: {:.2}x smaller", ratio);

        println!("Encoding time (compress): {}", format_duration_ns(elapsed_enc));
        let secs = elapsed_enc.as_secs_f64();
        if secs > 0.0 {
            let speed_mb = raw_planar_size_u64 as f64 / secs / (1024.0 * 1024.0);
            println!("Speed: {:.1} MB/s", speed_mb);
        }
        println!("Total encode time (full pipeline): {}", format_duration_ns(elapsed_total));
    }

    Ok(())
}

fn read_streams(
    data: &[u8],
    mut offset: usize,
    count: usize,
    expected_sizes: &[u64],
) -> Result<(Vec<Vec<u8>>, Vec<u64>, usize), Box<dyn Error>> {
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(count);
    let mut comp_sizes: Vec<u64> = Vec::with_capacity(count);

    for i in 0..count {
        if offset + 16 > data.len() {
            return Err("truncated file while reading stream header".into());
        }
        let orig_size = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let comp_size = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        offset += 16;

        let expected = expected_sizes.get(i).copied().unwrap_or(orig_size);
        if orig_size != expected {
            return Err(format!("stream orig_size mismatch: got {orig_size}, expected {expected} (stream #{i})").into());
        }

        if comp_size > (data.len() - offset) as u64 {
            return Err("truncated file: comp_size goes past EOF".into());
        }
        let cs = comp_size as usize;
        let comp_slice = &data[offset..offset + cs];
        offset += cs;

        let decompressed = bulk::decompress(comp_slice, orig_size as usize)?;
        if decompressed.len() != orig_size as usize {
            return Err(format!(
                "decompressed size mismatch: expected {}, got {}",
                orig_size,
                decompressed.len()
            )
            .into());
        }

        planes.push(decompressed);
        comp_sizes.push(comp_size);
    }

    Ok((planes, comp_sizes, offset))
}

fn decode_cmd(input: PathBuf, output: Option<PathBuf>, to: DecodeFormat, verbose: bool) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();
    let data = fs::read(&input)?;
    if data.len() < 12 {
        return Err("file too small to be vrawtex (need at least 12 bytes)".into());
    }

    let file_size = data.len() as u64;

    let flags = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let dimmask = u64::from_le_bytes(data[4..12].try_into().unwrap());
    let mut offset = 12usize;

    let (pixfmt_bits, qval, chans) = parse_flags(flags);
    if chans == 0 {
        return Err("invalid header: channels == 0".into());
    }
    if pixfmt_bits != 0x0001 {
        return Err(format!("unsupported pixel format: pixfmt=0x{pixfmt_bits:04X}").into());
    }

    let (delta, alpha_mode) = parse_feature_byte(qval);
    let (width, height) = parse_dimmask(dimmask);

    let w = width as u64;
    let h = height as u64;
    let pixels = w.checked_mul(h).ok_or("width*height overflow")?;
    let plane_size = pixels;
    let raw_planar_size = plane_size.checked_mul(chans as u64).ok_or("raw planar overflow")?;

    if plane_size > usize::MAX as u64 {
        return Err("image too large (usize overflow)".into());
    }

    if verbose {
        println!(
            "[vrawtex] Decoding {} ({}x{}, {} channels, U8)",
            input.display(),
            width,
            height,
            chans
        );
        println!(
            "[vrawtex] Features: delta={}, alpha_mode={:?}",
            delta, alpha_mode
        );
    }

    let plane_size_usize = plane_size as usize;
    let width_usize = width as usize;
    let height_usize = height as usize;

    let packed_alpha_size = (pixels + 7) / 8;

    // how many stored streams?
    let store_alpha_stream = !(alpha_mode == AlphaMode::Opaque255 || alpha_mode == AlphaMode::Transparent0);
    let stored_streams = if store_alpha_stream { 4usize } else { 3usize };

    let start_dec = Instant::now();

    // expected orig sizes per stored stream
    let mut expected: Vec<u64> = vec![plane_size; stored_streams];
    if store_alpha_stream && alpha_mode == AlphaMode::Mask1Bit {
        expected[3] = packed_alpha_size;
    }

    let (mut streams, comp_sizes, _off2) = read_streams(&data, offset, stored_streams, &expected)?;

    // Build logical planes: always output RGBA if chans==4
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(chans as usize);

    // R,G,B are always first 3
    if streams.len() < 3 {
        return Err("not enough streams".into());
    }
    planes.push(streams.remove(0));
    planes.push(streams.remove(0));
    planes.push(streams.remove(0));

    if chans == 4 {
        let a_plane = if !store_alpha_stream {
            match alpha_mode {
                AlphaMode::Opaque255 => vec![255u8; plane_size_usize],
                AlphaMode::Transparent0 => vec![0u8; plane_size_usize],
                _ => vec![255u8; plane_size_usize],
            }
        } else if alpha_mode == AlphaMode::Mask1Bit {
            // streams[0] is packed alpha
            if streams.is_empty() {
                return Err("missing alpha mask stream".into());
            }
            let mask = streams.remove(0);
            if mask.len() != packed_alpha_size as usize {
                return Err(format!("alpha mask size mismatch: got {}, expected {}", mask.len(), packed_alpha_size).into());
            }
            let mut a = vec![0u8; plane_size_usize];
            for i in 0..plane_size_usize {
                let byte = mask[i >> 3];
                let bit = (byte >> (i & 7)) & 1;
                a[i] = if bit != 0 { 255 } else { 0 };
            }
            a
        } else {
            // normal alpha stream
            if streams.is_empty() {
                return Err("missing alpha stream".into());
            }
            streams.remove(0)
        };
        planes.push(a_plane);
    }

    // delta decode if needed
    if delta {
        if chans >= 1 {
            delta_decode_plane_inplace(&mut planes[0], width_usize, height_usize)?;
        }
        if chans >= 2 {
            delta_decode_plane_inplace(&mut planes[1], width_usize, height_usize)?;
        }
        if chans >= 3 {
            delta_decode_plane_inplace(&mut planes[2], width_usize, height_usize)?;
        }
        if chans == 4 && alpha_mode == AlphaMode::Normal {
            delta_decode_plane_inplace(&mut planes[3], width_usize, height_usize)?;
        }
    }

    let elapsed_dec = start_dec.elapsed();
    let elapsed_total = start_total.elapsed();

    let base = output.unwrap_or_else(|| default_decode_base_path(&input));

    match to {
        DecodeFormat::Raw => {
            let mut raw_bytes: Vec<u8> = Vec::with_capacity(raw_planar_size as usize);
            for c in 0..(chans as usize) {
                raw_bytes.extend_from_slice(&planes[c]);
            }
            let raw_path = with_ext(&base, "raw");
            fs::write(&raw_path, &raw_bytes)?;
            println!(
                "Decoded {}x{} ({} channels) -> {}",
                width,
                height,
                chans,
                raw_path.display()
            );

            if verbose {
                print_decode_stats(file_size, raw_planar_size, &comp_sizes, chans, elapsed_dec, elapsed_total, "RAW", &raw_path);
            }
        }
        DecodeFormat::Png => {
            let pixels_count_usize = plane_size_usize;
            let chans_usize = chans as usize;
            let mut interleaved = vec![0u8; pixels_count_usize * chans_usize];

            if chans == 4 {
                interleave_planar_rgba(&planes, &mut interleaved);
            } else {
                // fallback for non-RGBA
                for i in 0..pixels_count_usize {
                    for c in 0..chans_usize {
                        interleaved[i * chans_usize + c] = planes[c][i];
                    }
                }
            }

            let png_path = with_ext(&base, "png");
            let color = match chans {
                1 => ColorType::L8,
                2 => ColorType::La8,
                3 => ColorType::Rgb8,
                4 => ColorType::Rgba8,
                _ => return Err(format!("unsupported channel count for PNG export: {chans}").into()),
            };

            image::save_buffer(&png_path, &interleaved, width, height, color)?;
            println!(
                "Decoded {}x{} ({} channels) -> {}",
                width,
                height,
                chans,
                png_path.display()
            );

            if verbose {
                print_decode_stats(file_size, raw_planar_size, &comp_sizes, chans, elapsed_dec, elapsed_total, "PNG", &png_path);
            }
        }
    }

    Ok(())
}

fn open_cmd(input: PathBuf, verbose: bool) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();
    let data = fs::read(&input)?;
    if data.len() < 12 {
        return Err("file too small to be vrawtex (need at least 12 bytes)".into());
    }

    let file_size = data.len() as u64;

    let flags = u32::from_le_bytes(data[0..4].try_into().unwrap());
    let dimmask = u64::from_le_bytes(data[4..12].try_into().unwrap());
    let mut offset = 12usize;

    let (pixfmt_bits, qval, chans) = parse_flags(flags);
    if chans == 0 {
        return Err("invalid header: channels == 0".into());
    }
    if pixfmt_bits != 0x0001 {
        return Err(format!("unsupported pixel format: pixfmt=0x{pixfmt_bits:04X}").into());
    }

    let (delta, alpha_mode) = parse_feature_byte(qval);
    let (width, height) = parse_dimmask(dimmask);

    let w = width as u64;
    let h = height as u64;
    let pixels = w.checked_mul(h).ok_or("width*height overflow")?;
    let plane_size = pixels;

    if plane_size > usize::MAX as u64 {
        return Err("image too large (usize overflow)".into());
    }

    let plane_size_usize = plane_size as usize;
    let width_usize = width as usize;
    let height_usize = height as usize;
    let raw_planar_size = plane_size.checked_mul(chans as u64).ok_or("raw planar overflow")?;

    if verbose {
        println!(
            "[vrawtex] Opening {} ({}x{}, {} channels, U8)",
            input.display(),
            width,
            height,
            chans
        );
        println!(
            "[vrawtex] Features: delta={}, alpha_mode={:?}",
            delta, alpha_mode
        );
    }

    let packed_alpha_size = (pixels + 7) / 8;
    let store_alpha_stream = !(alpha_mode == AlphaMode::Opaque255 || alpha_mode == AlphaMode::Transparent0);
    let stored_streams = if store_alpha_stream { 4usize } else { 3usize };

    // expected orig sizes per stored stream
    let mut expected: Vec<u64> = vec![plane_size; stored_streams];
    if store_alpha_stream && alpha_mode == AlphaMode::Mask1Bit {
        expected[3] = packed_alpha_size;
    }

    let start_dec = Instant::now();
    let (mut streams, comp_sizes, _off2) = read_streams(&data, offset, stored_streams, &expected)?;

    // Build RGBA planes
    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(chans as usize);
    planes.push(streams.remove(0));
    planes.push(streams.remove(0));
    planes.push(streams.remove(0));

    if chans == 4 {
        let a_plane = if !store_alpha_stream {
            match alpha_mode {
                AlphaMode::Opaque255 => vec![255u8; plane_size_usize],
                AlphaMode::Transparent0 => vec![0u8; plane_size_usize],
                _ => vec![255u8; plane_size_usize],
            }
        } else if alpha_mode == AlphaMode::Mask1Bit {
            let mask = streams.remove(0);
            let mut a = vec![0u8; plane_size_usize];
            for i in 0..plane_size_usize {
                let byte = mask[i >> 3];
                let bit = (byte >> (i & 7)) & 1;
                a[i] = if bit != 0 { 255 } else { 0 };
            }
            a
        } else {
            streams.remove(0)
        };
        planes.push(a_plane);
    }

    // delta decode if needed
    if delta {
        delta_decode_plane_inplace(&mut planes[0], width_usize, height_usize)?;
        delta_decode_plane_inplace(&mut planes[1], width_usize, height_usize)?;
        delta_decode_plane_inplace(&mut planes[2], width_usize, height_usize)?;
        if chans == 4 && alpha_mode == AlphaMode::Normal {
            delta_decode_plane_inplace(&mut planes[3], width_usize, height_usize)?;
        }
    }

    let elapsed_dec = start_dec.elapsed();
    let elapsed_total = start_total.elapsed();

    let img_w = width as usize;
    let img_h = height as usize;

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

    println!(
        "Opened {}x{} ({} channels) from {} (ESC to close)",
        width,
        height,
        chans,
        input.display()
    );

    if verbose {
        print_decode_stats(file_size, raw_planar_size, &comp_sizes, chans, elapsed_dec, elapsed_total, "VIEW", &input);
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
        }

        let sx = win_w as f64 / img_w as f64;
        let sy = win_h as f64 / img_h as f64;
        let scale = sx.min(sy);

        let draw_w = (img_w as f64 * scale).max(1.0).round() as usize;
        let draw_h = (img_h as f64 * scale).max(1.0).round() as usize;

        let off_x = (win_w.saturating_sub(draw_w)) / 2;
        let off_y = (win_h.saturating_sub(draw_h)) / 2;

        for y in 0..win_h {
            for x in 0..win_w {
                let mut out_r: u32 = 0;
                let mut out_g: u32 = 0;
                let mut out_b: u32 = 0;

                if x >= off_x && x < off_x + draw_w && y >= off_y && y < off_y + draw_h {
                    let rel_x = (x - off_x) as f64 + 0.5;
                    let rel_y = (y - off_y) as f64 + 0.5;

                    let src_x_f = rel_x / draw_w as f64 * img_w as f64;
                    let src_y_f = rel_y / draw_h as f64 * img_h as f64;

                    let mut src_x = src_x_f.floor() as isize;
                    let mut src_y = src_y_f.floor() as isize;

                    if src_x < 0 { src_x = 0; }
                    if src_y < 0 { src_y = 0; }
                    if src_x >= img_w as isize { src_x = img_w as isize - 1; }
                    if src_y >= img_h as isize { src_y = img_h as isize - 1; }

                    let src_idx = (src_y as usize) * img_w + (src_x as usize);

                    let sr = planes[0][src_idx] as u32;
                    let sg = planes[1][src_idx] as u32;
                    let sb = planes[2][src_idx] as u32;
                    let sa = if chans == 4 { planes[3][src_idx] as u32 } else { 255 };

                    // compose over black
                    out_r = (sr * sa + 127) / 255;
                    out_g = (sg * sa + 127) / 255;
                    out_b = (sb * sa + 127) / 255;
                }

                fb[y * win_w + x] = (out_r << 16) | (out_g << 8) | out_b;
            }
        }

        window.update_with_buffer(&fb, win_w, win_h)?;
        thread::sleep(Duration::from_millis(16));
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
    println!("Decoding time (decompress+build): {}", format_duration_ns(elapsed_dec));

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
