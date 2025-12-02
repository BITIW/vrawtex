use clap::{Parser, Subcommand, ValueEnum};
use image::{GenericImageView, ColorType};
use zstd::{bulk, stream::Encoder};
use minifb::{Key, Window, WindowOptions};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};
use walkdir::WalkDir;

const MAX_WINDOW_WIDTH: usize = 1920;
const MAX_WINDOW_HEIGHT: usize = 1080;
const ZSTD_LEVEL: i32 = 9;
const ZSTD_WORKERS: u32 = 6;
const CHUNK_TARGET: usize = 256 * 1024;

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
        } => {
            encode_cmd(input, output, recursive, cli.verbose)?;
        }
        Command::Decode { input, output, to } => {
            decode_cmd(input, output, to, cli.verbose)?;
        }
        Command::Open { input } => {
            open_cmd(input, cli.verbose)?;
        }
    }

    Ok(())
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
        ///
        /// Example:
        ///   vrawtex decode tex.vrawtex mytex
        /// => mytex.raw or mytex.png
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
/// 15..8  = QVAL   (u8)
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

/// Format Duration down to nanoseconds: "0.000000123 sec"
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
    orig_size: u64,
    comp_size: u64,
    data: Vec<u8>,
}

/// Encode: if input is file -> single file.
/// If input is directory + recursive=true -> recurse.
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
        encode_dir(&input, verbose)?;
        Ok(())
    } else {
        let out_path = output.unwrap_or_else(|| default_encode_output_path(&input));
        encode_one(&input, &out_path, verbose)?;
        Ok(())
    }
}

/// Recursive encode:
/// <root>/something.png -> <root_parent>/VRAWTEXed/<relative>/something.vrawtex
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

        // only process known image extensions
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
            if verbose {
                println!("[vrawtex] Skipping non-image: {}", path.display());
            }
            continue;
        }

        let rel = match path.strip_prefix(&root) {
            Ok(r) => r,
            Err(e) => {
                if verbose {
                    eprintln!(
                        "[vrawtex] strip_prefix failed for {}: {e}",
                        path.display()
                    );
                }
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
            Ok(_) => {
                processed += 1;
            }
            Err(e) => {
                failed += 1;
                if verbose {
                    eprintln!(
                        "[vrawtex] Failed to encode {}: {e}",
                        path.display()
                    );
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

/// Encode one image file into one .vrawtex
/// Планар делаем стримингом по строкам + chunk-буфера → не держим весь planar в памяти, но даём zstd жирные куски.
fn encode_one(input: &Path, out_path: &Path, verbose: bool) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();

    let original_size = fs::metadata(input)
        .map(|m| m.len())
        .unwrap_or(0);

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
    let rgba_bytes = rgba.into_raw(); // width * height * 4

    let width_u32 = width;
    let height_u32 = height;

    let width_usize = width_u32 as usize;
    let height_usize = height_u32 as usize;

    let pixels_count = (width_u32 as u64)
        .checked_mul(height_u32 as u64)
        .ok_or("width * height overflow")?;

    let chans: u8 = 4;
    let chans_u64 = chans as u64;
    let bytes_per_sample: u64 = 1;

    let plane_size = pixels_count
        .checked_mul(bytes_per_sample)
        .ok_or("plane_size overflow")?;
    let raw_planar_size = plane_size
        .checked_mul(chans_u64)
        .ok_or("raw planar size overflow")?;

    if plane_size > usize::MAX as u64 {
        return Err("plane too large for this build (usize overflow)".into());
    }

    let pixfmt_bits: u16 = 0x0001; // U8
    let qval: u8 = 0;
    let flags = build_flags(pixfmt_bits, qval, chans);
    let dimmask = build_dimmask(width_u32, height_u32);

    let orig_size = plane_size;
    let pledged = orig_size;

    // --- ZSTD streaming per-channel with multithread ---
    let writer_r: Vec<u8> = Vec::new();
    let writer_g: Vec<u8> = Vec::new();
    let writer_b: Vec<u8> = Vec::new();
    let writer_a: Vec<u8> = Vec::new();

    let mut enc_r = Encoder::new(writer_r, ZSTD_LEVEL)?;
    enc_r.multithread(ZSTD_WORKERS)?;
    enc_r.set_pledged_src_size(Some(pledged))?;

    let mut enc_g = Encoder::new(writer_g, ZSTD_LEVEL)?;
    enc_g.multithread(ZSTD_WORKERS)?;
    enc_g.set_pledged_src_size(Some(pledged))?;

    let mut enc_b = Encoder::new(writer_b, ZSTD_LEVEL)?;
    enc_b.multithread(ZSTD_WORKERS)?;
    enc_b.set_pledged_src_size(Some(pledged))?;

    let mut enc_a = Encoder::new(writer_a, ZSTD_LEVEL)?;
    enc_a.multithread(ZSTD_WORKERS)?;
    enc_a.set_pledged_src_size(Some(pledged))?;

    // Буферы для накопления чанк-data по каналам
    let mut buf_r = Vec::with_capacity(CHUNK_TARGET);
    let mut buf_g = Vec::with_capacity(CHUNK_TARGET);
    let mut buf_b = Vec::with_capacity(CHUNK_TARGET);
    let mut buf_a = Vec::with_capacity(CHUNK_TARGET);

    let stride = width_usize * 4;

    let start_enc = Instant::now();

    for y in 0..height_usize {
        let row_rgba = &rgba_bytes[y * stride..(y + 1) * stride];

        // Раскидываем по каналам + копим в чанки
        for x in 0..width_usize {
            let idx = x * 4;
            buf_r.push(row_rgba[idx]);
            buf_g.push(row_rgba[idx + 1]);
            buf_b.push(row_rgba[idx + 2]);
            buf_a.push(row_rgba[idx + 3]);
        }

        // Если чанк набился достаточно — пушим в zstd и чистим
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
        if buf_a.len() >= CHUNK_TARGET {
            enc_a.write_all(&buf_a)?;
            buf_a.clear();
        }
    }

    // Дописываем хвосты
    if !buf_r.is_empty() {
        enc_r.write_all(&buf_r)?;
    }
    if !buf_g.is_empty() {
        enc_g.write_all(&buf_g)?;
    }
    if !buf_b.is_empty() {
        enc_b.write_all(&buf_b)?;
    }
    if !buf_a.is_empty() {
        enc_a.write_all(&buf_a)?;
    }

    let writer_r = enc_r.finish()?;
    let writer_g = enc_g.finish()?;
    let writer_b = enc_b.finish()?;
    let writer_a = enc_a.finish()?;

    let elapsed_enc = start_enc.elapsed();

    let channels = vec![
        EncChannel {
            orig_size,
            comp_size: writer_r.len() as u64,
            data: writer_r,
        },
        EncChannel {
            orig_size,
            comp_size: writer_g.len() as u64,
            data: writer_g,
        },
        EncChannel {
            orig_size,
            comp_size: writer_b.len() as u64,
            data: writer_b,
        },
        EncChannel {
            orig_size,
            comp_size: writer_a.len() as u64,
            data: writer_a,
        },
    ];

    let mut total_comp_data: u64 = 0;
    for ch in &channels {
        total_comp_data = total_comp_data
            .checked_add(ch.comp_size)
            .ok_or("compressed data size overflow")?;
    }

    let header_size: u64 = 4 + 8; // flags + dimmask
    let per_channel_overhead: u64 = 16; // orig_size + comp_size
    let total_overhead = per_channel_overhead
        .checked_mul(chans_u64)
        .ok_or("overhead overflow")?;
    let out_capacity = header_size
        .checked_add(total_overhead)
        .and_then(|x| x.checked_add(total_comp_data))
        .ok_or("output size overflow")?;

    if out_capacity > usize::MAX as u64 {
        return Err("vrawtex too large for this build (usize overflow)".into());
    }

    let mut out: Vec<u8> = Vec::with_capacity(out_capacity as usize);
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

    println!(
        "Encoded {}x{} RGBA8 -> {}",
        width_u32,
        height_u32,
        out_path.display()
    );

    if verbose {
        println!("RAW planar size: {} bytes", raw_planar_size);
        println!("Channel sizes (orig/comp):");

        for (i, ch) in channels.iter().enumerate() {
            let pct = (ch.comp_size as f64 / ch.orig_size as f64) * 100.0;
            println!(
                "  {}: {} -> {} ({:.1}%)",
                channel_name(i, chans),
                ch.orig_size,
                ch.comp_size,
                pct
            );
        }

        println!("Total vrawtex size: {} bytes", vrawtex_size);

        if original_size > 0 {
            println!(
                "Original size -> RAW Planar -> VRAWTEX: {} -> {} -> {}",
                human_mb(original_size),
                human_mb(raw_planar_size),
                human_mb(vrawtex_size)
            );
        } else {
            println!(
                "RAW Planar -> VRAWTEX: {} -> {}",
                human_mb(raw_planar_size),
                human_mb(vrawtex_size)
            );
        }

        let ratio = raw_planar_size as f64 / vrawtex_size as f64;
        println!("Compression ratio vs raw: {:.2}x smaller", ratio);

        println!(
            "Encoding time (compress): {}",
            format_duration_ns(elapsed_enc)
        );

        let secs = elapsed_enc.as_secs_f64();
        if secs > 0.0 {
            let speed_mb = raw_planar_size as f64 / secs / (1024.0 * 1024.0);
            println!("Speed: {:.1} MB/s", speed_mb);
        }

        println!(
            "Total encode time (full pipeline): {}",
            format_duration_ns(elapsed_total)
        );
    }

    Ok(())
}

fn decode_cmd(
    input: PathBuf,
    output: Option<PathBuf>,
    to: DecodeFormat,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
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

    if !(pixfmt_bits == 0x0001 && qval == 0) {
        return Err(format!(
            "unsupported pixel format: pixfmt=0x{pixfmt_bits:04X}, qval={qval}"
        )
        .into());
    }

    let chans_usize = chans as usize;
    let (width, height) = parse_dimmask(dimmask);

    let w = width as u64;
    let h = height as u64;
    let bytes_per_sample: u64 = 1;

    let pixels = w
        .checked_mul(h)
        .ok_or("width * height overflow")?;
    let plane_size = pixels
        .checked_mul(bytes_per_sample)
        .ok_or("plane size overflow")?;
    let raw_planar_size = plane_size
        .checked_mul(chans as u64)
        .ok_or("raw planar size overflow")?;

    if plane_size > usize::MAX as u64 || raw_planar_size > usize::MAX as u64 {
        return Err("image too large for this build (usize overflow)".into());
    }

    let plane_size_usize = plane_size as usize;

    if verbose {
        println!(
            "[vrawtex] Decoding {} ({}x{}, {} channels, U8)",
            input.display(),
            width,
            height,
            chans
        );
    }

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(chans_usize);
    let mut comp_sizes: Vec<u64> = Vec::with_capacity(chans_usize);

    let start_dec = Instant::now();

    for _c in 0..chans_usize {
        if offset + 16 > data.len() {
            return Err("truncated file while reading channel header".into());
        }

        let orig_size = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let comp_size = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        offset += 16;

        if orig_size != plane_size {
            return Err(format!(
                "channel orig_size ({orig_size}) != expected plane_size ({plane_size})"
            )
            .into());
        }

        if comp_size > (data.len() - offset) as u64 {
            return Err("truncated file: comp_size goes past EOF".into());
        }

        let comp_size_usize = comp_size as usize;
        let comp_slice = &data[offset..offset + comp_size_usize];
        offset += comp_size_usize;

        let decompressed = bulk::decompress(comp_slice, plane_size_usize)?;
        if decompressed.len() != plane_size_usize {
            return Err(format!(
                "decompressed size mismatch: expected {}, got {}",
                plane_size_usize,
                decompressed.len()
            )
            .into());
        }

        planes.push(decompressed);
        comp_sizes.push(comp_size);
    }

    if planes.len() != chans_usize {
        return Err("not enough channels decoded".into());
    }

    let elapsed_dec = start_dec.elapsed();
    let elapsed_total = start_total.elapsed();

    let base = match output {
        Some(p) => p,
        None => default_decode_base_path(&input),
    };

    match to {
        DecodeFormat::Raw => {
            let mut raw_bytes: Vec<u8> = Vec::with_capacity(raw_planar_size as usize);
            for c in 0..chans_usize {
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
                print_decode_stats(
                    file_size,
                    raw_planar_size,
                    &comp_sizes,
                    chans,
                    elapsed_dec,
                    elapsed_total,
                    "RAW",
                    &raw_path,
                );
            }
        }
        DecodeFormat::Png => {
            let pixels_count_usize = pixels as usize;
            let mut interleaved = vec![0u8; pixels_count_usize * chans_usize];

            for i in 0..pixels_count_usize {
                for c in 0..chans_usize {
                    interleaved[i * chans_usize + c] = planes[c][i];
                }
            }

            let png_path = with_ext(&base, "png");

            let color = match chans {
                1 => ColorType::L8,
                2 => ColorType::La8,
                3 => ColorType::Rgb8,
                4 => ColorType::Rgba8,
                _ => {
                    return Err(format!(
                        "unsupported channel count for PNG export: {chans}"
                    )
                    .into())
                }
            };

            image::save_buffer(
                &png_path,
                &interleaved,
                width,
                height,
                color,
            )?;

            println!(
                "Decoded {}x{} ({} channels) -> {}",
                width,
                height,
                chans,
                png_path.display()
            );

            if verbose {
                print_decode_stats(
                    file_size,
                    raw_planar_size,
                    &comp_sizes,
                    chans,
                    elapsed_dec,
                    elapsed_total,
                    "PNG",
                    &png_path,
                );
            }
        }
    }

    Ok(())
}

/// Open vrawtex in a resizable window with alpha over black background.
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

    if !(pixfmt_bits == 0x0001 && qval == 0) {
        return Err(format!(
            "unsupported pixel format: pixfmt=0x{pixfmt_bits:04X}, qval={qval}"
        )
        .into());
    }

    let chans_usize = chans as usize;
    let (width, height) = parse_dimmask(dimmask);

    let w = width as u64;
    let h = height as u64;
    let bytes_per_sample: u64 = 1;

    let pixels = w
        .checked_mul(h)
        .ok_or("width * height overflow")?;
    let plane_size = pixels
        .checked_mul(bytes_per_sample)
        .ok_or("plane size overflow")?;
    let raw_planar_size = plane_size
        .checked_mul(chans as u64)
        .ok_or("raw planar size overflow")?;

    if plane_size > usize::MAX as u64 || raw_planar_size > usize::MAX as u64 {
        return Err("image too large for this build (usize overflow)".into());
    }

    let plane_size_usize = plane_size as usize;

    if verbose {
        println!(
            "[vrawtex] Opening {} ({}x{}, {} channels, U8)",
            input.display(),
            width,
            height,
            chans
        );
    }

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(chans_usize);
    let mut comp_sizes: Vec<u64> = Vec::with_capacity(chans_usize);

    let start_dec = Instant::now();

    for _c in 0..chans_usize {
        if offset + 16 > data.len() {
            return Err("truncated file while reading channel header".into());
        }

        let orig_size = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        let comp_size = u64::from_le_bytes(data[offset + 8..offset + 16].try_into().unwrap());
        offset += 16;

        if orig_size != plane_size {
            return Err(format!(
                "channel orig_size ({orig_size}) != expected plane_size ({plane_size})"
            )
            .into());
        }

        if comp_size > (data.len() - offset) as u64 {
            return Err("truncated file: comp_size goes past EOF".into());
        }

        let comp_size_usize = comp_size as usize;
        let comp_slice = &data[offset..offset + comp_size_usize];
        offset += comp_size_usize;

        let decompressed = bulk::decompress(comp_slice, plane_size_usize)?;
        if decompressed.len() != plane_size_usize {
            return Err(format!(
                "decompressed size mismatch: expected {}, got {}",
                plane_size_usize,
                decompressed.len()
            )
            .into());
        }

        planes.push(decompressed);
        comp_sizes.push(comp_size);
    }

    if planes.len() != chans_usize {
        return Err("not enough channels decoded".into());
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
        print_decode_stats(
            file_size,
            raw_planar_size,
            &comp_sizes,
            chans,
            elapsed_dec,
            elapsed_total,
            "VIEW",
            &input,
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
                let bg_r: u32 = 0;
                let bg_g: u32 = 0;
                let bg_b: u32 = 0;

                let mut out_r = bg_r;
                let mut out_g = bg_g;
                let mut out_b = bg_b;

                if x >= off_x && x < off_x + draw_w && y >= off_y && y < off_y + draw_h {
                    let rel_x = (x - off_x) as f64 + 0.5;
                    let rel_y = (y - off_y) as f64 + 0.5;

                    let src_x_f = rel_x / draw_w as f64 * img_w as f64;
                    let src_y_f = rel_y / draw_h as f64 * img_h as f64;

                    let mut src_x = src_x_f.floor() as isize;
                    let mut src_y = src_y_f.floor() as isize;

                    if src_x < 0 {
                        src_x = 0;
                    }
                    if src_y < 0 {
                        src_y = 0;
                    }
                    if src_x >= img_w as isize {
                        src_x = img_w as isize - 1;
                    }
                    if src_y >= img_h as isize {
                        src_y = img_h as isize - 1;
                    }

                    let src_x = src_x as usize;
                    let src_y = src_y as usize;
                    let src_idx = src_y * img_w + src_x;

                    let sr = planes[0][src_idx] as u32;
                    let sg = if chans_usize > 1 { planes[1][src_idx] as u32 } else { sr };
                    let sb = if chans_usize > 2 { planes[2][src_idx] as u32 } else { sr };
                    let sa = if chans_usize > 3 { planes[3][src_idx] as u32 } else { 255 };

                    let a = sa;
                    out_r = (sr * a + 127) / 255;
                    out_g = (sg * a + 127) / 255;
                    out_b = (sb * a + 127) / 255;
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