use clap::{Parser, Subcommand, ValueEnum};
use image::{DynamicImage, GenericImageView, ColorType};
use lz4::block::{self, CompressionMode};
use rayon::prelude::*;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Encode { input, output } => {
            encode_cmd(input, output, cli.verbose)?;
        }
        Command::Decode { input, output, to } => {
            decode_cmd(input, output, to, cli.verbose)?;
        }
    }

    Ok(())
}

#[derive(Parser)]
#[command(name = "vrawtex", about = "vrawtex encoder/decoder (RAW+LZ4HC, per-channel)")]
struct Cli {
    /// Verbose stats
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode any image file into .vrawtex
    Encode {
        /// Input image (jpg/png/etc)
        input: PathBuf,

        /// Output .vrawtex (optional, defaults to input with .vrawtex)
        output: Option<PathBuf>,
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

struct EncChannel {
    orig_size: u64,
    comp_size: u64,
    data: Vec<u8>,
}

fn encode_cmd(
    input: PathBuf,
    output: Option<PathBuf>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let start_total = Instant::now();

    // Размер исходного файла (PNG/JPG/whatever)
    let original_size = fs::metadata(&input)
        .map(|m| m.len())
        .unwrap_or(0);

    // 1) читаем картинку через image crate
    let img = image::open(&input)?;
    let (width, height) = img.dimensions();

    if verbose {
        println!(
            "[vrawtex] Encoding {} ({}x{}, RGBA8)",
            input.display(),
            width,
            height
        );
    }

    // Всё приводим к RGBA8 (4 канала, 8 бит)
    let rgba: DynamicImage = img.to_rgba8().into();
    let rgba_bytes = rgba.into_bytes(); // Vec<u8>, interleaved RGBA

    let width_u32 = width;
    let height_u32 = height;

    let pixels_count = (width_u32 as u64)
        .checked_mul(height_u32 as u64)
        .ok_or("width * height overflow")?;

    let chans: u8 = 4;
    let chans_u64 = chans as u64;
    let bytes_per_sample: u64 = 1; // u8

    let plane_size = pixels_count
        .checked_mul(bytes_per_sample)
        .ok_or("plane_size overflow")?;
    let raw_planar_size = plane_size
        .checked_mul(chans_u64)
        .ok_or("raw planar size overflow")?;

    if plane_size > usize::MAX as u64 {
        return Err("plane too large for this build (usize overflow)".into());
    }

    let plane_size_usize = plane_size as usize;

    // 2) раскладываем в planar: [R..][G..][B..][A..]
    let mut planes: Vec<Vec<u8>> = vec![vec![0u8; plane_size_usize]; chans as usize];

    let mut idx = 0usize;
    for i in 0..(pixels_count as usize) {
        let r = rgba_bytes[idx];
        let g = rgba_bytes[idx + 1];
        let b = rgba_bytes[idx + 2];
        let a = rgba_bytes[idx + 3];
        idx += 4;

        planes[0][i] = r;
        planes[1][i] = g;
        planes[2][i] = b;
        planes[3][i] = a;
    }

    // 3) собираем заголовок vrawtex
    // pixfmt = 0x0001 (U8), qval = 0, chans = 4
    let pixfmt_bits: u16 = 0x0001;
    let qval: u8 = 0;
    let flags = build_flags(pixfmt_bits, qval, chans);
    let dimmask = build_dimmask(width_u32, height_u32);

    // 4) компрессим каждый канал через LZ4HC в параллель
    let orig_size = plane_size;
    let comp_mode = CompressionMode::HIGHCOMPRESSION(12);

    let start_enc = Instant::now();
    let channels_res: Result<Vec<EncChannel>, std::io::Error> = planes
        .into_par_iter()
        .map(|plane| {
            let compressed = block::compress(&plane, Some(comp_mode), false)?;
            Ok(EncChannel {
                orig_size,
                comp_size: compressed.len() as u64,
                data: compressed,
            })
        })
        .collect();

    let channels = channels_res?;
    let elapsed_enc = start_enc.elapsed();

    // 5) собираем финальный буфер
    let mut total_comp_data: u64 = 0;
    for ch in &channels {
        total_comp_data = total_comp_data
            .checked_add(ch.comp_size)
            .ok_or("compressed data size overflow")?;
    }

    let header_size: u64 = 4 + 8; // flags + dimmask
    let per_channel_overhead: u64 = 16; // orig_size u64 + comp_size u64
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

    // 6) определяем выходной путь
    let out_path = match output {
        Some(p) => p,
        None => default_encode_output_path(&input),
    };

    fs::write(&out_path, &out)?;
    let vrawtex_size = out.len() as u64;
    let elapsed_total = start_total.elapsed();

    // короткий вывод всегда
    println!(
        "Encoded {}x{} RGBA8 -> {}",
        width_u32,
        height_u32,
        out_path.display()
    );

    if verbose {
        println!(
            "RAW planar size: {} bytes",
            raw_planar_size
        );
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

        println!(
            "Total vrawtex size: {} bytes",
            vrawtex_size
        );

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

        let secs = elapsed_enc.as_secs_f64();
        if secs > 0.0 {
            let speed_mb = raw_planar_size as f64 / secs / (1024.0 * 1024.0);
            println!("Encoding time (compress): {:.2} sec", secs);
            println!("Speed: {:.1} MB/s", speed_mb);
        }

        let total_secs = elapsed_total.as_secs_f64();
        println!("Total encode time (full pipeline): {:.2} sec", total_secs);
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

    // Сейчас поддерживаем только U8 без Q-форматов
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

    // читаем и распаковываем каналы (пока последовательно)
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

        if orig_size > i32::MAX as u64 {
            return Err("orig_size exceeds lz4::block limit (i32::MAX)".into());
        }

        let comp_size_usize = comp_size as usize;
        let comp_slice = &data[offset..offset + comp_size_usize];
        offset += comp_size_usize;

        let decompressed =
            block::decompress(comp_slice, Some(orig_size as i32))?;
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

    // Определяем базовое имя для output
    let base = match output {
        Some(p) => p,
        None => default_decode_base_path(&input),
    };
    let elapsed_total = start_total.elapsed();
    match to {
        DecodeFormat::Raw => {
            // planar RAW: [chan0][chan1][chan2]...
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
            // Собираем interleaved
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

fn print_decode_stats(
    file_size: u64,
    raw_planar_size: u64,
    comp_sizes: &[u64],
    chans: u8,
    elapsed_dec: std::time::Duration,
    elapsed_total: std::time::Duration,
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

    println!(
        "Decoded to {}: {}",
        target_kind,
        out_path.display()
    );

    let secs = elapsed_dec.as_secs_f64();
    if secs > 0.0 {
        let speed_mb = raw_planar_size as f64 / secs / (1024.0 * 1024.0);
        println!("Decoding time (decompress+build): {:.2} sec", secs);
        println!("Speed: {:.1} MB/s", speed_mb);
    }

    let total_secs = elapsed_total.as_secs_f64();
    println!("Total decode time (full pipeline): {:.2} sec", total_secs);
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