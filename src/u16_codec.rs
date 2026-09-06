use super::*;
use rayon::prelude::*;
use std::error::Error;
use std::io::Write;
use std::time::Instant;
use zstd::{bulk, stream::Encoder};

fn detect_alpha_mode_u16(rgba: &[u16]) -> AlphaMode {
    let Some(&first) = rgba.get(3) else {
        return AlphaMode::Normal;
    };
    let mut all_same = true;
    let mut binary = true;
    for alpha in rgba.iter().skip(3).step_by(4).copied() {
        all_same &= alpha == first;
        binary &= alpha == 0 || alpha == u16::MAX;
        if !all_same && !binary {
            break;
        }
    }
    if all_same && first == u16::MAX {
        AlphaMode::Opaque255
    } else if all_same && first == 0 {
        AlphaMode::Transparent0
    } else if binary {
        AlphaMode::Mask1Bit
    } else {
        AlphaMode::Normal
    }
}

#[inline]
fn transformed_sample(
    rgba: &[u16],
    pixel: usize,
    channel: usize,
    transform: ColorTransform,
) -> u16 {
    let base = pixel * 4;
    let r = rgba[base];
    let g = rgba[base + 1];
    let b = rgba[base + 2];
    match (transform, channel) {
        (ColorTransform::None, 0..=3) => rgba[base + channel],
        (ColorTransform::SubGreen, 0) => r.wrapping_sub(g),
        (ColorTransform::SubGreen, 1) => g,
        (ColorTransform::SubGreen, 2) => b.wrapping_sub(g),
        (ColorTransform::SubGreen, 3) => rgba[base + 3],
        (ColorTransform::SubRed, 0) => r,
        (ColorTransform::SubRed, 1) => g.wrapping_sub(r),
        (ColorTransform::SubRed, 2) => b.wrapping_sub(r),
        (ColorTransform::SubRed, 3) => rgba[base + 3],
        (ColorTransform::SubBlue, 0) => r.wrapping_sub(b),
        (ColorTransform::SubBlue, 1) => g.wrapping_sub(b),
        (ColorTransform::SubBlue, 2) => b,
        (ColorTransform::SubBlue, 3) => rgba[base + 3],
        _ => 0,
    }
}

fn extract_row(
    rgba: &[u16],
    width: usize,
    y: usize,
    channel: usize,
    transform: ColorTransform,
    out: &mut [u16],
) {
    let row_start = y * width;
    for (x, sample) in out.iter_mut().enumerate() {
        *sample = transformed_sample(rgba, row_start + x, channel, transform);
    }
}

#[inline]
fn paeth_u16(left: u16, up: u16, up_left: u16) -> u16 {
    let left = left as i64;
    let up = up as i64;
    let up_left = up_left as i64;
    let p = left + up - up_left;
    let pa = (p - left).abs();
    let pb = (p - up).abs();
    let pc = (p - up_left).abs();
    if pa <= pb && pa <= pc {
        left as u16
    } else if pb <= pc {
        up as u16
    } else {
        up_left as u16
    }
}

fn predict_row(src: &[u16], prev: &[u16], predictor: Predictor, dst: &mut [u16]) {
    match predictor {
        Predictor::None => dst.copy_from_slice(src),
        Predictor::Delta => {
            if let Some(first) = src.first().copied() {
                dst[0] = first;
                for i in 1..src.len() {
                    dst[i] = src[i].wrapping_sub(src[i - 1]);
                }
            }
        }
        Predictor::Up => {
            for i in 0..src.len() {
                dst[i] = src[i].wrapping_sub(prev[i]);
            }
        }
        Predictor::Paeth => {
            if src.is_empty() {
                return;
            }
            dst[0] = src[0].wrapping_sub(prev[0]);
            for i in 1..src.len() {
                dst[i] = src[i].wrapping_sub(paeth_u16(src[i - 1], prev[i], prev[i - 1]));
            }
        }
    }
}

fn append_u16_le(samples: &[u16], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(samples.len() * 2);
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
}

fn collect_sample(
    rgba: &[u16],
    width: usize,
    height: usize,
    channel: usize,
    transform: ColorTransform,
    predictor: Predictor,
) -> Vec<u8> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let sample_limit = PREDICTOR_SAMPLE_BYTES / 2;
    let rows_to_take = ((sample_limit / width).max(1)).min(height);
    let mut sample = Vec::with_capacity(rows_to_take * width * 2);
    let mut row = vec![0u16; width];
    let mut prev = vec![0u16; width];
    let mut residual = vec![0u16; width];
    for i in 0..rows_to_take {
        let y = if rows_to_take == height {
            i
        } else {
            i * height / rows_to_take
        };
        extract_row(rgba, width, y, channel, transform, &mut row);
        if predictor.uses_prev_row() && y > 0 {
            extract_row(rgba, width, y - 1, channel, transform, &mut prev);
        } else {
            prev.fill(0);
        }
        predict_row(&row, &prev, predictor, &mut residual);
        for value in &residual {
            sample.extend_from_slice(&value.to_le_bytes());
        }
    }
    sample.truncate(PREDICTOR_SAMPLE_BYTES);
    sample
}

fn choose_channel(
    rgba: &[u16],
    width: usize,
    height: usize,
    channel: usize,
    transform: ColorTransform,
) -> Result<ChannelAutoChoice, Box<dyn Error>> {
    let evals = predictor_candidates()
        .into_par_iter()
        .map(|predictor| -> Result<PredictorEval, String> {
            let sample = collect_sample(rgba, width, height, channel, transform, predictor);
            let size = bulk::compress(&sample, AUTO_SELECT_ZSTD_LEVEL)
                .map_err(|error| error.to_string())?
                .len();
            Ok(PredictorEval { predictor, size })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| -> Box<dyn Error> { error.into() })?;
    let (chosen, chosen_size) = choose_channel_predictor_from_evals(&evals)?;
    Ok(ChannelAutoChoice {
        chosen,
        chosen_size,
        evals,
    })
}

fn choose_rgb(
    rgba: &[u16],
    width: usize,
    height: usize,
) -> Result<(ColorTransform, [Predictor; 3], Vec<RgbTransformChoice>), Box<dyn Error>> {
    let decisions = color_transform_candidates()
        .into_par_iter()
        .map(|transform| -> Result<RgbTransformChoice, String> {
            let channels = (0..3usize)
                .into_par_iter()
                .map(|channel| {
                    choose_channel(rgba, width, height, channel, transform)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let total_size = channels.iter().fold(0usize, |total, choice| {
                total.saturating_add(choice.chosen_size)
            });
            Ok(RgbTransformChoice {
                transform,
                total_size,
                channels,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| -> Box<dyn Error> { error.into() })?;

    let best_idx = decisions
        .iter()
        .enumerate()
        .min_by_key(|(_, choice)| {
            (
                choice.total_size,
                color_transform_auto_rank(choice.transform),
            )
        })
        .map(|(index, _)| index)
        .ok_or("transform auto-select produced no candidates")?;
    let none_idx = decisions
        .iter()
        .position(|choice| choice.transform == ColorTransform::None)
        .ok_or("transform auto-select missing identity transform")?;
    let chosen_idx = if decisions[best_idx].transform == ColorTransform::None {
        best_idx
    } else {
        let best = decisions[best_idx].total_size as u64;
        let none = decisions[none_idx].total_size as u64;
        if best.saturating_mul(10_000) <= none.saturating_mul(10_000 - COLOR_TRANSFORM_MIN_GAIN_BPS)
        {
            best_idx
        } else {
            none_idx
        }
    };
    let chosen = &decisions[chosen_idx];
    Ok((
        chosen.transform,
        [
            chosen.channels[0].chosen,
            chosen.channels[1].chosen,
            chosen.channels[2].chosen,
        ],
        decisions,
    ))
}

pub(crate) fn estimate_candidate_size(
    rgba: &Rgba16Image,
    pixel_format: EncodePixelFormat,
) -> Result<usize, Box<dyn Error>> {
    let (width, height) = rgba.dimensions();
    let (transform, _, choices) = choose_rgb(rgba.as_raw(), width as usize, height as usize)?;
    let mut size = choices
        .iter()
        .find(|choice| choice.transform == transform)
        .ok_or("missing U16 animation transform estimate")?
        .total_size;
    if pixel_format.has_alpha() {
        match detect_alpha_mode_u16(rgba.as_raw()) {
            AlphaMode::Normal => {
                size = size.saturating_add(
                    choose_channel(
                        rgba.as_raw(),
                        width as usize,
                        height as usize,
                        3,
                        ColorTransform::None,
                    )?
                    .chosen_size,
                );
            }
            AlphaMode::Mask1Bit => {
                size = size.saturating_add((width as usize * height as usize).div_ceil(8));
            }
            AlphaMode::Opaque255 | AlphaMode::Transparent0 => {}
        }
    }
    Ok(size)
}

fn encode_plane(
    rgba: &[u16],
    width: usize,
    height: usize,
    channel: usize,
    transform: ColorTransform,
    predictor: Predictor,
    level: i32,
    workers: u32,
) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let plane_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|samples| samples.checked_mul(2))
        .ok_or("U16 plane size overflow")?;
    let mut encoder = Encoder::new(Vec::new(), level)?;
    encoder.multithread(workers)?;
    encoder.set_pledged_src_size(Some(plane_bytes))?;
    let mut row = vec![0u16; width];
    let mut prev = vec![0u16; width];
    let mut residual = vec![0u16; width];
    let mut bytes = Vec::with_capacity(width * 2);
    for y in 0..height {
        extract_row(rgba, width, y, channel, transform, &mut row);
        predict_row(&row, &prev, predictor, &mut residual);
        append_u16_le(&residual, &mut bytes);
        encoder.write_all(&bytes)?;
        if predictor.uses_prev_row() {
            prev.copy_from_slice(&row);
        }
    }
    Ok(encoder.finish()?)
}

fn encode_mask(
    rgba: &[u16],
    pixels: usize,
    level: i32,
    workers: u32,
) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let mut packed = vec![0u8; pixels.div_ceil(8)];
    for i in 0..pixels {
        if rgba[i * 4 + 3] != 0 {
            packed[i >> 3] |= 1 << (i & 7);
        }
    }
    let mut encoder = Encoder::new(Vec::new(), level)?;
    encoder.multithread(workers)?;
    encoder.set_pledged_src_size(Some(packed.len() as u64))?;
    encoder.write_all(&packed)?;
    Ok(encoder.finish()?)
}

pub(crate) fn encode_rgba16_with_meta_to_vec(
    rgba: &Rgba16Image,
    meta: Option<&[u8]>,
    pixel_format: EncodePixelFormat,
    profile: CompressionProfile,
    verbose: bool,
    original_size: Option<u64>,
    start_total: Instant,
) -> Result<Vec<u8>, Box<dyn Error>> {
    if !pixel_format.is_16_bit() {
        return Err("U16 encoder requires RGB16 or RGBA16".into());
    }
    let (width, height) = rgba.dimensions();
    let width_usize = width as usize;
    let height_usize = height as usize;
    let pixels = width_usize
        .checked_mul(height_usize)
        .ok_or("width*height overflow")?;
    let plane_size = (pixels as u64).checked_mul(2).ok_or("plane overflow")?;
    let channels_count = pixel_format.channels();
    let raw_planar_size = plane_size
        .checked_mul(channels_count as u64)
        .ok_or("raw size overflow")?;
    let alpha_mode = if pixel_format.has_alpha() {
        detect_alpha_mode_u16(rgba.as_raw())
    } else {
        AlphaMode::Opaque255
    };
    let store_alpha = pixel_format.has_alpha()
        && !matches!(alpha_mode, AlphaMode::Opaque255 | AlphaMode::Transparent0);
    let stored_streams = 3 + usize::from(store_alpha);
    let workers_total = zstd_workers_total();
    let workers = split_workers(workers_total, stored_streams);
    let level = profile.zstd_level();

    let start_auto = Instant::now();
    let (transform, rgb_predictors, choices) =
        choose_rgb(rgba.as_raw(), width_usize, height_usize)?;
    let alpha_choice = if store_alpha && alpha_mode == AlphaMode::Normal {
        Some(choose_channel(
            rgba.as_raw(),
            width_usize,
            height_usize,
            3,
            ColorTransform::None,
        )?)
    } else {
        None
    };
    let alpha_predictor = alpha_choice
        .as_ref()
        .map(|choice| choice.chosen)
        .unwrap_or(Predictor::None);
    let auto_elapsed = start_auto.elapsed();

    if verbose {
        println!(
            "[vrawtex] Auto-select time: {}",
            format_duration_ns(auto_elapsed)
        );
        println!(
            "[vrawtex] Color transform sample: {} -> {}",
            choices
                .iter()
                .map(|choice| format!("{}={}", choice.transform.as_str(), choice.total_size))
                .collect::<Vec<_>>()
                .join(" "),
            transform.as_str()
        );
    }

    let start_encode = Instant::now();
    let configs = (0..stored_streams)
        .map(|stream| {
            let predictor = if stream < 3 {
                rgb_predictors[stream]
            } else {
                alpha_predictor
            };
            (stream, predictor)
        })
        .collect::<Vec<_>>();
    let encoded = configs
        .into_par_iter()
        .map(|(stream, predictor)| -> Result<EncChannel, String> {
            let data = if stream == 3 && alpha_mode == AlphaMode::Mask1Bit {
                encode_mask(
                    rgba.as_raw(),
                    pixels,
                    level,
                    worker_for_stream(&workers, stream),
                )
            } else {
                encode_plane(
                    rgba.as_raw(),
                    width_usize,
                    height_usize,
                    stream,
                    if stream < 3 {
                        transform
                    } else {
                        ColorTransform::None
                    },
                    predictor,
                    level,
                    worker_for_stream(&workers, stream),
                )
            }
            .map_err(|error| error.to_string())?;
            let orig_size = if stream == 3 && alpha_mode == AlphaMode::Mask1Bit {
                pixels.div_ceil(8) as u64
            } else {
                plane_size
            };
            Ok(EncChannel {
                name: ["R", "G", "B", "A"][stream],
                orig_size,
                comp_size: data.len() as u64,
                predictor,
                data,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| -> Box<dyn Error> { error.into() })?;
    let encode_elapsed = start_encode.elapsed();

    let has_predictor = encoded
        .iter()
        .any(|stream| stream.predictor != Predictor::None);
    let qval = feature_byte(has_predictor, alpha_mode, meta.is_some(), transform);
    let flags = build_flags(pixel_format.pixfmt_bits(), qval, channels_count);
    let dimmask = build_dimmask(width, height);
    let meta_len = meta.map_or(0usize, <[u8]>::len);
    let meta_len_u32 = u32::try_from(meta_len).map_err(|_| "meta too large")?;
    let total_comp = encoded
        .iter()
        .try_fold(0usize, |sum, stream| sum.checked_add(stream.data.len()))
        .ok_or("output size overflow")?;
    let mut out = Vec::with_capacity(
        HEADER_V1_SIZE
            + if meta.is_some() { 4 + meta_len } else { 0 }
            + STREAM_HEADER_V1_SIZE * encoded.len()
            + total_comp,
    );
    out.extend_from_slice(&FILE_MAGIC);
    out.push(FILE_VERSION);
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&dimmask.to_le_bytes());
    if let Some(meta) = meta {
        out.extend_from_slice(&meta_len_u32.to_le_bytes());
        out.extend_from_slice(meta);
    }
    for stream in &encoded {
        out.extend_from_slice(&stream.orig_size.to_le_bytes());
        out.extend_from_slice(&stream.comp_size.to_le_bytes());
        out.push(stream.predictor as u8);
        out.extend_from_slice(&stream.data);
    }

    if verbose {
        println!(
            "[vrawtex] Features: pixel_format={}, predictor={}, alpha_mode={:?}, has_meta={}, color_transform={}, profile={}, zstd_level={}, workers_total={} (split={:?})",
            pixel_format.as_str(),
            has_predictor,
            alpha_mode,
            meta.is_some(),
            transform.as_str(),
            profile.as_str(),
            level,
            workers_total,
            workers
        );
        println!("RAW planar size: {} bytes", raw_planar_size);
        println!("Channel sizes (orig/comp):");
        for stream in &encoded {
            println!(
                "  {}: {} -> {} ({:.1}%, pred={})",
                stream.name,
                stream.orig_size,
                stream.comp_size,
                stream.comp_size as f64 / stream.orig_size as f64 * 100.0,
                stream.predictor.as_str()
            );
        }
        if pixel_format.has_alpha() && !store_alpha {
            println!(
                "  A: {} (not stored)",
                if alpha_mode == AlphaMode::Opaque255 {
                    "ALL 65535"
                } else {
                    "ALL 0"
                }
            );
        }
        println!("Total vrawtex size: {} bytes", out.len());
        if let Some(original_size) = original_size.filter(|size| *size > 0) {
            println!(
                "Original size -> RAW Planar -> VRAWTEX: {} -> {} -> {}",
                human_mb(original_size),
                human_mb(raw_planar_size),
                human_mb(out.len() as u64)
            );
        }
        let seconds = encode_elapsed.as_secs_f64();
        println!(
            "Encoding time (compress): {}",
            format_duration_ns(encode_elapsed)
        );
        if seconds > 0.0 {
            println!(
                "Speed: {:.1} MB/s",
                raw_planar_size as f64 / seconds / (1024.0 * 1024.0)
            );
        }
        println!(
            "Total encode time (full pipeline): {}",
            format_duration_ns(start_total.elapsed())
        );
    }
    Ok(out)
}

fn bytes_to_u16(bytes: Vec<u8>) -> Result<Vec<u16>, Box<dyn Error>> {
    if bytes.len() % 2 != 0 {
        return Err("U16 stream has an odd byte length".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_predictor(plane: &mut [u16], width: usize, height: usize, predictor: Predictor) {
    match predictor {
        Predictor::None => {}
        Predictor::Delta => {
            for row in plane.chunks_exact_mut(width).take(height) {
                for i in 1..row.len() {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
            }
        }
        Predictor::Up | Predictor::Paeth => {
            let zero = vec![0u16; width];
            for y in 0..height {
                let start = y * width;
                let (head, tail) = plane.split_at_mut(start);
                let row = &mut tail[..width];
                let prev = if y == 0 {
                    zero.as_slice()
                } else {
                    &head[start - width..start]
                };
                if predictor == Predictor::Up {
                    for i in 0..width {
                        row[i] = row[i].wrapping_add(prev[i]);
                    }
                } else if width > 0 {
                    row[0] = row[0].wrapping_add(prev[0]);
                    for i in 1..width {
                        row[i] = row[i].wrapping_add(paeth_u16(row[i - 1], prev[i], prev[i - 1]));
                    }
                }
            }
        }
    }
}

fn inverse_transform(planes: &mut [Vec<u16>], transform: ColorTransform) {
    if transform == ColorTransform::None {
        return;
    }
    let (r, gb) = planes.split_at_mut(1);
    let (g, b) = gb.split_at_mut(1);
    let r = &mut r[0];
    let g = &mut g[0];
    let b = &mut b[0];
    match transform {
        ColorTransform::None => {}
        ColorTransform::SubGreen => {
            for i in 0..r.len() {
                r[i] = r[i].wrapping_add(g[i]);
                b[i] = b[i].wrapping_add(g[i]);
            }
        }
        ColorTransform::SubRed => {
            for i in 0..r.len() {
                g[i] = g[i].wrapping_add(r[i]);
                b[i] = b[i].wrapping_add(r[i]);
            }
        }
        ColorTransform::SubBlue => {
            for i in 0..r.len() {
                r[i] = r[i].wrapping_add(b[i]);
                g[i] = g[i].wrapping_add(b[i]);
            }
        }
    }
}

pub(crate) fn decode_container_to_planes_u16(
    parsed: &ParsedContainer,
    data: &[u8],
    safety: DecodeSafety,
    materialize_constant_alpha: bool,
) -> Result<(Vec<Vec<u16>>, Vec<Predictor>, Vec<u64>, usize), Box<dyn Error>> {
    if parsed.sample_bytes != 2 {
        return Err("U16 decoder requires a U16 container".into());
    }
    let (streams, comp_sizes, stream_predictors, end_offset) = read_streams(
        data,
        parsed.stream_offset,
        &parsed.expected_sizes,
        parsed.format,
        &[],
    )?;
    if safety == DecodeSafety::Strict && end_offset != data.len() {
        return Err("strict mode: trailing bytes after U16 stream section".into());
    }
    let mut planes = Vec::with_capacity(parsed.chans as usize);
    let mut predictors = Vec::with_capacity(parsed.chans as usize);
    for index in 0..3 {
        planes.push(bytes_to_u16(streams[index].clone())?);
        predictors.push(stream_predictors[index]);
    }
    if parsed.chans == 4 {
        if !parsed.store_alpha_stream {
            if materialize_constant_alpha {
                planes.push(vec![
                    if parsed.alpha_mode == AlphaMode::Transparent0 {
                        0
                    } else {
                        u16::MAX
                    };
                    parsed.pixels as usize
                ]);
            }
            predictors.push(Predictor::None);
        } else if parsed.alpha_mode == AlphaMode::Mask1Bit {
            let mask = &streams[3];
            let mut alpha = vec![0u16; parsed.pixels as usize];
            for (index, sample) in alpha.iter_mut().enumerate() {
                *sample = if (mask[index >> 3] >> (index & 7)) & 1 != 0 {
                    u16::MAX
                } else {
                    0
                };
            }
            planes.push(alpha);
            predictors.push(Predictor::None);
        } else {
            planes.push(bytes_to_u16(streams[3].clone())?);
            predictors.push(stream_predictors[3]);
        }
    }
    planes
        .par_iter_mut()
        .zip(predictors.par_iter())
        .for_each(|(plane, predictor)| {
            decode_predictor(
                plane,
                parsed.width as usize,
                parsed.height as usize,
                *predictor,
            );
        });
    inverse_transform(&mut planes, parsed.color_transform);
    Ok((planes, predictors, comp_sizes, end_offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paeth_u16_roundtrips() {
        let width = 5;
        let height = 3;
        let original = vec![
            0, 1, 255, 256, 65535, 4, 300, 1024, 65534, 12, 9, 301, 4096, 50000, 20,
        ];
        let mut encoded = vec![0u16; original.len()];
        let zero = vec![0u16; width];
        for y in 0..height {
            let start = y * width;
            let prev = if y == 0 {
                zero.as_slice()
            } else {
                &original[start - width..start]
            };
            predict_row(
                &original[start..start + width],
                prev,
                Predictor::Paeth,
                &mut encoded[start..start + width],
            );
        }
        decode_predictor(&mut encoded, width, height, Predictor::Paeth);
        assert_eq!(encoded, original);
    }

    #[test]
    fn rgba16_container_preserves_all_sample_bits() {
        let samples = vec![
            0x0000, 0x00ff, 0x0100, 0x1234, 0xffff, 0x8000, 0x7fff, 0xabcd, 0x0001, 0xff00, 0x4242,
            0xeeee, 0x1357, 0x2468, 0x369c, 0x5555, 0xaaaa, 0x1111, 0x9999, 0x0000, 0xdead, 0xbeef,
            0xcafe, 0xffff,
        ];
        let image = Rgba16Image::from_raw(3, 2, samples.clone()).unwrap();
        let encoded = encode_rgba16_with_meta_to_vec(
            &image,
            None,
            EncodePixelFormat::Rgba16,
            CompressionProfile::Fast,
            false,
            None,
            Instant::now(),
        )
        .unwrap();
        let parsed = parse_container(&encoded, DecodeSafety::Strict).unwrap();
        assert_eq!(parsed.pixfmt_bits, 0x0002);
        assert_eq!(parsed.sample_bytes, 2);
        let (planes, _, _, _) =
            decode_container_to_planes_u16(&parsed, &encoded, DecodeSafety::Strict, true).unwrap();
        for index in 0..6 {
            for channel in 0..4 {
                assert_eq!(planes[channel][index], samples[index * 4 + channel]);
            }
        }
    }

    #[test]
    fn rgb16_discards_alpha_only() {
        let image =
            Rgba16Image::from_raw(2, 1, vec![1, 2, 3, 4, 65535, 32768, 256, 12345]).unwrap();
        let encoded = encode_rgba16_with_meta_to_vec(
            &image,
            None,
            EncodePixelFormat::Rgb16,
            CompressionProfile::Balance,
            false,
            None,
            Instant::now(),
        )
        .unwrap();
        let parsed = parse_container(&encoded, DecodeSafety::Strict).unwrap();
        let (planes, _, _, _) =
            decode_container_to_planes_u16(&parsed, &encoded, DecodeSafety::Strict, true).unwrap();
        assert_eq!(parsed.chans, 3);
        assert_eq!(planes, vec![vec![1, 65535], vec![2, 32768], vec![3, 256]]);
    }
}
