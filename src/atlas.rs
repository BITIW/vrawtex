use crate::lanczos;
use image::RgbaImage;
use rayon::prelude::*;
use serde::Serialize;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

const DEFAULT_MAX_SIDE: u32 = 16384;
const DEFAULT_PAD: u32 = 2;

// edgecost knobs (v1)
const EDGE_K: u32 = 64;         // signature length
const EDGE_THICKNESS: u32 = 2;  // average 2px band
const EDGE_RADIUS_PX: u32 = 5;  // small, fast

#[derive(Clone)]
struct TexItem {
    id: u32,
    w: u32,
    h: u32,
    rgba: RgbaImage,
    sig: lanczos::EdgeSig,
    area: u64,
}

#[derive(Clone, Copy, Debug)]
struct Placed {
    id: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

#[derive(Clone, Copy, Debug)]
struct PackResult {
    side: u32,
    packed: usize,
}

fn is_image_ext(p: &Path) -> bool {
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "tga" | "tif" | "tiff" | "gif"
    )
}

fn collect_inputs(inputs: &[PathBuf]) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut out = Vec::new();
    for p in inputs {
        if p.is_file() {
            if is_image_ext(p) {
                out.push(p.clone());
            }
            continue;
        }
        if p.is_dir() {
            for e in WalkDir::new(p).follow_links(true) {
                let e = match e {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if !e.file_type().is_file() {
                    continue;
                }
                let path = e.path().to_path_buf();
                if is_image_ext(&path) {
                    out.push(path);
                }
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}


fn pack_shelf(order: &[usize], items: &[TexItem], side: u32, pad: u32) -> (Vec<Placed>, Vec<usize>) {
    let mut placed = Vec::new();
    let mut remaining = Vec::new();

    let mut x: u32 = 0;
    let mut y: u32 = 0;
    let mut shelf_h: u32 = 0;

    let side_lim = side;

    for &idx in order {
        let it = &items[idx];
        let w = it.w + pad * 2;
        let h = it.h + pad * 2;

        if w > side_lim || h > side_lim {
            remaining.push(idx);
            continue;
        }

        if x + w > side_lim {
            // new shelf
            x = 0;
            y += shelf_h;
            shelf_h = 0;
        }

        if y + h > side_lim {
            remaining.push(idx);
            continue;
        }

        placed.push(Placed {
            id: it.id,
            x: x + pad,
            y: y + pad,
            w: it.w,
            h: it.h,
        });

        x += w;
        shelf_h = shelf_h.max(h);
    }

    (placed, remaining)
}

fn find_min_side(order: &[usize], items: &[TexItem], max_side: u32, pad: u32) -> u32 {
    let mut lo = 1u32;
    for &idx in order {
        let it = &items[idx];
        lo = lo.max(it.w + pad * 2);
        lo = lo.max(it.h + pad * 2);
    }
    let mut hi = max_side;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let (pl, rem) = pack_shelf(order, items, mid, pad);
        if rem.is_empty() && pl.len() == order.len() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn order_area(items: &[TexItem]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..items.len()).collect();
    idx.sort_by_key(|&i| {
        let it = &items[i];
        // big first
        (std::cmp::Reverse(it.area), std::cmp::Reverse(it.w.max(it.h)))
    });
    idx
}

fn order_edge_greedy_lr(items: &[TexItem]) -> Vec<usize> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 32];
    for i in 0..n {
        buckets[items[i].sig.left_q as usize].push(i);
    }

    let mut used = vec![false; n];
    let mut out = Vec::with_capacity(n);

    // seed: самый большой
    let mut cur = (0..n).max_by_key(|&i| items[i].area).unwrap();
    used[cur] = true;
    out.push(cur);

    for _ in 1..n {
        let target = items[cur].sig.right_q as i32;

        // search buckets: target, +-1, +-2, +-3
        let mut best_i: Option<usize> = None;
        let mut best_cost: u32 = u32::MAX;

        for d in 0..=3 {
            for sign in [-1, 1] {
                let q = if d == 0 { target } else { target + sign * d };
                if q < 0 || q >= 32 {
                    continue;
                }
                let b = &buckets[q as usize];


                let take = b.len().min(64);
                for &cand in b.iter().rev().take(take) {
                    if used[cand] {
                        continue;
                    }
                    let cost = lanczos::edge_mismatch_lr(&items[cur].sig, &items[cand].sig);
                    if cost < best_cost {
                        best_cost = cost;
                        best_i = Some(cand);
                        if best_cost == 0 {
                            break;
                        }
                    }
                }
                if best_cost == 0 {
                    break;
                }
            }
            if best_cost == 0 {
                break;
            }
        }

        let next = if let Some(v) = best_i {
            v
        } else {

            (0..n).find(|&i| !used[i]).unwrap()
        };

        used[next] = true;
        out.push(next);
        cur = next;
    }

    out
}

fn order_edge_greedy_bt(items: &[TexItem]) -> Vec<usize> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }

    fn top_q(sig: &lanczos::EdgeSig) -> u8 {
        let k = sig.k as usize;
        let mut s: u32 = 0;
        for i in 0..k {
            let j = i * 3;
            let r = sig.top[j];
            let g = sig.top[j + 1];
            let b = sig.top[j + 2];
            let y = (77u32 * r as u32 + 150u32 * g as u32 + 29u32 * b as u32) >> 8;
            s += y;
        }
        ((s / k as u32) as u8) >> 3
    }
    fn bottom_q(sig: &lanczos::EdgeSig) -> u8 {
        let k = sig.k as usize;
        let mut s: u32 = 0;
        for i in 0..k {
            let j = i * 3;
            let r = sig.bottom[j];
            let g = sig.bottom[j + 1];
            let b = sig.bottom[j + 2];
            let y = (77u32 * r as u32 + 150u32 * g as u32 + 29u32 * b as u32) >> 8;
            s += y;
        }
        ((s / k as u32) as u8) >> 3
    }

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 32];
    for i in 0..n {
        buckets[top_q(&items[i].sig) as usize].push(i);
    }

    let mut used = vec![false; n];
    let mut out = Vec::with_capacity(n);

    let mut cur = (0..n).max_by_key(|&i| items[i].area).unwrap();
    used[cur] = true;
    out.push(cur);

    for _ in 1..n {
        let target = bottom_q(&items[cur].sig) as i32;

        let mut best_i: Option<usize> = None;
        let mut best_cost: u32 = u32::MAX;

        for d in 0..=3 {
            for sign in [-1, 1] {
                let q = if d == 0 { target } else { target + sign * d };
                if q < 0 || q >= 32 {
                    continue;
                }
                let b = &buckets[q as usize];
                let take = b.len().min(64);

                for &cand in b.iter().rev().take(take) {
                    if used[cand] {
                        continue;
                    }
                    let cost = lanczos::edge_mismatch_bt(&items[cur].sig, &items[cand].sig);
                    if cost < best_cost {
                        best_cost = cost;
                        best_i = Some(cand);
                        if best_cost == 0 {
                            break;
                        }
                    }
                }

                if best_cost == 0 {
                    break;
                }
            }
            if best_cost == 0 {
                break;
            }
        }

        let next = if let Some(v) = best_i {
            v
        } else {
            (0..n).find(|&i| !used[i]).unwrap()
        };

        used[next] = true;
        out.push(next);
        cur = next;
    }

    out
}

fn blit_atlas(side: u32, pad: u32, placements: &[Placed], items: &[TexItem]) -> RgbaImage {
    let mut atlas = RgbaImage::from_pixel(side, side, image::Rgba([0, 0, 0, 0]));

    for p in placements {
        let it = items.iter().find(|t| t.id == p.id).expect("id");
        let src = it.rgba.as_raw();

        let src_w = it.w as usize;
        let src_h = it.h as usize;
        let dst_w = side as usize;

        for row in 0..src_h {
            let dst_off = ((p.y as usize + row) * dst_w + p.x as usize) * 4;
            let src_off = row * src_w * 4;
            let count = src_w * 4;

            // atlas mutable slice
            let buf: &mut [u8] = atlas.as_flat_samples_mut().samples;
            let (left, right) = buf.split_at_mut(dst_off);
            let dst_row = &mut right[..count];

            // src immutable
            dst_row.copy_from_slice(&src[src_off..src_off + count]);

            drop(left);
        }
        fill_padding(&mut atlas, side, pad, p);
    }

    atlas
}

fn fill_padding(atlas: &mut RgbaImage, side: u32, pad: u32, p: &Placed) {
    if pad == 0 {
        return;
    }
    let side_u = side as i32;
    let x0 = p.x as i32;
    let y0 = p.y as i32;
    let w = p.w as i32;
    let h = p.h as i32;
    let pad = pad as i32;

    // clamp helpers
    let clamp = |v: i32| v.max(0).min(side_u - 1);

    // top/bottom pad
    for dy in 1..=pad {
        let sy_top = clamp(y0);
        let sy_bot = clamp(y0 + h - 1);
        let dy_top = clamp(y0 - dy);
        let dy_bot = clamp(y0 + h - 1 + dy);

        for x in 0..w {
            let sx = clamp(x0 + x);
            let c_top = atlas.get_pixel(sx as u32, sy_top as u32).0;
            let c_bot = atlas.get_pixel(sx as u32, sy_bot as u32).0;
            atlas.put_pixel(sx as u32, dy_top as u32, image::Rgba(c_top));
            atlas.put_pixel(sx as u32, dy_bot as u32, image::Rgba(c_bot));
        }
    }

    // left/right pad
    for dx in 1..=pad {
        let sx_left = clamp(x0);
        let sx_right = clamp(x0 + w - 1);
        let dx_left = clamp(x0 - dx);
        let dx_right = clamp(x0 + w - 1 + dx);

        for y in 0..h {
            let sy = clamp(y0 + y);
            let c_l = atlas.get_pixel(sx_left as u32, sy as u32).0;
            let c_r = atlas.get_pixel(sx_right as u32, sy as u32).0;
            atlas.put_pixel(dx_left as u32, sy as u32, image::Rgba(c_l));
            atlas.put_pixel(dx_right as u32, sy as u32, image::Rgba(c_r));
        }
    }
}

// meta: (pad, vec<(id, rect_u64)>)
#[derive(Serialize)]
struct AtlasMeta(u16, Vec<(u32, u64)>);

fn pack_rect_u64(x: u16, y: u16, w: u16, h: u16) -> u64 {
    ((x as u64) << 48) | ((y as u64) << 32) | ((w as u64) << 16) | (h as u64)
}

fn make_meta(pad: u32, placements: &[Placed]) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut v = Vec::with_capacity(placements.len());
    for p in placements {
        let x = p.x as u16;
        let y = p.y as u16;
        let w = p.w as u16;
        let h = p.h as u16;
        let rect = pack_rect_u64(x, y, w, h);
        v.push((p.id, rect));
    }

    let meta = AtlasMeta(pad as u16, v);
    let bytes = rmp_serde::to_vec(&meta)?;
    Ok(bytes)
}

fn output_path_for_chunk(base: &Path, chunk: usize) -> PathBuf {
    let mut p = base.to_path_buf();
    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("vrawtex").to_string();
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("atlas").to_string();
    p.set_file_name(format!("{stem}_{chunk}.{ext}"));
    p
}

pub fn atlas_cmd(
    inputs: Vec<PathBuf>,
    output: Option<PathBuf>,
    max_side: Option<u32>,
    pad: Option<u32>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let max_side = max_side.unwrap_or(DEFAULT_MAX_SIDE);
    let pad = pad.unwrap_or(DEFAULT_PAD);
    let out_base = output.unwrap_or_else(|| PathBuf::from("./atlas.vrawtex"));

    if verbose {
        println!(
            "[vrawtex] Atlas build: inputs={}, max_side={}, pad={}, out_base={}",
            inputs.len(),
            max_side,
            pad,
            out_base.display()
        );
    }

    let paths = collect_inputs(&inputs)?;
    if paths.is_empty() {
        return Err("atlas: no images found".into());
    }

    if verbose {
        println!("[vrawtex] atlas: loading 0/{} ...", paths.len());
    }

    let t_load = Instant::now();
    let items: Vec<TexItem> = paths
        .par_iter()
        .filter_map(|p| {
            let img = image::open(p).ok()?.to_rgba8();
            let (w, h) = img.dimensions();
            let sig = lanczos::edge_signature(&img, EDGE_K, EDGE_THICKNESS, EDGE_RADIUS_PX);
            Some(TexItem {
                id: 0,
                w,
                h,
                rgba: img,
                sig,
                area: w as u64 * h as u64,
            })
        })
        .collect();

    let mut items: Vec<TexItem> = items
        .into_iter()
        .enumerate()
        .map(|(i, mut it)| {
            it.id = i as u32;
            it
        })
        .collect();

    if verbose {
        println!(
            "[vrawtex] atlas: loaded {} image(s) in {:.9} sec",
            items.len(),
            t_load.elapsed().as_secs_f64()
        );
    }

    // split loop
    let mut chunk_idx = 0usize;
    while !items.is_empty() {
        if verbose {
            println!(
                "[vrawtex] atlas: building chunk #{} (remaining={})",
                chunk_idx,
                items.len()
            );
        }

        // candidates: order variants
        let ord0 = order_area(&items);
        let ord1 = order_edge_greedy_lr(&items);
        let ord2 = order_edge_greedy_bt(&items);

        // first: pack as many as possible at max_side using ord0 (baseline split)
        let (placed0, rem0) = pack_shelf(&ord0, &items, max_side, pad);
        if placed0.is_empty() {
            return Err("atlas: cannot pack even 1 texture into max_side (too big?)".into());
        }
        // determine which indices are packed
        let mut packed_flags = vec![false; items.len()];
        for p in &placed0 {
            let idx = p.id as usize; // id == index for current chunk list, because we renumber each loop?
            // not guaranteed after removals, so do safer mapping by scan:
            // We'll map by id -> position in items:
            let _ = idx;
        }
        let mut chunk_items = Vec::with_capacity(placed0.len());
        let mut rest_items = Vec::with_capacity(rem0.len());

        // build a set of placed ids
        let mut placed_ids = std::collections::HashSet::with_capacity(placed0.len());
        for p in &placed0 {
            placed_ids.insert(p.id);
        }

        for it in items.into_iter() {
            if placed_ids.contains(&it.id) {
                chunk_items.push(it);
            } else {
                rest_items.push(it);
            }
        }

        // Now we have chunk_items to actually optimize. Rebuild candidates on chunk_items.
        // Reindex not needed; ids stay stable for meta.
        let c0 = order_area(&chunk_items);
        let c1 = order_edge_greedy_lr(&chunk_items);
        let c2 = order_edge_greedy_bt(&chunk_items);

        let side0 = find_min_side(&c0, &chunk_items, max_side, pad);

        // candidate evaluation (pack + build atlas + test encode size)
        let mut best = None::<(usize, u32, u64, Vec<Placed>)>; // (cand_id, side, size, placements)

        for (cand_id, ord) in [(0usize, c0), (1usize, c1), (2usize, c2)] {
            let t_pack = Instant::now();
            let (pl, rem) = pack_shelf(&ord, &chunk_items, side0, pad);
            if !rem.is_empty() || pl.len() != chunk_items.len() {
                // if some didn't fit, skip this candidate
                continue;
            }
            let pack_dt = t_pack.elapsed();

            if verbose {
                println!(
                    "[vrawtex] atlas: candidate #{} pack ok: side={} (in {:.9} sec)",
                    cand_id,
                    side0,
                    pack_dt.as_secs_f64()
                );
            }

            let t_blit = Instant::now();
            let atlas_img = blit_atlas(side0, pad, &pl, &chunk_items);
            let blit_dt = t_blit.elapsed();

            let t_meta = Instant::now();
            let meta = make_meta(pad, &pl)?;
            let meta_dt = t_meta.elapsed();

            // test encode size (важно: это твой же encoder, только в Vec)
            let test = crate::encode_rgba8_with_meta_to_vec(
                                                                    &atlas_img,
                                                                    Some(&meta),
                                                                    false, // verbose для теста
                                                                    None,
                                                                    std::time::Instant::now(),
                                                                    )?;
            let test_size = test.len() as u64;

            if verbose {
                println!(
                    "[vrawtex] atlas: candidate #{}: side={}, test_size={} bytes (blit {:.6} ms, meta {:.6} ms)",
                    cand_id,
                    side0,
                    test_size,
                    blit_dt.as_secs_f64() * 1000.0,
                    meta_dt.as_secs_f64() * 1000.0
                );
            }

            match best {
                None => best = Some((cand_id, side0, test_size, pl)),
                Some((_bid, _bs, bsz, _)) => {
                    if test_size < bsz {
                        best = Some((cand_id, side0, test_size, pl));
                    }
                }
            }
        }

        let (best_id, best_side, best_size, best_pl) = best.ok_or("atlas: all candidates failed to pack")?;

        // Build final atlas + meta and encode to file
        let atlas_img = blit_atlas(best_side, pad, &best_pl, &chunk_items);
        let meta = make_meta(pad, &best_pl)?;

        let out_path = if rest_items.is_empty() && chunk_idx == 0 {
            out_base.clone()
        } else {
            output_path_for_chunk(&out_base, chunk_idx)
        };

        if verbose {
            println!(
                "[vrawtex] atlas: encoding {} ({}x{}) -> {}",
                chunk_idx,
                best_side,
                best_side,
                out_path.display()
            );
        }

        crate::encode_rgba8_with_meta_to_file(&atlas_img, Some(&meta), &out_path, verbose)?;

        if verbose {
            println!(
                "[vrawtex] atlas: chosen candidate: #{} side={} total_vrawtex_bytes={}",
                best_id, best_side, best_size
            );
        }

        // next chunk
        items = rest_items;
        chunk_idx += 1;
    }

    Ok(())
}
