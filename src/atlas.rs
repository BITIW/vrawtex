use crate::image_input;
use crate::lanczos;
use image::RgbaImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

const DEFAULT_MAX_SIDE: u32 = 16384;
pub const MAX_ATLAS_SIDE: u32 = 22_000;
const DEFAULT_PAD: u32 = 2;

// edgecost knobs (v1)
const EDGE_K: u32 = 64; // signature length
const EDGE_THICKNESS: u32 = 2; // average 2px band
const EDGE_RADIUS_PX: u32 = 5; // small, fast
const MINECRAFT_META_KIND: &str = "vrawtex.minecraft_atlas";
const MINECRAFT_META_VERSION: u16 = 2;
const TEXTURE_PACK_KIND: &str = "vrawtex.texture_pack";
const TEXTURE_PACK_HEADER_VERSION: u16 = 1;
const TEXTURE_PACK_CONTAINER_VERSION: u16 = 1;
const TEXTURE_PACK_MAGIC: &[u8; 8] = b"VRAWVTP\0";
const TEXTURE_PACK_PREAMBLE_LEN: usize = 8 + 2 + 8;

#[derive(Clone, Debug, Default)]
pub struct MinecraftPackOptions {
    pub name: Option<String>,
    pub description: Option<String>,
    pub icon: Option<PathBuf>,
}

#[derive(Clone)]
struct TexItem {
    id: u32,
    w: u32,
    h: u32,
    rgba: RgbaImage,
    sig: lanczos::EdgeSig,
    area: u64,
    minecraft: Option<MinecraftResource>,
}

#[derive(Clone)]
struct MinecraftResource {
    location: String,
    overlay: Option<String>,
    source_width: u32,
    source_height: u32,
    mcmeta: Option<String>,
}

struct MinecraftPack {
    root: PathBuf,
    pack_mcmeta: Option<String>,
    layers: Vec<MinecraftLayer>,
    sidecars: Vec<MinecraftAtlasSidecar>,
}

struct MinecraftLayer {
    overlay: Option<String>,
    assets: PathBuf,
}

struct TexturePackBlobData {
    format: String,
    bytes: Vec<u8>,
}

struct TexturePackAtlasData {
    width: u32,
    height: u32,
    entries: usize,
    bytes: Vec<u8>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TexturePackHeader {
    pub kind: String,
    pub version: u16,
    pub name: String,
    pub description: String,
    pub pack_mcmeta: Option<String>,
    #[serde(default)]
    pub sidecars: Vec<MinecraftAtlasSidecar>,
    pub blob_section_offset: u64,
    pub icon: Option<TexturePackBlob>,
    pub atlases: Vec<TexturePackAtlasBlob>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TexturePackBlob {
    pub offset: u64,
    pub len: u64,
    pub format: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TexturePackAtlasBlob {
    pub index: u32,
    pub offset: u64,
    pub len: u64,
    pub width: u32,
    pub height: u32,
    pub entries: u32,
}

#[derive(Clone, Copy, Debug)]
struct Placed {
    id: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

fn is_image_ext(p: &Path) -> bool {
    crate::image_input::is_supported_input_ext(p)
}

fn minecraft_pack(inputs: &[PathBuf]) -> Result<MinecraftPack, Box<dyn Error>> {
    if inputs.len() != 1 || !inputs[0].is_dir() {
        let got = inputs
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "atlas --minecraft requires exactly one resource-pack root directory (got {}: [{}])",
            inputs.len(),
            got
        )
        .into());
    }

    let root = inputs[0].clone();
    if !root.join("assets").is_dir() {
        return Err(format!(
            "atlas --minecraft: {} does not contain an assets directory",
            root.display()
        )
        .into());
    }

    let pack_mcmeta_path = root.join("pack.mcmeta");
    let pack_mcmeta = if pack_mcmeta_path.is_file() {
        Some(fs::read_to_string(&pack_mcmeta_path).map_err(|e| {
            format!(
                "atlas --minecraft: cannot read {}: {e}",
                pack_mcmeta_path.display()
            )
        })?)
    } else {
        None
    };

    let mut layers = vec![MinecraftLayer {
        overlay: None,
        assets: root.join("assets"),
    }];
    if let Some(raw) = pack_mcmeta.as_deref() {
        let json: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| format!("atlas --minecraft: invalid pack.mcmeta: {e}"))?;
        if let Some(entries) = json
            .get("overlays")
            .and_then(|value| value.get("entries"))
            .and_then(serde_json::Value::as_array)
        {
            for entry in entries {
                let Some(directory) = entry.get("directory").and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let assets = root.join(directory).join("assets");
                if assets.is_dir() {
                    layers.push(MinecraftLayer {
                        overlay: Some(directory.to_owned()),
                        assets,
                    });
                }
            }
        }
    }

    let mut sidecars = Vec::new();
    for layer in &layers {
        for entry in WalkDir::new(&layer.assets).follow_links(true) {
            let entry = match entry {
                Ok(value) => value,
                Err(_) => continue,
            };
            if !entry.file_type().is_file() {
                continue;
            }
            let Some(file_name) = entry.path().file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            let Some(png_name) = file_name.strip_suffix(".png.mcmeta") else {
                continue;
            };
            let png_path = entry.path().with_file_name(format!("{png_name}.png"));
            if png_path.is_file() {
                continue;
            }
            sidecars.push(MinecraftAtlasSidecar {
                resource: minecraft_location(&layer.assets, &png_path)?,
                overlay: layer.overlay.clone(),
                mcmeta: fs::read_to_string(entry.path()).map_err(|e| {
                    format!(
                        "atlas --minecraft: cannot read {}: {e}",
                        entry.path().display()
                    )
                })?,
            });
        }
    }
    sidecars.sort_by(|a, b| (&a.overlay, &a.resource).cmp(&(&b.overlay, &b.resource)));

    Ok(MinecraftPack {
        root,
        pack_mcmeta,
        layers,
        sidecars,
    })
}

fn pack_description(pack_mcmeta: Option<&str>) -> Option<String> {
    let raw = pack_mcmeta?;
    let json: serde_json::Value = serde_json::from_str(raw).ok()?;
    let description = json.get("pack")?.get("description")?;
    if let Some(text) = description.as_str() {
        Some(text.to_owned())
    } else {
        Some(description.to_string())
    }
}

fn default_pack_name(root: &Path) -> String {
    root.file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("Minecraft Texture Pack")
        .to_owned()
}

fn texture_pack_output_path(output: Option<PathBuf>, pack: &MinecraftPack) -> PathBuf {
    output.unwrap_or_else(|| {
        let stem = pack
            .root
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|name| !name.is_empty())
            .unwrap_or("pack");
        PathBuf::from(format!("{stem}.vtp"))
    })
}

fn blob_format(path: &Path) -> String {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .filter(|ext| matches!(ext.as_str(), "png" | "raw" | "vrawtex"))
        .unwrap_or_else(|| "raw".to_owned())
}

fn load_icon_blob(
    pack: &MinecraftPack,
    options: &MinecraftPackOptions,
) -> Result<Option<TexturePackBlobData>, Box<dyn Error>> {
    let path = if let Some(path) = options.icon.as_ref() {
        Some(path.clone())
    } else {
        let pack_icon = pack.root.join("pack.png");
        pack_icon.is_file().then_some(pack_icon)
    };
    let Some(path) = path else {
        return Ok(None);
    };
    if !path.is_file() {
        return Err(format!("atlas --minecraft --ico: {} is not a file", path.display()).into());
    }
    let bytes = fs::read(&path).map_err(|e| {
        format!(
            "atlas --minecraft --ico: cannot read {}: {e}",
            path.display()
        )
    })?;
    Ok(Some(TexturePackBlobData {
        format: blob_format(&path),
        bytes,
    }))
}

fn collect_inputs(
    inputs: &[PathBuf],
    minecraft: Option<&MinecraftPack>,
) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut out = Vec::new();
    if let Some(pack) = minecraft {
        for layer in &pack.layers {
            for e in WalkDir::new(&layer.assets).follow_links(true) {
                let e = match e {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if e.file_type().is_file()
                    && e.path()
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
                {
                    out.push(e.path().to_path_buf());
                }
            }
        }
        out.sort();
        out.dedup();
        return Ok(out);
    }

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

fn minecraft_location(assets: &Path, path: &Path) -> Result<String, Box<dyn Error>> {
    let relative = path.strip_prefix(assets).map_err(|_| {
        format!(
            "atlas --minecraft: {} is outside the pack assets directory",
            path.display()
        )
    })?;
    let mut components = relative.components();
    let namespace = components
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .ok_or_else(|| {
            format!(
                "atlas --minecraft: {} has no valid namespace",
                path.display()
            )
        })?;
    let resource_path = components
        .map(|component| {
            component
                .as_os_str()
                .to_str()
                .map(str::to_owned)
                .ok_or_else(|| {
                    format!(
                        "atlas --minecraft: {} contains a non-UTF-8 path",
                        path.display()
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?
        .join("/");
    if resource_path.is_empty() {
        return Err(format!("atlas --minecraft: {} has no resource path", path.display()).into());
    }

    Ok(format!("{namespace}:{resource_path}"))
}

fn minecraft_resource(
    pack: &MinecraftPack,
    path: &Path,
    source_width: u32,
    source_height: u32,
) -> Result<MinecraftResource, Box<dyn Error>> {
    let layer = pack
        .layers
        .iter()
        .find(|layer| path.starts_with(&layer.assets))
        .ok_or_else(|| {
            format!(
                "atlas --minecraft: {} is outside declared asset layers",
                path.display()
            )
        })?;
    let mut mcmeta_os = path.as_os_str().to_os_string();
    mcmeta_os.push(".mcmeta");
    let mcmeta_path = PathBuf::from(mcmeta_os);
    let mcmeta = if mcmeta_path.is_file() {
        Some(fs::read_to_string(&mcmeta_path).map_err(|e| {
            format!(
                "atlas --minecraft: cannot read {}: {e}",
                mcmeta_path.display()
            )
        })?)
    } else {
        None
    };

    Ok(MinecraftResource {
        location: minecraft_location(&layer.assets, path)?,
        overlay: layer.overlay.clone(),
        source_width,
        source_height,
        mcmeta,
    })
}

fn pack_shelf(
    order: &[usize],
    items: &[TexItem],
    side: u32,
    pad: u32,
) -> (Vec<Placed>, Vec<usize>) {
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
        (
            std::cmp::Reverse(it.area),
            std::cmp::Reverse(it.w.max(it.h)),
        )
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
    let by_id: HashMap<u32, &TexItem> = items.iter().map(|it| (it.id, it)).collect();

    for p in placements {
        let it = *by_id.get(&p.id).expect("id");
        let src = it.rgba.as_raw();

        let src_w = it.w as usize;
        let src_h = it.h as usize;
        let dst_w = side as usize;

        {
            let buf: &mut [u8] = atlas.as_flat_samples_mut().samples;
            for row in 0..src_h {
                let dst_off = ((p.y as usize + row) * dst_w + p.x as usize) * 4;
                let src_off = row * src_w * 4;
                let count = src_w * 4;
                buf[dst_off..dst_off + count].copy_from_slice(&src[src_off..src_off + count]);
            }
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

    let side_us = side as usize;
    let idx = |x: i32, y: i32| ((y as usize) * side_us + (x as usize)) * 4;
    let buf: &mut [u8] = atlas.as_flat_samples_mut().samples;

    // top/bottom pad
    for dy in 1..=pad {
        let sy_top = clamp(y0);
        let sy_bot = clamp(y0 + h - 1);
        let dy_top = clamp(y0 - dy);
        let dy_bot = clamp(y0 + h - 1 + dy);

        for x in 0..w {
            let sx = clamp(x0 + x);
            let st = idx(sx, sy_top);
            let sb = idx(sx, sy_bot);
            let dt = idx(sx, dy_top);
            let db = idx(sx, dy_bot);

            let c_top = [buf[st], buf[st + 1], buf[st + 2], buf[st + 3]];
            let c_bot = [buf[sb], buf[sb + 1], buf[sb + 2], buf[sb + 3]];
            buf[dt..dt + 4].copy_from_slice(&c_top);
            buf[db..db + 4].copy_from_slice(&c_bot);
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
            let sl = idx(sx_left, sy);
            let sr = idx(sx_right, sy);
            let dl = idx(dx_left, sy);
            let dr = idx(dx_right, sy);

            let c_l = [buf[sl], buf[sl + 1], buf[sl + 2], buf[sl + 3]];
            let c_r = [buf[sr], buf[sr + 1], buf[sr + 2], buf[sr + 3]];
            buf[dl..dl + 4].copy_from_slice(&c_l);
            buf[dr..dr + 4].copy_from_slice(&c_r);
        }
    }

    // corners: without these, bilinear sampling can pull transparent/black texels.
    for dy in 1..=pad {
        for dx in 1..=pad {
            for (sx, sy, dst_x, dst_y) in [
                (x0, y0, x0 - dx, y0 - dy),
                (x0 + w - 1, y0, x0 + w - 1 + dx, y0 - dy),
                (x0, y0 + h - 1, x0 - dx, y0 + h - 1 + dy),
                (x0 + w - 1, y0 + h - 1, x0 + w - 1 + dx, y0 + h - 1 + dy),
            ] {
                let src = idx(clamp(sx), clamp(sy));
                let dst = idx(clamp(dst_x), clamp(dst_y));
                let color = [buf[src], buf[src + 1], buf[src + 2], buf[src + 3]];
                buf[dst..dst + 4].copy_from_slice(&color);
            }
        }
    }
}

// meta: (pad, vec<(id, rect_u64)>)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AtlasMeta(pub u16, pub Vec<(u32, u64)>);

#[derive(Serialize, Clone, Debug)]
pub struct AtlasRect {
    pub id: u32,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinecraftAtlasMeta {
    pub kind: String,
    pub version: u16,
    pub pad: u16,
    pub pack_mcmeta: Option<String>,
    #[serde(default)]
    pub sidecars: Vec<MinecraftAtlasSidecar>,
    pub entries: Vec<MinecraftAtlasEntry>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinecraftAtlasSidecar {
    pub resource: String,
    pub overlay: Option<String>,
    pub mcmeta: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MinecraftAtlasEntry {
    pub id: u32,
    pub resource: String,
    pub overlay: Option<String>,
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
    pub source_width: u32,
    pub source_height: u32,
    pub mcmeta: Option<String>,
}

fn pack_rect_u64(x: u16, y: u16, w: u16, h: u16) -> u64 {
    ((x as u64) << 48) | ((y as u64) << 32) | ((w as u64) << 16) | (h as u64)
}

pub fn unpack_rect_u64(rect: u64) -> (u16, u16, u16, u16) {
    let x = (rect >> 48) as u16;
    let y = ((rect >> 32) & 0xFFFF) as u16;
    let w = ((rect >> 16) & 0xFFFF) as u16;
    let h = (rect & 0xFFFF) as u16;
    (x, y, w, h)
}

pub fn decode_meta(bytes: &[u8]) -> Result<AtlasMeta, Box<dyn Error>> {
    let meta: AtlasMeta = rmp_serde::from_slice(bytes)?;
    Ok(meta)
}

pub fn decode_minecraft_meta(bytes: &[u8]) -> Result<MinecraftAtlasMeta, Box<dyn Error>> {
    let meta: MinecraftAtlasMeta = rmp_serde::from_slice(bytes)?;
    if meta.kind != MINECRAFT_META_KIND {
        return Err("not vrawtex minecraft atlas metadata".into());
    }
    if meta.version != MINECRAFT_META_VERSION {
        return Err(format!(
            "unsupported minecraft atlas metadata version: {}",
            meta.version
        )
        .into());
    }
    Ok(meta)
}

pub fn is_texture_pack(data: &[u8]) -> bool {
    data.starts_with(TEXTURE_PACK_MAGIC)
}

pub fn decode_texture_pack_header(data: &[u8]) -> Result<TexturePackHeader, Box<dyn Error>> {
    if data.len() < TEXTURE_PACK_PREAMBLE_LEN {
        return Err("truncated vtp: header preamble missing".into());
    }
    if !is_texture_pack(data) {
        return Err("not a vrawtex texture pack".into());
    }

    let version = u16::from_le_bytes([data[8], data[9]]);
    if version != TEXTURE_PACK_CONTAINER_VERSION {
        return Err(format!("unsupported vtp container version: {version}").into());
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&data[10..18]);
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let header_start = TEXTURE_PACK_PREAMBLE_LEN;
    let header_end = header_start
        .checked_add(header_len)
        .ok_or("vtp header length overflow")?;
    if header_end > data.len() {
        return Err("truncated vtp: header bytes missing".into());
    }

    let header: TexturePackHeader = rmp_serde::from_slice(&data[header_start..header_end])?;
    if header.kind != TEXTURE_PACK_KIND {
        return Err("not vrawtex texture pack metadata".into());
    }
    if header.version != TEXTURE_PACK_HEADER_VERSION {
        return Err(format!(
            "unsupported texture pack header version: {}",
            header.version
        )
        .into());
    }
    if header.blob_section_offset != header_end as u64 {
        return Err("invalid vtp: blob_section_offset does not match header size".into());
    }

    if let Some(icon) = header.icon.as_ref() {
        validate_blob_range(data.len(), icon.offset, icon.len, "icon")?;
    }
    for atlas in &header.atlases {
        validate_blob_range(data.len(), atlas.offset, atlas.len, "atlas")?;
    }

    Ok(header)
}

fn validate_blob_range(
    file_len: usize,
    offset: u64,
    len: u64,
    kind: &str,
) -> Result<(), Box<dyn Error>> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| format!("invalid vtp: {kind} blob range overflow"))?;
    if end > file_len as u64 {
        return Err(format!("invalid vtp: {kind} blob range exceeds file size").into());
    }
    Ok(())
}

pub fn meta_rects(meta: &AtlasMeta) -> Vec<AtlasRect> {
    meta.1
        .iter()
        .map(|(id, rect)| {
            let (x, y, w, h) = unpack_rect_u64(*rect);
            AtlasRect {
                id: *id,
                x,
                y,
                w,
                h,
            }
        })
        .collect()
}

fn make_meta(
    pad: u32,
    placements: &[Placed],
    items: &[TexItem],
    minecraft: Option<&MinecraftPack>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let pad_u16 = u16::try_from(pad).map_err(|_| "atlas meta overflow: pad does not fit u16")?;
    if let Some(pack) = minecraft {
        let by_id: HashMap<u32, &TexItem> = items.iter().map(|item| (item.id, item)).collect();
        let mut entries = Vec::with_capacity(placements.len());
        for p in placements {
            let resource = by_id
                .get(&p.id)
                .and_then(|item| item.minecraft.as_ref())
                .ok_or("atlas --minecraft: resource metadata missing")?;
            entries.push(MinecraftAtlasEntry {
                id: p.id,
                resource: resource.location.clone(),
                overlay: resource.overlay.clone(),
                x: u16::try_from(p.x).map_err(|_| "atlas meta overflow: x does not fit u16")?,
                y: u16::try_from(p.y).map_err(|_| "atlas meta overflow: y does not fit u16")?,
                w: u16::try_from(p.w).map_err(|_| "atlas meta overflow: w does not fit u16")?,
                h: u16::try_from(p.h).map_err(|_| "atlas meta overflow: h does not fit u16")?,
                source_width: resource.source_width,
                source_height: resource.source_height,
                mcmeta: resource.mcmeta.clone(),
            });
        }
        entries.sort_by(|a, b| (&a.overlay, &a.resource).cmp(&(&b.overlay, &b.resource)));
        return Ok(rmp_serde::to_vec_named(&MinecraftAtlasMeta {
            kind: MINECRAFT_META_KIND.to_owned(),
            version: MINECRAFT_META_VERSION,
            pad: pad_u16,
            pack_mcmeta: pack.pack_mcmeta.clone(),
            sidecars: pack.sidecars.clone(),
            entries,
        })?);
    }

    let mut v = Vec::with_capacity(placements.len());
    for p in placements {
        let x = u16::try_from(p.x).map_err(|_| "atlas meta overflow: x does not fit u16")?;
        let y = u16::try_from(p.y).map_err(|_| "atlas meta overflow: y does not fit u16")?;
        let w = u16::try_from(p.w).map_err(|_| "atlas meta overflow: w does not fit u16")?;
        let h = u16::try_from(p.h).map_err(|_| "atlas meta overflow: h does not fit u16")?;
        let rect = pack_rect_u64(x, y, w, h);
        v.push((p.id, rect));
    }

    let meta = AtlasMeta(pad_u16, v);
    let bytes = rmp_serde::to_vec(&meta)?;
    Ok(bytes)
}

fn output_path_for_chunk(base: &Path, chunk: usize) -> PathBuf {
    let mut p = base.to_path_buf();
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("vrawtex")
        .to_string();
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("atlas")
        .to_string();
    p.set_file_name(format!("{stem}_{chunk}.{ext}"));
    p
}

fn output_path_for_mip(base: &Path, level: usize) -> PathBuf {
    let mut p = base.to_path_buf();
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("vrawtex")
        .to_string();
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("atlas")
        .to_string();
    p.set_file_name(format!("{stem}_mip{level}.{ext}"));
    p
}

fn write_texture_pack(
    output: &Path,
    pack: &MinecraftPack,
    options: &MinecraftPackOptions,
    atlases: Vec<TexturePackAtlasData>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    if atlases.is_empty() {
        return Err("atlas --minecraft: cannot write empty texture pack".into());
    }

    let icon_data = load_icon_blob(pack, options)?;
    let name = options
        .name
        .clone()
        .unwrap_or_else(|| default_pack_name(&pack.root));
    let description = options
        .description
        .clone()
        .or_else(|| pack_description(pack.pack_mcmeta.as_deref()))
        .unwrap_or_default();

    let mut blob_cursor = 0u64;
    let mut icon = None;
    if let Some(icon_data) = icon_data.as_ref() {
        icon = Some(TexturePackBlob {
            offset: blob_cursor,
            len: icon_data.bytes.len() as u64,
            format: icon_data.format.clone(),
        });
        blob_cursor = blob_cursor.saturating_add(icon_data.bytes.len() as u64);
    }

    let mut atlas_headers = Vec::with_capacity(atlases.len());
    for (index, atlas) in atlases.iter().enumerate() {
        atlas_headers.push(TexturePackAtlasBlob {
            index: index as u32,
            offset: blob_cursor,
            len: atlas.bytes.len() as u64,
            width: atlas.width,
            height: atlas.height,
            entries: u32::try_from(atlas.entries)
                .map_err(|_| "atlas --minecraft: too many atlas entries")?,
        });
        blob_cursor = blob_cursor.saturating_add(atlas.bytes.len() as u64);
    }

    let mut header = TexturePackHeader {
        kind: TEXTURE_PACK_KIND.to_owned(),
        version: TEXTURE_PACK_HEADER_VERSION,
        name,
        description,
        pack_mcmeta: pack.pack_mcmeta.clone(),
        sidecars: pack.sidecars.clone(),
        blob_section_offset: 0,
        icon,
        atlases: atlas_headers,
    };

    let mut header_bytes = rmp_serde::to_vec_named(&header)?;
    let mut converged = false;
    for _ in 0..8 {
        let blob_section_offset = (TEXTURE_PACK_PREAMBLE_LEN + header_bytes.len()) as u64;
        let mut absolute_cursor = blob_section_offset;
        if let Some(icon) = header.icon.as_mut() {
            icon.offset = absolute_cursor;
            absolute_cursor = absolute_cursor.saturating_add(icon.len);
        }
        for atlas in &mut header.atlases {
            atlas.offset = absolute_cursor;
            absolute_cursor = absolute_cursor.saturating_add(atlas.len);
        }
        header.blob_section_offset = blob_section_offset;

        let next = rmp_serde::to_vec_named(&header)?;
        if next.len() == header_bytes.len() {
            header_bytes = next;
            converged = true;
            break;
        }
        header_bytes = next;
    }
    if !converged {
        return Err("atlas --minecraft: failed to stabilize vtp header layout".into());
    }

    let mut out = Vec::with_capacity(
        TEXTURE_PACK_PREAMBLE_LEN
            .saturating_add(header_bytes.len())
            .saturating_add(blob_cursor as usize),
    );
    out.extend_from_slice(TEXTURE_PACK_MAGIC);
    out.extend_from_slice(&TEXTURE_PACK_CONTAINER_VERSION.to_le_bytes());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    if let Some(icon_data) = icon_data.as_ref() {
        out.extend_from_slice(&icon_data.bytes);
    }
    for atlas in &atlases {
        out.extend_from_slice(&atlas.bytes);
    }

    fs::write(output, &out)?;

    if verbose {
        println!(
            "[vrawtex] VTP: wrote {} atlas blob(s), icon={}, header={} bytes, total={} bytes -> {}",
            atlases.len(),
            header.icon.is_some(),
            header_bytes.len(),
            out.len(),
            output.display()
        );
    }

    Ok(())
}

fn scale_dimension(value: u32, source_side: u32, target_side: u32) -> u32 {
    ((value as u64 * target_side as u64 + source_side as u64 / 2) / source_side as u64)
        .max(1)
        .min(value as u64) as u32
}

fn scale_padding(pad: u32, source_side: u32, target_side: u32) -> u32 {
    (pad as u64 * target_side as u64 / source_side as u64) as u32
}

pub fn atlas_cmd(
    inputs: Vec<PathBuf>,
    output: Option<PathBuf>,
    max_side: Option<u32>,
    pad: Option<u32>,
    mipchain: Option<crate::mipchain::MipChainSpec>,
    pixel_format: crate::EncodePixelFormat,
    profile: crate::CompressionProfile,
    minecraft_options: Option<MinecraftPackOptions>,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let max_side = max_side.unwrap_or(DEFAULT_MAX_SIDE);
    let pad = pad.unwrap_or(DEFAULT_PAD);
    let with_mipchain = mipchain.is_some();
    let minecraft = minecraft_options.is_some();
    let minecraft_pack = minecraft.then(|| minecraft_pack(&inputs)).transpose()?;
    let out_base = if let Some(pack) = minecraft_pack.as_ref() {
        texture_pack_output_path(output, pack)
    } else {
        output.unwrap_or_else(|| PathBuf::from("./atlas.vrawtex"))
    };

    if max_side > MAX_ATLAS_SIDE {
        return Err(format!(
            "atlas: max-side {} exceeds supported limit {}",
            max_side, MAX_ATLAS_SIDE
        )
        .into());
    }

    if verbose {
        println!(
            "[vrawtex] Atlas build: inputs={}, max_side={}, pad={}, mipchain={}, minecraft={}, profile={}, zstd_level={}, out_base={}",
            inputs.len(),
            max_side,
            pad,
            with_mipchain,
            minecraft,
            profile.as_str(),
            profile.zstd_level(),
            out_base.display()
        );
    }

    let paths = collect_inputs(&inputs, minecraft_pack.as_ref())?;
    if paths.is_empty() {
        return Err("atlas: no images found".into());
    }

    if verbose {
        println!("[vrawtex] atlas: loading 0/{} ...", paths.len());
    }

    let t_load = Instant::now();
    let load_results: Vec<Result<TexItem, String>> = paths
        .par_iter()
        .map(|p| {
            let img = image_input::load_rgba8(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let (w, h) = img.dimensions();
            let sig = lanczos::edge_signature(&img, EDGE_K, EDGE_THICKNESS, EDGE_RADIUS_PX);
            let minecraft = minecraft_pack
                .as_ref()
                .map(|pack| minecraft_resource(pack, p, w, h))
                .transpose()
                .map_err(|e| e.to_string())?;
            Ok(TexItem {
                id: 0,
                w,
                h,
                rgba: img,
                sig,
                area: w as u64 * h as u64,
                minecraft,
            })
        })
        .collect();

    let mut load_errors: Vec<String> = Vec::new();
    let items: Vec<TexItem> = load_results
        .into_iter()
        .filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                load_errors.push(e);
                None
            }
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

    if !load_errors.is_empty() {
        eprintln!(
            "[vrawtex] atlas: skipped {} image(s) due to load errors",
            load_errors.len()
        );
        if verbose {
            for msg in load_errors.iter().take(20) {
                eprintln!("  {msg}");
            }
            if load_errors.len() > 20 {
                eprintln!("  ... and {} more", load_errors.len() - 20);
            }
        }
    }

    if items.is_empty() {
        return Err("atlas: failed to load any input image".into());
    }

    if verbose {
        println!(
            "[vrawtex] atlas: loaded {} image(s) in {:.9} sec",
            items.len(),
            t_load.elapsed().as_secs_f64()
        );
    }

    let mut texture_pack_atlases = Vec::new();

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

        // first: pack as many as possible at max_side using ord0 (baseline split)
        let (placed0, rem0) = pack_shelf(&ord0, &items, max_side, pad);
        if placed0.is_empty() {
            return Err("atlas: cannot pack even 1 texture into max_side (too big?)".into());
        }
        let mut chunk_items = Vec::with_capacity(placed0.len());
        let mut rest_items = Vec::with_capacity(rem0.len());

        // build a set of placed ids
        let mut placed_ids = HashSet::with_capacity(placed0.len());
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
        let mut candidates = Vec::with_capacity(3);
        for (candidate_id, order) in [(0usize, c0), (1usize, c1), (2usize, c2)] {
            if !candidates
                .iter()
                .any(|(_existing_id, existing_order)| existing_order == &order)
            {
                candidates.push((candidate_id, order));
            }
        }

        // candidate evaluation (pack + build atlas + test encode size)
        let mut best = None::<(usize, u32, u64, Vec<u8>)>;

        for (cand_id, ord) in candidates {
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
            let meta = make_meta(pad, &pl, &chunk_items, minecraft_pack.as_ref())?;
            let meta_dt = t_meta.elapsed();

            // test encode size (важно: это твой же encoder, только в Vec)
            let test = crate::encode_rgba8_with_meta_to_vec(
                &atlas_img,
                Some(&meta),
                pixel_format,
                profile,
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

            let replace_best = match best.as_ref() {
                None => true,
                Some((_bid, _bs, best_size, _best_bytes)) => test_size < *best_size,
            };
            if replace_best {
                best = Some((cand_id, side0, test_size, test));
            }
        }

        let (best_id, best_side, best_size, best_encoded) =
            best.ok_or("atlas: all candidates failed to pack")?;

        let chunk_out_path = if rest_items.is_empty() && chunk_idx == 0 {
            out_base.clone()
        } else {
            output_path_for_chunk(&out_base, chunk_idx)
        };

        if with_mipchain {
            let spec = mipchain.as_ref().expect("mipchain spec exists");
            if !spec.has_explicit_heights()
                && let Some(requested) = spec.additional_levels()
            {
                let mut available = 0usize;
                let mut max_dimension = chunk_items
                    .iter()
                    .map(|item| item.w.max(item.h))
                    .max()
                    .unwrap_or(1);
                while max_dimension > 1 {
                    max_dimension = (max_dimension / 2).max(1);
                    available += 1;
                }
                if requested > available {
                    return Err(format!(
                        "atlas: requested {requested} mip level(s), but this chunk only supports {available} distinct level(s)"
                    )
                    .into());
                }
            }

            let mip0_path = output_path_for_mip(&chunk_out_path, 0);
            if verbose {
                println!(
                    "[vrawtex] atlas: writing chosen chunk {} mip0 ({}x{}) -> {}",
                    chunk_idx,
                    best_side,
                    best_side,
                    mip0_path.display()
                );
            }
            std::fs::write(&mip0_path, &best_encoded)?;

            let explicit_sides = if spec.has_explicit_heights() {
                Some(spec.target_heights(best_side)?)
            } else {
                None
            };
            let mut mip_items = chunk_items;
            let mut level_idx = 1usize;
            let mut previous_side = best_side;
            while explicit_sides
                .as_ref()
                .is_some_and(|sides| level_idx <= sides.len())
                || (explicit_sides.is_none()
                    && mip_items.iter().any(|item| item.w > 1 || item.h > 1)
                    && spec
                        .additional_levels()
                        .is_none_or(|limit| level_idx <= limit))
            {
                let t_mip = Instant::now();
                let target_side = explicit_sides.as_ref().map(|sides| sides[level_idx - 1]);
                let (taps, ops) = mip_items
                    .par_iter_mut()
                    .map(|item| {
                        if item.w == 1 && item.h == 1 {
                            return (0u64, 0u64);
                        }

                        let (dst_w, dst_h) = if let Some(target_side) = target_side {
                            (
                                scale_dimension(item.w, previous_side, target_side),
                                scale_dimension(item.h, previous_side, target_side),
                            )
                        } else {
                            ((item.w / 2).max(1), (item.h / 2).max(1))
                        };
                        let (resized, stats) =
                            lanczos::resize_lanczos_rgba_percent(&item.rgba, dst_w, dst_h, 100);
                        item.rgba = resized;
                        item.w = dst_w;
                        item.h = dst_h;
                        item.area = dst_w as u64 * dst_h as u64;
                        (stats.taps_total, stats.ops_total)
                    })
                    .reduce(
                        || (0u64, 0u64),
                        |a, b| (a.0.saturating_add(b.0), a.1.saturating_add(b.1)),
                    );

                let order = order_area(&mip_items);
                let level_pad = target_side
                    .map(|side| scale_padding(pad, best_side, side))
                    .unwrap_or(pad);
                let side = target_side
                    .unwrap_or_else(|| find_min_side(&order, &mip_items, max_side, level_pad));
                let (placements, remaining) = pack_shelf(&order, &mip_items, side, level_pad);
                if !remaining.is_empty() || placements.len() != mip_items.len() {
                    return Err(format!(
                        "atlas: failed to pack mip{level_idx} into requested {}x{} atlas",
                        side, side
                    )
                    .into());
                }

                let level = blit_atlas(side, level_pad, &placements, &mip_items);
                let meta = make_meta(level_pad, &placements, &mip_items, minecraft_pack.as_ref())?;
                let out_path = output_path_for_mip(&chunk_out_path, level_idx);
                if verbose {
                    println!(
                        "[vrawtex] atlas: encoding chunk {} mip{} ({}x{}) -> {} (lanczos_radius=100%, taps={}, ops={}, build={:.9} sec)",
                        chunk_idx,
                        level_idx,
                        level.width(),
                        level.height(),
                        out_path.display(),
                        taps,
                        ops,
                        t_mip.elapsed().as_secs_f64()
                    );
                }
                crate::encode_rgba8_with_meta_to_file(
                    &level,
                    Some(&meta),
                    &out_path,
                    pixel_format,
                    profile,
                    verbose,
                )?;
                previous_side = side;
                level_idx += 1;
            }
        } else if minecraft {
            if verbose {
                println!(
                    "[vrawtex] atlas: staging chosen chunk {} ({}x{}, entries={}) into VTP",
                    chunk_idx,
                    best_side,
                    best_side,
                    chunk_items.len()
                );
            }
            texture_pack_atlases.push(TexturePackAtlasData {
                width: best_side,
                height: best_side,
                entries: chunk_items.len(),
                bytes: best_encoded,
            });
        } else {
            if verbose {
                println!(
                    "[vrawtex] atlas: writing chosen candidate {} ({}x{}) -> {}",
                    chunk_idx,
                    best_side,
                    best_side,
                    chunk_out_path.display()
                );
            }
            std::fs::write(&chunk_out_path, &best_encoded)?;
        }

        if verbose {
            println!(
                "[vrawtex] atlas: chosen candidate: #{} side={} mip0_vrawtex_bytes={}",
                best_id, best_side, best_size
            );
        }

        // next chunk
        items = rest_items;
        chunk_idx += 1;
    }

    if let (Some(pack), Some(options)) = (minecraft_pack.as_ref(), minecraft_options.as_ref()) {
        write_texture_pack(&out_base, pack, options, texture_pack_atlases, verbose)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn atlas_mipchain_writes_rgb8_levels_with_atlas_meta() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vrawtex-atlas-test-{unique}"));
        fs::create_dir_all(&root).unwrap();

        let a_path = root.join("a.png");
        let b_path = root.join("b.png");
        RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 255]))
            .save(&a_path)
            .unwrap();
        RgbaImage::from_pixel(2, 3, Rgba([0, 255, 0, 128]))
            .save(&b_path)
            .unwrap();

        let out = root.join("atlas.vrawtex");
        atlas_cmd(
            vec![a_path, b_path],
            Some(out.clone()),
            Some(64),
            Some(1),
            crate::mipchain::MipChainSpec::from_cli(Some(0), Vec::new()).unwrap(),
            crate::EncodePixelFormat::Rgb8,
            crate::CompressionProfile::Balance,
            None,
            false,
        )
        .unwrap();

        for level in 0..=2 {
            let path = output_path_for_mip(&out, level);
            let bytes = fs::read(path).unwrap();
            let parsed = crate::parse_container(&bytes, crate::DecodeSafety::Strict).unwrap();
            assert_eq!(parsed.chans, 3);
            assert!(decode_meta(parsed.meta_raw.as_deref().unwrap()).is_ok());
        }
        assert!(!output_path_for_mip(&out, 3).exists());
        assert!(!out.exists());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atlas_explicit_mip_sizes_are_exact() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vrawtex-atlas-size-test-{unique}"));
        fs::create_dir_all(&root).unwrap();

        let input = root.join("input.png");
        RgbaImage::from_pixel(32, 16, Rgba([10, 20, 30, 255]))
            .save(&input)
            .unwrap();
        let out = root.join("atlas.vrawtex");
        let spec = crate::mipchain::MipChainSpec::from_cli(Some(3), vec![30, 15, 8]).unwrap();

        atlas_cmd(
            vec![input],
            Some(out.clone()),
            Some(64),
            Some(1),
            spec,
            crate::EncodePixelFormat::Rgba8,
            crate::CompressionProfile::Balance,
            None,
            false,
        )
        .unwrap();

        for (level, expected_side) in [(1usize, 30u32), (2, 15), (3, 8)] {
            let bytes = fs::read(output_path_for_mip(&out, level)).unwrap();
            let parsed = crate::parse_container(&bytes, crate::DecodeSafety::Strict).unwrap();
            assert_eq!(
                (parsed.width, parsed.height),
                (expected_side, expected_side)
            );
        }

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn minecraft_atlas_preserves_resource_locations_and_mcmeta() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vrawtex-minecraft-atlas-test-{unique}"));
        let texture_dir = root.join("assets/example/textures/block");
        fs::create_dir_all(&texture_dir).unwrap();
        fs::write(
            root.join("pack.mcmeta"),
            r#"{"pack":{"description":"test","pack_format":1},"overlays":{"entries":[{"directory":"patch","formats":[1,99]}]}}"#,
        )
        .unwrap();
        RgbaImage::from_pixel(2, 2, Rgba([200, 100, 50, 255]))
            .save(root.join("pack.png"))
            .unwrap();

        let texture = texture_dir.join("animated.png");
        RgbaImage::from_pixel(4, 8, Rgba([10, 20, 30, 255]))
            .save(&texture)
            .unwrap();
        fs::write(
            texture_dir.join("animated.png.mcmeta"),
            r#"{"animation":{"frametime":2}}"#,
        )
        .unwrap();
        fs::write(
            texture_dir.join("fallback.png.mcmeta"),
            r#"{"texture":{"blur":true}}"#,
        )
        .unwrap();
        let overlay_texture_dir = root.join("patch/assets/example/textures/block");
        fs::create_dir_all(&overlay_texture_dir).unwrap();
        RgbaImage::from_pixel(8, 8, Rgba([30, 20, 10, 255]))
            .save(overlay_texture_dir.join("animated.png"))
            .unwrap();

        let out = root.join("pack.vtp");
        atlas_cmd(
            vec![root.clone()],
            Some(out.clone()),
            Some(64),
            Some(1),
            None,
            crate::EncodePixelFormat::Rgba8,
            crate::CompressionProfile::Balance,
            Some(MinecraftPackOptions {
                name: Some("Fixture Pack".to_owned()),
                description: Some("Tiny fixture".to_owned()),
                icon: None,
            }),
            false,
        )
        .unwrap();

        let bytes = fs::read(out).unwrap();
        let header = decode_texture_pack_header(&bytes).unwrap();
        assert_eq!(header.kind, TEXTURE_PACK_KIND);
        assert_eq!(header.name, "Fixture Pack");
        assert_eq!(header.description, "Tiny fixture");
        assert!(header.icon.is_some());
        assert_eq!(header.sidecars.len(), 1);
        assert_eq!(header.atlases.len(), 1);
        assert!(header.atlases[0].offset > header.blob_section_offset);

        let atlas = &header.atlases[0];
        let atlas_start = atlas.offset as usize;
        let atlas_end = atlas_start + atlas.len as usize;
        let parsed =
            crate::parse_container(&bytes[atlas_start..atlas_end], crate::DecodeSafety::Strict)
                .unwrap();
        let meta = decode_minecraft_meta(parsed.meta_raw.as_deref().unwrap()).unwrap();
        assert_eq!(meta.kind, MINECRAFT_META_KIND);
        assert!(meta.pack_mcmeta.as_deref().unwrap().contains("\"pack\""));
        assert_eq!(meta.entries.len(), 2);
        assert_eq!(meta.sidecars.len(), 1);
        assert_eq!(
            meta.sidecars[0].resource,
            "example:textures/block/fallback.png"
        );
        assert_eq!(meta.sidecars[0].overlay, None);
        assert_eq!(
            meta.entries
                .iter()
                .find(|entry| entry.overlay.as_deref() == Some("patch"))
                .unwrap()
                .resource,
            "example:textures/block/animated.png"
        );
        assert_eq!(
            meta.entries
                .iter()
                .find(|entry| entry.overlay.is_none())
                .unwrap()
                .resource,
            "example:textures/block/animated.png"
        );
        let base = meta
            .entries
            .iter()
            .find(|entry| entry.overlay.is_none())
            .unwrap();
        assert_eq!((base.source_width, base.source_height), (4, 8));
        assert!(base.mcmeta.as_deref().unwrap().contains("\"animation\""));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn minecraft_vtp_can_hold_multiple_atlas_blobs() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("vrawtex-minecraft-vtp-split-test-{unique}"));
        let texture_dir = root.join("assets/example/textures/block");
        fs::create_dir_all(&texture_dir).unwrap();
        fs::write(
            root.join("pack.mcmeta"),
            r#"{"pack":{"description":"split","pack_format":1}}"#,
        )
        .unwrap();
        RgbaImage::from_pixel(8, 8, Rgba([255, 0, 0, 255]))
            .save(texture_dir.join("a.png"))
            .unwrap();
        RgbaImage::from_pixel(8, 8, Rgba([0, 255, 0, 255]))
            .save(texture_dir.join("b.png"))
            .unwrap();

        let out = root.join("split.vtp");
        atlas_cmd(
            vec![root.clone()],
            Some(out.clone()),
            Some(10),
            Some(1),
            None,
            crate::EncodePixelFormat::Rgba8,
            crate::CompressionProfile::Balance,
            Some(MinecraftPackOptions::default()),
            false,
        )
        .unwrap();

        let bytes = fs::read(out).unwrap();
        let header = decode_texture_pack_header(&bytes).unwrap();
        assert_eq!(header.atlases.len(), 2);
        assert_eq!(
            header
                .atlases
                .iter()
                .map(|atlas| atlas.entries)
                .sum::<u32>(),
            2
        );

        fs::remove_dir_all(root).unwrap();
    }
}
