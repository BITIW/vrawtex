use image::RgbaImage;
use rayon::prelude::*;

// ===== Fixed-point Q32.32 =====
pub type Fixed = i64;
pub const FRAC_BITS: u32 = 32;
pub const ONE: Fixed = 1i64 << FRAC_BITS;
pub const HALF: Fixed = ONE >> 1;

// pi в Q32.32
pub const PI_Q: Fixed = 13_493_037_705;
// sin poly коэффициенты
pub const C3_Q: Fixed = -715_827_883;
pub const C5_Q: Fixed = 35_791_394;
pub const C7_Q: Fixed = -852_176;

#[inline(always)]
fn fixed_from_int(n: i64) -> Fixed {
    n << FRAC_BITS
}

#[inline(always)]
fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    ((a as i128 * b as i128) >> FRAC_BITS) as Fixed
}

#[inline(always)]
fn fixed_div(a: Fixed, b: Fixed) -> Fixed {
    let res = ((a as i128) << FRAC_BITS) / (b as i128);
    res as Fixed
}

/// acc в Q32.32 на i64 → u8
#[inline(always)]
fn q_to_u8_from_acc_i64(acc: i64) -> u8 {
    let v = (acc >> FRAC_BITS) as i64;
    if v <= 0 {
        0
    } else if v >= 255 {
        255
    } else {
        v as u8
    }
}

// ===== Integer-only sin / sinc / Lanczos =====
#[inline(always)]
fn sin_poly(t: Fixed) -> Fixed {
    let t2 = fixed_mul(t, t);
    let mut p = C7_Q;
    p = C5_Q + fixed_mul(p, t2);
    p = C3_Q + fixed_mul(p, t2);
    let t3 = fixed_mul(t2, t);
    t + fixed_mul(t3, p)
}

/// sin(pi * x), x в Q32.32
#[inline(always)]
fn sin_pi_x(x: Fixed) -> Fixed {
    if x == 0 {
        return 0;
    }
    let mut x_abs = x;
    let mut sign = 1i64;
    if x_abs < 0 {
        x_abs = -x_abs;
        sign = -1;
    }

    // x = k + r
    let k = (x_abs >> FRAC_BITS) as i64;
    let r = x_abs - (k << FRAC_BITS);

    // t = pi * r
    let t = fixed_mul(PI_Q, r);
    let s = sin_poly(t);

    // sin(pix) = (-1)^k * sin(pir)
    let sign_k = if (k & 1) == 0 { 1i64 } else { -1i64 };
    s * sign * sign_k
}

/// sinc(x) = sin(pix)/(pix), x в Q32.32
#[inline(always)]
fn sinc_basic(x: Fixed) -> Fixed {
    if x == 0 {
        return ONE;
    }
    let sinv = sin_pi_x(x);
    let pix = fixed_mul(PI_Q, x);
    fixed_div(sinv, pix)
}

/// Lanczos: L(d)=sinc(d)*sinc(d/a), 0<=d<a
#[inline(always)]
fn lanczos_kernel(dist: Fixed, radius_q: Fixed) -> Fixed {
    if dist >= radius_q {
        return 0;
    }
    let s1 = sinc_basic(dist);
    let dist_over_r = fixed_div(dist, radius_q);
    let s2 = sinc_basic(dist_over_r);
    fixed_mul(s1, s2)
}

#[derive(Debug, Clone)]
struct Kernel1D {
    indices: Vec<u32>,
    weights_q: Vec<i64>, // Q32.32
}

#[derive(Debug, Clone, Copy)]
pub struct ResizeStats {
    pub taps_total: u64,
    pub ops_total: u64,
}

/// build kernels for 1D resample, radius_px = "a"
fn build_kernels_1d(src_len: u32, dst_len: u32, radius_q: Fixed) -> Vec<Kernel1D> {
    let scale_q: Fixed = ((src_len as i64) << FRAC_BITS) / (dst_len as i64);

    (0..dst_len)
        .into_par_iter()
        .map(|dst_i| {
            // center = (i+0.5)*scale - 0.5
            let i_q = (dst_i as i64) << FRAC_BITS;
            let mut center_q = i_q + HALF;
            center_q = fixed_mul(center_q, scale_q);
            center_q -= HALF;

            let left_q = center_q - radius_q;
            let right_q = center_q + radius_q;
            let left = ((left_q >> FRAC_BITS).max(0) as u32).min(src_len - 1);
            let right = ((right_q >> FRAC_BITS).max(0) as u32).min(src_len - 1);

            let capacity = right.saturating_sub(left) as usize + 1;
            let mut indices = Vec::with_capacity(capacity);
            let mut weights_tmp = Vec::with_capacity(capacity);
            let mut sum_w: i128 = 0;

            for src_i in left..=right {
                let src_q = (src_i as i64) << FRAC_BITS;
                let mut dist_q = center_q - src_q;
                if dist_q < 0 {
                    dist_q = -dist_q;
                }
                if dist_q >= radius_q {
                    continue;
                }

                let w_q = lanczos_kernel(dist_q, radius_q);
                if w_q == 0 {
                    continue;
                }
                indices.push(src_i);
                weights_tmp.push(w_q);
                sum_w += w_q as i128;
            }

            // normalize weights
            let mut weights_q = Vec::with_capacity(weights_tmp.len());
            if sum_w != 0 {
                for w in weights_tmp {
                    let w_norm = ((w as i128) << FRAC_BITS) / sum_w;
                    weights_q.push(w_norm as i64);
                }
            }

            Kernel1D { indices, weights_q }
        })
        .collect()
}

/// 1D resample RGB (src_len*3 bytes) -> (dst_len*3 bytes)
/// Используется для edge-signature
pub fn resample_1d_rgb(src_rgb: &[u8], src_len: u32, dst_len: u32, radius_px: u32) -> Vec<u8> {
    assert_eq!(src_rgb.len(), src_len as usize * 3);
    if src_len == 0 || dst_len == 0 {
        return Vec::new();
    }
    if src_len == dst_len {
        return src_rgb.to_vec();
    }

    let kernels = build_kernels_1d(src_len, dst_len, fixed_from_int(radius_px as i64));
    let mut out = vec![0u8; dst_len as usize * 3];

    out.par_chunks_mut(3).enumerate().for_each(|(dx, pix)| {
        let k = &kernels[dx];

        let mut acc_r: i64 = 0;
        let mut acc_g: i64 = 0;
        let mut acc_b: i64 = 0;

        for (&idx, &wq) in k.indices.iter().zip(k.weights_q.iter()) {
            let i = idx as usize * 3;
            acc_r += (src_rgb[i] as i64) * wq;
            acc_g += (src_rgb[i + 1] as i64) * wq;
            acc_b += (src_rgb[i + 2] as i64) * wq;
        }

        pix[0] = q_to_u8_from_acc_i64(acc_r);
        pix[1] = q_to_u8_from_acc_i64(acc_g);
        pix[2] = q_to_u8_from_acc_i64(acc_b);
    });

    out
}

fn radius_percent_q(src_w: u32, src_h: u32, radius_percent: u32) -> Fixed {
    let radius_frac_q: Fixed = (((radius_percent as i128) << FRAC_BITS) / 100i128) as Fixed;
    fixed_mul(
        fixed_from_int(src_w.min(src_h).max(1) as i64),
        radius_frac_q,
    )
    .max(ONE)
}

/// Полный 2D Lanczos RGBA с радиусом в процентах от меньшей стороны source.
pub fn resize_lanczos_rgba_percent(
    src: &RgbaImage,
    dst_w: u32,
    dst_h: u32,
    radius_percent: u32,
) -> (RgbaImage, ResizeStats) {
    let (src_w, src_h) = src.dimensions();
    let radius_q = radius_percent_q(src_w, src_h, radius_percent.max(1));
    resize_lanczos_rgba_fixed(src, dst_w, dst_h, radius_q)
}

fn resize_lanczos_rgba_fixed(
    src: &RgbaImage,
    dst_w: u32,
    dst_h: u32,
    radius_q: Fixed,
) -> (RgbaImage, ResizeStats) {
    let (src_w, src_h) = src.dimensions();
    assert!(src_w > 0 && src_h > 0 && dst_w > 0 && dst_h > 0);
    if src_w == dst_w && src_h == dst_h {
        return (
            src.clone(),
            ResizeStats {
                taps_total: 0,
                ops_total: 0,
            },
        );
    }
    let src_buf = src.as_raw();

    // ----- Horizontal -----
    let kernels_x = build_kernels_1d(src_w, dst_w, radius_q);
    let taps_x_per_row: u64 = kernels_x.iter().map(|k| k.weights_q.len() as u64).sum();

    let mut tmp_buf = vec![0u8; dst_w as usize * src_h as usize * 4];

    {
        let row_stride_src = src_w as usize * 4;
        let row_stride_dst = dst_w as usize * 4;

        tmp_buf
            .par_chunks_mut(row_stride_dst)
            .enumerate()
            .for_each(|(row, dst_row)| {
                let src_row_start = row * row_stride_src;
                let src_row = &src_buf[src_row_start..src_row_start + row_stride_src];

                for (x, k) in kernels_x.iter().enumerate() {
                    let mut acc_r: i64 = 0;
                    let mut acc_g: i64 = 0;
                    let mut acc_b: i64 = 0;
                    let mut acc_a: i64 = 0;

                    for (&idx, &wq) in k.indices.iter().zip(k.weights_q.iter()) {
                        let i = idx as usize * 4;
                        acc_r += (src_row[i] as i64) * wq;
                        acc_g += (src_row[i + 1] as i64) * wq;
                        acc_b += (src_row[i + 2] as i64) * wq;
                        acc_a += (src_row[i + 3] as i64) * wq;
                    }

                    let base = x * 4;
                    dst_row[base] = q_to_u8_from_acc_i64(acc_r);
                    dst_row[base + 1] = q_to_u8_from_acc_i64(acc_g);
                    dst_row[base + 2] = q_to_u8_from_acc_i64(acc_b);
                    dst_row[base + 3] = q_to_u8_from_acc_i64(acc_a);
                }
            });
    }

    let tmp: RgbaImage = RgbaImage::from_raw(dst_w, src_h, tmp_buf).expect("tmp buf");

    // ----- Vertical -----
    let tmp_buf = tmp.into_raw();
    let kernels_y = build_kernels_1d(src_h, dst_h, radius_q);
    let taps_y_per_col: u64 = kernels_y.iter().map(|k| k.weights_q.len() as u64).sum();

    let mut out_buf = vec![0u8; dst_w as usize * dst_h as usize * 4];

    {
        let row_stride = dst_w as usize * 4;

        out_buf
            .par_chunks_mut(row_stride)
            .enumerate()
            .for_each_init(
                || vec![0i64; row_stride],
                |acc, (dst_y, dst_row)| {
                    let k = &kernels_y[dst_y];
                    acc.fill(0);

                    for (&src_y, &wq) in k.indices.iter().zip(k.weights_q.iter()) {
                        let src_off = src_y as usize * row_stride;
                        let src_row = &tmp_buf[src_off..src_off + row_stride];

                        for (sum, &sample) in acc.iter_mut().zip(src_row.iter()) {
                            *sum += (sample as i64) * wq;
                        }
                    }

                    for (dst_pixel, sums) in dst_row.chunks_exact_mut(4).zip(acc.chunks_exact(4)) {
                        dst_pixel[0] = q_to_u8_from_acc_i64(sums[0]);
                        dst_pixel[1] = q_to_u8_from_acc_i64(sums[1]);
                        dst_pixel[2] = q_to_u8_from_acc_i64(sums[2]);
                        dst_pixel[3] = q_to_u8_from_acc_i64(sums[3]);
                    }
                },
            );
    }

    let out = RgbaImage::from_raw(dst_w, dst_h, out_buf).expect("out buf");

    let taps_total_horiz = taps_x_per_row * src_h as u64;
    let taps_total_vert = taps_y_per_col * dst_w as u64;
    let taps_total = taps_total_horiz + taps_total_vert;
    let ops_total = taps_total * 8;

    let stats = ResizeStats {
        taps_total,
        ops_total,
    };
    (out, stats)
}

/// Edge-signature для edgecost.
/// Делает premultiply RGB по A и усредняет по thickness,
/// затем downsample до K через Lanczos 1D.
/// radius_px тут должен быть маленький (обычно 3..6).
#[derive(Clone, Debug)]
pub struct EdgeSig {
    pub k: u32,
    pub top: Vec<u8>,    // K*3
    pub bottom: Vec<u8>, // K*3
    pub left: Vec<u8>,   // K*3
    pub right: Vec<u8>,  // K*3
    pub left_q: u8,      // 0..31 bucket key
    pub right_q: u8,     // 0..31 bucket key
}

#[inline(always)]
fn luma_u8(r: u8, g: u8, b: u8) -> u8 {
    // cheap-ish integer luma: (0.299, 0.587, 0.114) ~ (77,150,29)/256
    let y = (77u32 * r as u32 + 150u32 * g as u32 + 29u32 * b as u32) >> 8;
    y as u8
}

#[inline(always)]
fn quant5(x: u8) -> u8 {
    // 0..255 -> 0..31
    x >> 3
}

fn edge_row_premul_avg_rgb(src: &RgbaImage, y0: u32, thickness: u32) -> Vec<u8> {
    let (w, h) = src.dimensions();
    let t = thickness.min(h.max(1)).max(1);
    let y0 = y0.min(h.saturating_sub(1));

    let mut out = vec![0u8; w as usize * 3];

    for x in 0..w {
        let mut sr: u32 = 0;
        let mut sg: u32 = 0;
        let mut sb: u32 = 0;

        for dy in 0..t {
            let y = (y0 + dy).min(h - 1);
            let p = src.get_pixel(x, y).0;
            let a = p[3] as u32;
            // premultiply
            sr += (p[0] as u32 * a + 127) / 255;
            sg += (p[1] as u32 * a + 127) / 255;
            sb += (p[2] as u32 * a + 127) / 255;
        }

        sr /= t as u32;
        sg /= t as u32;
        sb /= t as u32;

        let i = x as usize * 3;
        out[i] = sr as u8;
        out[i + 1] = sg as u8;
        out[i + 2] = sb as u8;
    }

    out
}

fn edge_col_premul_avg_rgb(src: &RgbaImage, x0: u32, thickness: u32) -> Vec<u8> {
    let (w, h) = src.dimensions();
    let t = thickness.min(w.max(1)).max(1);
    let x0 = x0.min(w.saturating_sub(1));

    let mut out = vec![0u8; h as usize * 3];

    for y in 0..h {
        let mut sr: u32 = 0;
        let mut sg: u32 = 0;
        let mut sb: u32 = 0;

        for dx in 0..t {
            let x = (x0 + dx).min(w - 1);
            let p = src.get_pixel(x, y).0;
            let a = p[3] as u32;
            sr += (p[0] as u32 * a + 127) / 255;
            sg += (p[1] as u32 * a + 127) / 255;
            sb += (p[2] as u32 * a + 127) / 255;
        }

        sr /= t as u32;
        sg /= t as u32;
        sb /= t as u32;

        let i = y as usize * 3;
        out[i] = sr as u8;
        out[i + 1] = sg as u8;
        out[i + 2] = sb as u8;
    }

    out
}

pub fn edge_signature(src: &RgbaImage, k: u32, thickness: u32, radius_px: u32) -> EdgeSig {
    let (w, h) = src.dimensions();
    let k = k.max(4);
    let radius_px = radius_px.max(1);

    // top/bottom as rows
    let top_src = edge_row_premul_avg_rgb(src, 0, thickness);
    let bot_src = edge_row_premul_avg_rgb(src, h.saturating_sub(1), thickness);

    // left/right as cols
    let left_src = edge_col_premul_avg_rgb(src, 0, thickness);
    let right_src = edge_col_premul_avg_rgb(src, w.saturating_sub(1), thickness);

    let top = resample_1d_rgb(&top_src, w, k, radius_px);
    let bottom = resample_1d_rgb(&bot_src, w, k, radius_px);
    let left = resample_1d_rgb(&left_src, h, k, radius_px);
    let right = resample_1d_rgb(&right_src, h, k, radius_px);

    // bucket keys: avg luma of left/right (на downsampled)
    let mut l_sum: u32 = 0;
    let mut r_sum: u32 = 0;
    for i in 0..(k as usize) {
        let li = i * 3;
        let ri = i * 3;
        l_sum += luma_u8(left[li], left[li + 1], left[li + 2]) as u32;
        r_sum += luma_u8(right[ri], right[ri + 1], right[ri + 2]) as u32;
    }
    let l_avg = (l_sum / k as u32) as u8;
    let r_avg = (r_sum / k as u32) as u8;

    EdgeSig {
        k,
        top,
        bottom,
        left,
        right,
        left_q: quant5(l_avg),
        right_q: quant5(r_avg),
    }
}

/// L1 mismatch (right(A) vs left(B)) — базовый edgecost
pub fn edge_mismatch_lr(a: &EdgeSig, b: &EdgeSig) -> u32 {
    debug_assert_eq!(a.k, b.k);
    let mut s: u32 = 0;
    for i in 0..(a.k as usize * 3) {
        let da = a.right[i] as i32 - b.left[i] as i32;
        s += da.unsigned_abs();
    }
    s
}

/// (опционально) вертикаль: bottom(A) vs top(B)
pub fn edge_mismatch_bt(a: &EdgeSig, b: &EdgeSig) -> u32 {
    debug_assert_eq!(a.k, b.k);
    let mut s: u32 = 0;
    for i in 0..(a.k as usize * 3) {
        let da = a.bottom[i] as i32 - b.top[i] as i32;
        s += da.unsigned_abs();
    }
    s
}
