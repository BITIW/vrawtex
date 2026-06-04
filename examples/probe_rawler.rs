use std::env;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).ok_or("need path")?;
    let raw = rawler::decode_file(&path)?;

    println!("make={} model={}", raw.make, raw.model);
    println!(
        "size={}x{} cpp={} bps={} photometric={:?}",
        raw.width, raw.height, raw.cpp, raw.bps, raw.photometric
    );
    println!("wb={:?}", raw.wb_coeffs);
    println!("white={:?}", raw.whitelevel.as_vec());
    println!(
        "black width={} height={} cpp={} levels={:?}",
        raw.blacklevel.width,
        raw.blacklevel.height,
        raw.blacklevel.cpp,
        raw.blacklevel.as_vec()
    );
    println!(
        "active={:?} crop={:?} orientation={:?}",
        raw.active_area, raw.crop_area, raw.orientation
    );
    println!("color_matrix={:?}", raw.color_matrix);

    let mut min = vec![f32::INFINITY; raw.cpp];
    let mut max = vec![f32::NEG_INFINITY; raw.cpp];
    let mut sum = vec![0.0f64; raw.cpp];
    let pixels = raw.width * raw.height;

    match &raw.data {
        rawler::RawImageData::Integer(v) => {
            for pix in v.chunks_exact(raw.cpp) {
                for c in 0..raw.cpp {
                    let value = pix[c] as f32;
                    min[c] = min[c].min(value);
                    max[c] = max[c].max(value);
                    sum[c] += value as f64;
                }
            }
            println!("first 24 integer samples={:?}", &v[..v.len().min(24)]);
        }
        rawler::RawImageData::Float(v) => {
            for pix in v.chunks_exact(raw.cpp) {
                for c in 0..raw.cpp {
                    let value = pix[c];
                    min[c] = min[c].min(value);
                    max[c] = max[c].max(value);
                    sum[c] += value as f64;
                }
            }
            println!("first 24 float samples={:?}", &v[..v.len().min(24)]);
        }
    }

    println!("min={min:?}");
    println!("max={max:?}");
    let mean: Vec<f64> = sum.into_iter().map(|v| v / pixels as f64).collect();
    println!("mean={mean:?}");

    Ok(())
}
