# VRAWTEX and VTP format specification

This document describes the binary formats produced by the current `vrawtex`
encoder:

- VRAWTEX container version 2 (`.vrawtex`);
- VRAWTEX Texture Pack container version 1 (`.vtp`);
- MessagePack metadata used by atlases, mipchains, and Minecraft texture packs.

The format is lossless for the selected pixel format. `RGBA8` preserves all four
8-bit channels. `RGB8` intentionally discards input alpha before encoding.

## Conventions

- All fixed-width integers in binary container headers are unsigned and
  little-endian.
- Pixel coordinates use a top-left origin.
- Pixel and plane order is row-major, left-to-right and top-to-bottom.
- Arithmetic in color transforms and predictors is modulo 256 (`wrapping_add`
  and `wrapping_sub` on `u8`).
- MessagePack integers use normal MessagePack representation, not the
  fixed-width little-endian convention.
- `ceil_div(a, b)` means `(a + b - 1) / b` for non-negative integers.

## VRAWTEX v2 container

### Top-level layout

```text
+---------------------------+
| fixed header (17 bytes)   |
+---------------------------+
| metadata length (optional)|  u32 LE
+---------------------------+
| metadata bytes (optional) |  MessagePack for known schemas
+---------------------------+
| stream 0 header           |  R
| stream 0 Zstd frame       |
+---------------------------+
| stream 1 header           |  G
| stream 1 Zstd frame       |
+---------------------------+
| stream 2 header           |  B
| stream 2 Zstd frame       |
+---------------------------+
| stream 3 header (optional)|  A or packed alpha mask
| stream 3 Zstd frame       |
+---------------------------+
```

Stream headers are interleaved with their stream data. The file does not contain
one shared table followed by all compressed payloads.

### Fixed header

| Offset | Size | Field | Meaning |
|---:|---:|---|---|
| 0 | 4 | `magic` | ASCII `VRTX` (`56 52 54 58`) |
| 4 | 1 | `version` | `2` for the current format |
| 5 | 4 | `flags` | Pixel format, feature byte, channel count |
| 9 | 8 | `dimmask` | Width in high 32 bits, height in low 32 bits |

The fixed header size is 17 bytes.

```text
flags = (pixfmt << 16) | (qval << 8) | channels
dimmask = (width << 32) | height
```

### Flags

```text
31                         16 15             8 7              0
+----------------------------+----------------+----------------+
| pixfmt (u16)               | qval (u8)     | channels (u8)  |
+----------------------------+----------------+----------------+
```

Current values:

| Field | Value | Meaning |
|---|---:|---|
| `pixfmt` | `0x0001` | Unsigned 8-bit samples |
| `channels` | `3` | RGB8 |
| `channels` | `4` | RGBA8 |

The `qval` feature byte is laid out as follows:

| Bits | Name | Meaning |
|---:|---|---|
| 0 | `has_predictor` | At least one stored stream uses a predictor; informational in v2 because each stream also stores its predictor ID |
| 1..2 | `alpha_mode` | Alpha representation, values 0 through 3 |
| 3 | `has_meta` | A metadata length and metadata block follow the fixed header |
| 4..5 | `color_transform` | Reversible RGB transform, values 0 through 3 |
| 6..7 | reserved | Must be written as zero |

### Metadata block

If `has_meta == 1`, the fixed header is followed by:

| Size | Field |
|---:|---|
| 4 | `meta_len`, u32 LE |
| `meta_len` | Opaque metadata bytes |

The core image decoder does not need to understand metadata. Known metadata
schemas in this project use MessagePack and are described below.

If `has_meta == 0`, neither `meta_len` nor metadata bytes are present.

### Alpha modes and stream count

| ID | Name | Stored alpha payload |
|---:|---|---|
| 0 | `Normal` | One full `width * height` byte plane |
| 1 | `Opaque255` | No alpha stream; reconstruct every alpha sample as 255 |
| 2 | `Transparent0` | No alpha stream; reconstruct every alpha sample as 0 |
| 3 | `Mask1Bit` | One packed bit per pixel, then Zstd-compressed |

RGB streams are always present in order `R`, `G`, `B`. The alpha stream is
present only when `channels == 4` and alpha mode is `Normal` or `Mask1Bit`.

For `Mask1Bit`, pixels are visited in normal raster order. Bits are packed
LSB-first:

```text
byte_index = pixel_index >> 3
bit_index  = pixel_index & 7
bit 0 means alpha 0
bit 1 means alpha 255
packed_size = ceil_div(width * height, 8)
```

Unused high bits in the final byte are zero. The mask stream always uses
predictor ID 0 (`None`).

### Stream layout

Each stored stream consists of a 17-byte header immediately followed by one
complete Zstd frame:

| Relative offset | Size | Field |
|---:|---:|---|
| 0 | 8 | `orig_size`, u64 LE |
| 8 | 8 | `comp_size`, u64 LE |
| 16 | 1 | `predictor`, u8 |
| 17 | `comp_size` | Zstd frame |

Expected `orig_size` values:

- RGB stream: `width * height`;
- normal alpha stream: `width * height`;
- 1-bit alpha stream: `ceil_div(width * height, 8)`.

Each stream is compressed independently. Zstd level, worker count, and chunk
size are encoder tuning parameters and are not part of the file format. A
decoder only needs a conforming Zstd frame decoder.

The reference CLI exposes `fast`, `balance`, and `compact` compression profiles,
currently mapped to Zstd levels 8, 10, and 16. `balance` is the default. These
profiles do not alter container flags, stream order, or decoder compatibility.

### Color transforms

Color transforms run before per-plane predictors. They only affect RGB.

| ID | Name | Forward transform |
|---:|---|---|
| 0 | `None` | `R'=R, G'=G, B'=B` |
| 1 | `SubGreen` | `R'=R-G, G'=G, B'=B-G` |
| 2 | `SubRed` | `R'=R, G'=G-R, B'=B-R` |
| 3 | `SubBlue` | `R'=R-B, G'=G-B, B'=B` |

All subtraction wraps modulo 256. Inverse transforms add the unchanged
reference channel back:

```text
SubGreen: R = R' + G', G = G',      B = B' + G'
SubRed:   R = R',      G = G' + R', B = B' + R'
SubBlue:  R = R' + B', G = G' + B', B = B'
```

### Predictors

Predictors operate independently on each transformed plane and reset at every
row. Let `x` be the current original sample, `left` the decoded sample to its
left, `up` the decoded sample above it, and `up_left` the decoded diagonal
sample. Missing neighbours are zero.

| ID | Name | Encoded residual |
|---:|---|---|
| 0 | `None` | `e = x` |
| 1 | `Delta` | first column: `e=x`; otherwise `e=x-left` |
| 2 | `Paeth` | `e=x-paeth(left, up, up_left)` |
| 3 | `Up` | `e=x-up` |

The Paeth selector is equivalent to the PNG Paeth predictor:

```text
p  = left + up - up_left
pa = abs(p - left)
pb = abs(p - up)
pc = abs(p - up_left)

choose left if pa <= pb and pa <= pc
else choose up if pb <= pc
else choose up_left
```

Decoding adds the same prediction to each residual modulo 256. Rows must be
decoded top-to-bottom for `Up` and `Paeth`. Different planes may be decoded in
parallel.

### Encoder pipeline

The current encoder performs these logical steps:

1. Decode the source image and convert it to interleaved RGBA8.
2. Select output pixel format (`RGBA8` or `RGB8`).
3. Detect constant or binary alpha representation.
4. Sample the image and choose one RGB color transform plus one predictor per
   stored channel.
5. Process rows in raster order:
   - deinterleave into planar channels;
   - apply the selected RGB color transform;
   - apply each channel predictor;
   - pack binary alpha when required;
   - feed each stream to an independent Zstd encoder.
6. Write the fixed header, optional metadata, and stream header/payload pairs.

The sampling heuristics affect size and speed, but not decoder behaviour or the
binary grammar.

### Decoder pipeline

A conforming decoder can use the following order:

```text
read and validate VRTX header
derive width, height, stream count, and expected decompressed sizes
read optional metadata

for each stored stream in R, G, B, [A] order:
    read orig_size, comp_size, predictor
    bounds-check comp_size
    Zstd-decompress exactly orig_size bytes

reconstruct implicit alpha or unpack Mask1Bit alpha
inverse-predict normal byte planes
inverse RGB color transform
interleave planes if packed RGBA/RGB output is required
```

Strict decoding in the reference implementation also rejects unsupported pixel
formats, invalid dimensions, unexpected stream sizes, oversized metadata, and
trailing bytes after the final stream.

The current strict limits are:

| Limit | Value |
|---|---:|
| Maximum width or height | 22,000 pixels |
| Maximum pixel count | 484,000,000 pixels |
| Maximum decoded planar data | 2,000,000,000 bytes |
| Maximum metadata block | 16 MiB |

## VRAWTEX metadata schemas

Metadata is selected by context and by schema-specific `kind` fields where
available. Readers should treat unrecognised metadata as opaque bytes.

### Generic atlas metadata

Generic atlas metadata is encoded with `rmp_serde::to_vec` as the MessagePack
equivalent of this tuple:

```text
[pad_u16, [[id_u32, packed_rect_u64], ...]]
```

Rectangle packing:

```text
packed_rect = (x << 48) | (y << 32) | (width << 16) | height
```

Each rectangle component is limited to `u16`.

### Single-image mipchain metadata

This schema is encoded as a named MessagePack map:

```json
{
  "kind": "vrawtex.mipchain",
  "version": 1,
  "pad": 1,
  "levels": [
    { "level": 0, "x": 1, "y": 1, "w": 1024, "h": 512 }
  ]
}
```

The VRAWTEX image is an atlas containing all listed mip levels. `x/y/w/h`
identify the actual level pixels; surrounding padding is outside the rectangle.

### Minecraft atlas metadata

Minecraft atlas metadata version 2 is a named MessagePack map:

```json
{
  "kind": "vrawtex.minecraft_atlas",
  "version": 2,
  "pad": 2,
  "pack_mcmeta": "optional original JSON text",
  "sidecars": [
    {
      "resource": "minecraft:textures/block/example.png",
      "overlay": null,
      "mcmeta": "original .png.mcmeta JSON text"
    }
  ],
  "entries": [
    {
      "id": 0,
      "resource": "minecraft:textures/block/stone.png",
      "overlay": null,
      "x": 2,
      "y": 2,
      "w": 16,
      "h": 16,
      "source_width": 16,
      "source_height": 16,
      "mcmeta": null
    }
  ]
}
```

`overlay == null` denotes the base `assets/` tree. A string denotes the overlay
directory declared by `pack.mcmeta`. `sidecars` stores `.png.mcmeta` files whose
PNG is supplied by another resource-pack layer.

## VTP v1 texture-pack container

VTP groups pack-level information, an optional icon, and one or more complete
VRAWTEX atlas files into a single `.vtp` file.

### Binary layout

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | Magic `VRAWVTP\0` |
| 8 | 2 | Container version, u16 LE, currently `1` |
| 10 | 8 | `header_len`, u64 LE |
| 18 | `header_len` | Named MessagePack `TexturePackHeader` |
| `18 + header_len` | remaining | Icon and atlas blobs |

There is no alignment or padding between blobs. All offsets stored in the header
are absolute offsets from the beginning of the `.vtp` file.

### VTP MessagePack header

The header is equivalent to:

```json
{
  "kind": "vrawtex.texture_pack",
  "version": 1,
  "name": "Display name",
  "description": "Short description",
  "pack_mcmeta": "optional original JSON text",
  "sidecars": [],
  "blob_section_offset": 1234,
  "icon": {
    "offset": 1234,
    "len": 4096,
    "format": "png"
  },
  "atlases": [
    {
      "index": 0,
      "offset": 5330,
      "len": 1000000,
      "width": 8192,
      "height": 8192,
      "entries": 3500
    }
  ]
}
```

`icon` may be `null`. Its payload is stored unchanged; `format` currently comes
from the source extension (`png`, `raw`, or `vrawtex`, otherwise `raw`). Each
atlas payload is a complete VRAWTEX v2 file and contains its own Minecraft atlas
metadata for resource-to-rectangle lookup.

Producers write blobs in this order:

1. icon, when present;
2. atlas 0;
3. atlas 1, and so on.

Readers should use offsets rather than assuming this order and must validate
every `offset + len` against the VTP file size before slicing a blob.

VTP currently accelerates and packages pixel resources. Non-pixel Minecraft
resources such as models, blockstates, shaders, and OptiFine properties are not
embedded and remain the responsibility of the normal resource-pack layer.

## Compatibility modes

### VRAWTEX v1

The decoder accepts `VRTX` files with container version `1`. They use the same
17-byte fixed header, optional metadata block, and 17-byte per-stream headers as
v2. Predictor IDs are stored per stream. Version 1 has no RGB color transform,
so readers must treat it as `ColorTransform::None`; the current encoder only
writes version 2.

### Legacy files without magic

The decoder has a relaxed compatibility path for old files without `VRTX`
magic. This is not the recommended format for new producers.

Legacy layout begins directly with:

```text
flags   : u32 LE
dimmask : u64 LE
```

Legacy stream headers contain only `orig_size` and `comp_size` (16 bytes). The
global feature bit selects `Delta` for stored streams; a packed alpha mask still
uses `None`. Legacy files have no v2 color transform.

Use strict mode for owned/current files and relaxed mode only when old data must
be recovered.

## Versioning and forward compatibility

- Current VRAWTEX container version: `2`.
- Current VTP container/header version: `1`.
- Current mipchain metadata version: `1`.
- Current Minecraft atlas metadata version: `2`.
- Reserved bits must be zero when writing.
- Readers must reject unsupported mandatory container versions.
- Readers may preserve or ignore unknown metadata because image stream decoding
  does not depend on metadata contents.
