# VRAWTEX API

Публичный интерфейс существует в двух вариантах:

- Rust API из модуля `vrawtex::api`;
- стабильный C ABI из `libvrawtex.so`, пригодный для C, C++, JVM/JNA,
  Python `ctypes`, C# P/Invoke и других FFI.

Бинарная грамматика контейнеров описана в [format.md](format.md).

## Подключение как Rust-библиотеки

Пока crate не опубликован в crates.io, добавьте локальную зависимость:

```toml
[dependencies]
vrawtex = { path = "/path/to/vrawtex" }
```

Минимальная упаковка RGBA16 из памяти:

```rust
use vrawtex::api::{EncodeOptions, encode_rgba16};
use vrawtex::{CompressionProfile, EncodePixelFormat};

let pixels: Vec<u16> = vec![
    1, 255, 256, 65535,
    4096, 8192, 16384, 32768,
];
let options = EncodeOptions {
    pixel_format: EncodePixelFormat::Rgba16,
    compression: CompressionProfile::Balance,
    verbose: false,
};
let encoded = encode_rgba16(2, 1, &pixels, None, &options)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Упаковка файла автоматически распознаёт still image или анимацию:

```rust
use std::path::Path;
use vrawtex::api::{EncodeOptions, encode_file};

encode_file(
    Path::new("input.webp"),
    Path::new("output.vrawtex"),
    &EncodeOptions::default(),
)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Декодирование из памяти:

```rust
use vrawtex::DecodeSafety;
use vrawtex::api::{DecodedAsset, decode_asset};

match decode_asset(&encoded, DecodeSafety::Strict)? {
    DecodedAsset::Image(image) => println!("{}x{}", image.width, image.height),
    DecodedAsset::Animation(animation) => println!("{} frames", animation.frames.len()),
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

### Rust-функции

| Функция | Назначение |
|---|---|
| `encode_rgba8` | Упаковывает interleaved RGBA8 из памяти в один VRAWTEX blob. `RGB8` намеренно отбрасывает alpha. |
| `encode_rgba16` | То же для native `u16`; на диск каналы записываются как U16LE. |
| `encode_animation_rgba8` | Упаковывает массив `AnimationFrame8` в VRAWANM. Принимает только RGB8/RGBA8. |
| `encode_animation_rgba16` | Упаковывает native 16-bit кадры без потери младших битов. |
| `decode_image` | Декодирует обычный VRAWTEX в `DecodedImage`; для анимации возвращает ошибку. |
| `decode_asset` | Автоматически возвращает `DecodedAsset::Image` или `DecodedAsset::Animation`. |
| `encode_file_to_vec` | Загружает поддерживаемый файл и возвращает готовый контейнер в памяти. |
| `encode_file` | Загружает файл и сразу записывает контейнер по указанному пути. |
| `inspect_animation` | Читает MessagePack header анимации без распаковки кадров. |

`EncodeOptions` задаёт `pixel_format`, `compression` и подробный вывод.
`PixelData` содержит `Vec<u8>` либо `Vec<u16>`. `metadata` в `DecodedImage`
остаётся opaque blob, поэтому сторонний проект может разбирать собственную
MessagePack-схему независимо от кодека.

Для анимации `duration_num_ms / duration_den_ms` задаёт длительность кадра в
миллисекундах. Нулевой denominator запрещён. `loop_count == 0` означает
бесконечное повторение. Encoder пишет VRAWANM v2 и адаптивно выбирает `full`,
`copy`, `rect` или `motion`; decoder также принимает старый VRAWANM v1.

Полная rustdoc-документация строится так:

```bash
cargo doc --no-deps --open
```

## Подключение скомпилированной библиотеки

Сборка создаёт Rust `rlib` и shared library:

```bash
cargo build --release --lib
```

На Linux результат находится в `target/release/libvrawtex.so`. C-заголовок:
[include/vrawtex.h](include/vrawtex.h).

Пример компоновки:

```bash
cc example.c -I./include -L./target/release -lvrawtex \
  -Wl,-rpath,"$PWD/target/release" -o example
```

Пример вызова memory API:

```c
#include <stdio.h>
#include "vrawtex.h"

int main(void) {
    uint8_t rgba[8] = {255, 0, 0, 255, 0, 255, 0, 128};
    VrawtexBuffer out = {0};
    int32_t status = vrawtex_encode_rgba8(
        rgba, 8, 2, 1, VRAWTEX_RGBA8, VRAWTEX_BALANCE, &out);
    if (status != 0) {
        fprintf(stderr, "%s\n", vrawtex_last_error_message());
        return status;
    }
    fwrite(out.data, 1, out.len, stdout);
    vrawtex_buffer_free(out);
    return 0;
}
```

### C ABI-функции

| Функция | Контракт |
|---|---|
| `vrawtex_last_error_message` | Возвращает thread-local UTF-8 сообщение последней ошибки. Pointer действует до следующего API-вызова в этом потоке. |
| `vrawtex_buffer_free` | Освобождает успешный `VrawtexBuffer`. Вызывать ровно один раз. |
| `vrawtex_encode_file` | Кодирует still image/GIF/WebP/MP4/JXL по UTF-8 путям. JXL встроен; только MP4/MOV требует опциональный FFmpeg backend. |
| `vrawtex_decode_file` | Пишет PNG/RAW; для анимации создаёт каталог кадров и `animation.json`. |
| `vrawtex_encode_rgba8` | Кодирует `width * height * 4` байтов из памяти. Допустимы RGBA8/RGB8. |
| `vrawtex_encode_rgba16` | Кодирует столько же `uint16_t` samples. Допустимы RGBA16/RGB16. |

Коды возврата: `0` — успех, `1` — проверенная ошибка входа/кодека/I/O,
`2` — перехваченная panic. При ошибке output buffer не принадлежит вызывающему
коду и освобождать его не нужно.

Значения enum зафиксированы в заголовке. Профили соответствуют Zstd
`fast=8`, `balance=10`, `compact=16`, `ultra=22`.
