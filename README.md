# vrawtex

`vrawtex` -- небольшой инструмент на Rust для упаковки изображений в собственный контейнер `.vrawtex`, обратного декодирования, просмотра и сборки больших текстурных атласов.

Текущий пайплайн основан на planar `U8`/`U16LE` каналах, быстрых обратимых transform/predictor шагах и `zstd`.

Полная бинарная спецификация VRAWTEX v2 и VTP v1: [format.md](format.md).
Интеграция через Rust API или `libvrawtex.so`: [API.md](API.md).

## Сборка

```bash
cargo build --release
```

Путь к бинарнику:

```bash
./target/release/vrawtex
```

## Команды

Закодировать одно изображение:

```bash
./target/release/vrawtex encode input.png
```

Выбрать профиль сжатия:

```bash
./target/release/vrawtex encode --profile fast input.png
./target/release/vrawtex encode --profile balance input.png
./target/release/vrawtex encode --profile compact input.png
./target/release/vrawtex encode --profile ultra input.png
```

| Профиль | Zstd | Назначение |
|---|---:|---|
| `fast` | 8 | Максимальная скорость упаковки |
| `balance` | 10 | Баланс скорости и размера, используется по умолчанию |
| `compact` | 16 | Минимальный размер ценой заметно более долгой упаковки |
| `ultra` | 22 | Максимальное усилие Zstd; полезен для финальной раздачи, но очень медленный |

Отбросить альфа-канал и сохранить RGB8:

```bash
./target/release/vrawtex encode --rgb8 input.png
```

Сохранить настоящий 16-битный цвет или отбросить только alpha:

```bash
./target/release/vrawtex encode --rgba16 input-16bit.png
./target/release/vrawtex encode --rgb16 input-16bit.png
```

GIF, animated WebP, MP4/MOV и animated JXL автоматически упаковываются в
animation-контейнер. VRAWANM v2 адаптивно выбирает полноценный keyframe,
дельту от первого/предыдущего кадра, zero-blob copy, локальный changed rect или
глобальный motion residual. Каждый непустой кадр остаётся отдельным VRAWTEX
blob; старые VRAWANM v1 продолжают читаться:

```bash
./target/release/vrawtex encode animation.webp
./target/release/vrawtex encode --rgba16 animation.jxl
./target/release/vrawtex decode animation.vrawtex
```

При декодировании создаётся каталог `animation_frames/` с PNG/RAW кадрами и
`animation.json`. `open` воспроизводит контейнер с сохранёнными задержками и
потоково декодирует кадры: prefetch стремится к 2 секундам, но ограничен 256 MiB.

Собрать полную mipchain в одном VRAWTEX-атласе:

```bash
./target/release/vrawtex encode --mipchain input.png
```

Ограничить mipchain четырьмя дополнительными уровнями после `mip0`:

```bash
./target/release/vrawtex encode --mipchain 4 input.png
```

Задать точные высоты уровней с сохранением пропорций:

```bash
./target/release/vrawtex encode --mipchain 4 --size 1080,720,240,32 input.png
```

На вход также поддерживаются `.dng` файлы:

```bash
./target/release/vrawtex encode photo.dng
```

Закодировать директорию рекурсивно:

```bash
./target/release/vrawtex encode -r assets/
```

Декодировать в PNG:

```bash
./target/release/vrawtex decode texture.vrawtex
```

Декодировать в planar RAW:

```bash
./target/release/vrawtex decode texture.vrawtex -t raw
```

Открыть `.vrawtex` во встроенном просмотрщике:

```bash
./target/release/vrawtex open texture.vrawtex
```

Управление просмотрщиком: колесо мыши или `+/-` меняют масштаб, `0` возвращает fit-to-window, `1` включает 100%, ЛКМ перетаскивает изображение, стрелки сдвигают viewport.

Посмотреть заголовки и метаданные:

```bash
./target/release/vrawtex inspect texture.vrawtex
```

Собрать текстурный атлас:

```bash
./target/release/vrawtex -v atlas --max-side 22000 assets/
```

Собрать отдельные `atlas_mip0.vrawtex`, `atlas_mip1.vrawtex` и последующие уровни:

```bash
./target/release/vrawtex -v atlas --mipchain assets/
```

Для atlas `--size` задаёт точные стороны выходных atlas mip-файлов:

```bash
./target/release/vrawtex -v atlas --mipchain 4 --size 1080,720,240,32 assets/
```

Упаковать Minecraft resource pack в `.vtp`:

```bash
./target/release/vrawtex -v atlas --minecraft \
  --name "Faithful 32x" \
  --desc "The go-to 32x resource pack" \
  --ico faithful/pack.png \
  -o faithful.vtp \
  faithful/
```

## Примечания

- Команда `atlas` принимает как отдельные файлы, так и директории.
- `atlas --minecraft` принимает ровно один корень resource pack и пишет один `.vtp` контейнер: header с `name`/`description`/layout, icon blob и один или несколько VRAWTEX atlas blobs.
- `--name`, `--desc` и `--ico` работают только вместе с `--minecraft`. Если `--ico` не задан, используется `pack.png`, если он есть. Если `--desc` не задан, описание берётся из `pack.mcmeta`.
- PNG- и VRAWTEX-иконки сохраняются без перекодирования. Остальные поддерживаемые входы, включая DNG/RAW, JPEG и JXL, декодируются в RGBA8 и кладутся в VTP как VRAWTEX icon blob, чтобы Minecraft-мод мог прочитать их без внешнего image backend.
- Minecraft atlas blobs сканируют `assets/<namespace>/**/*.png` и объявленные в `pack.mcmeta` overlay-слои. В metadata сохраняются resource location, имя overlay, исходный размер, atlas rect, соседний `.png.mcmeta`, orphan `.png.mcmeta` без собственной текстуры и `pack.mcmeta`. Остальные файлы resource pack остаются обычными файлами для Minecraft ResourceManager.
- `--minecraft` пока несовместим с `--mipchain`, RGB8/RGB16 и RGBA16: Minecraft-атлас хранится как RGBA8, включая точную 8-битную прозрачность.
- `--rgb8`, `--rgba16` и `--rgb16` доступны для `encode` и обычного `atlas`; RGB-варианты полностью выбрасывают alpha.
- RGB16/RGBA16 mipchain использует отдельный native-U16 integer Lanczos и не обрезает samples до 8 бит.
- Анимация несовместима с `--mipchain`; каждый кадр уже является отдельным blob одного размера.
- `--profile fast|balance|compact|ultra` доступен для `encode` и `atlas`, включая recursive encode, mipchain и Minecraft VTP. Профиль меняет только уровень Zstd и не влияет на совместимость декодера.
- `--mipchain N` создаёт `N` дополнительных уровней после `mip0`; без `N` цепочка строится полностью.
- `--size` задаёт размеры дополнительных уровней и требует такое же количество значений, как в `--mipchain N`. Значения должны строго уменьшаться и быть меньше оригинала.
- Mip-уровни уменьшаются последовательно до `1x1` через integer Lanczos с радиусом 100% от меньшей стороны предыдущего уровня.
- Одиночный mipchain хранит уровни в одном атласе с mipchain-meta. В `atlas --mipchain` каждая исходная текстура уменьшается отдельно, после чего уровень заново пакуется с обычной atlas-meta.
- Входные изображения: PNG, JPEG, BMP, TGA, TIFF, GIF, WebP, JXL, MP4/M4V/MOV и DNG. GIF/WebP декодируются через `image`, а static/animated JXL — встроенным `jxl-oxide`, в том числе в native 16-bit. Только MP4/M4V/MOV используют опциональные `ffprobe`/`ffmpeg`, потому что внутри контейнера могут быть AV1, H.264, HEVC и другие видеокодеки.
- MP4/MOV читается потоково: в raw-видеопамяти одновременно находятся только frame0, предыдущий и текущий кадр. Для FFmpeg 8 автоматически исправляется некорректная/зарезервированная color-primaries metadata перед RGB-конвертацией.
- Для portable-сборки `ffmpeg` и `ffprobe` можно положить рядом с `vrawtex`; установка в систему не требуется. Нестандартные пути задаются переменными `VRAWTEX_FFMPEG` и `VRAWTEX_FFPROBE`.
- DNG сначала проверяется на встроенный полноразмерный JPEG preview (часто встречается в Samsung/LinearRaw/JPEG XL DNG), затем декодируется встроенными Rust-декодерами; для некоторых mobile/LinearRaw DNG используется fallback через ImageMagick (`magick`), если он установлен в системе.
- Для `decode` и `inspect` поддерживается `--safety strict|relaxed`.
- Распознанные vtp/atlas/minecraft-atlas/mipchain-метаданные можно выгрузить через `--dump-meta`.
- Подробный режим включается через `-v` (Рекомендую к использованию, можно узнать много интересного)
