# vrawtex

`vrawtex` -- небольшой инструмент на Rust для упаковки изображений в собственный контейнер `.vrawtex`, обратного декодирования, просмотра и сборки больших текстурных атласов.

Текущий пайплайн основан на planar `U8` каналах, быстрых обратимых transform/predictor шагах и `zstd`.

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

Отбросить альфа-канал и сохранить RGB8:

```bash
./target/release/vrawtex encode --rgb8 input.png
```

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
- Minecraft atlas blobs сканируют `assets/<namespace>/**/*.png` и объявленные в `pack.mcmeta` overlay-слои. В metadata сохраняются resource location, имя overlay, исходный размер, atlas rect, соседний `.png.mcmeta`, orphan `.png.mcmeta` без собственной текстуры и `pack.mcmeta`. Остальные файлы resource pack остаются обычными файлами для Minecraft ResourceManager.
- `--minecraft` пока несовместим с `--mipchain` и `--rgb8`: mip-уровни анимированных frame-strip текстур нужно уменьшать покадрово, а RGB8 уничтожает используемую Minecraft прозрачность.
- `--rgb8` доступен для `encode` и `atlas`; альфа при этом полностью выбрасывается.
- `--mipchain N` создаёт `N` дополнительных уровней после `mip0`; без `N` цепочка строится полностью.
- `--size` задаёт размеры дополнительных уровней и требует такое же количество значений, как в `--mipchain N`. Значения должны строго уменьшаться и быть меньше оригинала.
- Mip-уровни уменьшаются последовательно до `1x1` через integer Lanczos с радиусом 100% от меньшей стороны предыдущего уровня.
- Одиночный mipchain хранит уровни в одном атласе с mipchain-meta. В `atlas --mipchain` каждая исходная текстура уменьшается отдельно, после чего уровень заново пакуется с обычной atlas-meta.
- Входные изображения: PNG, JPEG, BMP, TGA, TIFF, GIF и DNG.
- DNG сначала проверяется на встроенный полноразмерный JPEG preview (часто встречается в Samsung/LinearRaw/JPEG XL DNG), затем декодируется встроенными Rust-декодерами; для некоторых mobile/LinearRaw DNG используется fallback через ImageMagick (`magick`), если он установлен в системе.
- Для `decode` и `inspect` поддерживается `--safety strict|relaxed`.
- Распознанные vtp/atlas/minecraft-atlas/mipchain-метаданные можно выгрузить через `--dump-meta`.
- Подробный режим включается через `-v` (Рекомендую к использованию, можно узнать много интересного)
