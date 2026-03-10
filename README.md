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

Посмотреть заголовки и метаданные:

```bash
./target/release/vrawtex inspect texture.vrawtex
```

Собрать текстурный атлас:

```bash
./target/release/vrawtex -v atlas --max-side 22000 assets/
```

## Примечания

- Команда `atlas` принимает как отдельные файлы, так и директории.
- Для `decode` и `inspect` поддерживается `--safety strict|relaxed`.
- Метаданные атласа можно выгрузить через `--dump-meta`.
- Подробный режим включается через `-v` (Рекомендую к использованию, можно узнать много интересного)
