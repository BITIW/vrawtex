#ifndef VRAWTEX_H
#define VRAWTEX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VrawtexPixelFormat {
    VRAWTEX_RGBA8 = 0,
    VRAWTEX_RGB8 = 1,
    VRAWTEX_RGBA16 = 2,
    VRAWTEX_RGB16 = 3
} VrawtexPixelFormat;

typedef enum VrawtexCompressionProfile {
    VRAWTEX_FAST = 0,
    VRAWTEX_BALANCE = 1,
    VRAWTEX_COMPACT = 2,
    VRAWTEX_ULTRA = 3
} VrawtexCompressionProfile;

typedef enum VrawtexDecodeTarget {
    VRAWTEX_DECODE_PNG = 0,
    VRAWTEX_DECODE_RAW = 1
} VrawtexDecodeTarget;

typedef enum VrawtexDecodeSafety {
    VRAWTEX_STRICT = 0,
    VRAWTEX_RELAXED = 1
} VrawtexDecodeSafety;

typedef struct VrawtexBuffer {
    uint8_t *data;
    size_t len;
    size_t capacity;
} VrawtexBuffer;

/* Returned pointer is thread-local and valid until the next API call. */
const char *vrawtex_last_error_message(void);

/* Release a successful memory-encode result exactly once. */
void vrawtex_buffer_free(VrawtexBuffer buffer);

/* Encode any supported still image or animation from a UTF-8 filesystem path. */
int32_t vrawtex_encode_file(const char *input,
                            const char *output,
                            uint32_t pixel_format,
                            uint32_t profile);

/* Decode to PNG/RAW. Animations produce a frame directory and animation.json. */
int32_t vrawtex_decode_file(const char *input,
                            const char *output,
                            uint32_t target,
                            uint32_t safety);

/* rgba_len is the number of uint8_t channel samples (width * height * 4). */
int32_t vrawtex_encode_rgba8(const uint8_t *rgba,
                             size_t rgba_len,
                             uint32_t width,
                             uint32_t height,
                             uint32_t pixel_format,
                             uint32_t profile,
                             VrawtexBuffer *output);

/* rgba_len is the number of uint16_t channel samples (width * height * 4). */
int32_t vrawtex_encode_rgba16(const uint16_t *rgba,
                              size_t rgba_len,
                              uint32_t width,
                              uint32_t height,
                              uint32_t pixel_format,
                              uint32_t profile,
                              VrawtexBuffer *output);

#ifdef __cplusplus
}
#endif

#endif
