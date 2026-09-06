#include <stdio.h>
#include <stdlib.h>

#include "vrawtex.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s OUTPUT.vrawtex\n", argv[0]);
        return 64;
    }

    const uint16_t rgba[] = {
        1, 255, 256, 65535,
        4096, 8192, 16384, 32768,
    };
    VrawtexBuffer output = {0};
    int32_t status = vrawtex_encode_rgba16(
        rgba,
        sizeof(rgba) / sizeof(rgba[0]),
        2,
        1,
        VRAWTEX_RGBA16,
        VRAWTEX_FAST,
        &output);
    if (status != 0) {
        fprintf(stderr, "vrawtex: %s\n", vrawtex_last_error_message());
        return status;
    }

    FILE *file = fopen(argv[1], "wb");
    if (file == NULL || fwrite(output.data, 1, output.len, file) != output.len) {
        fprintf(stderr, "could not write %s\n", argv[1]);
        if (file != NULL) {
            fclose(file);
        }
        vrawtex_buffer_free(output);
        return 74;
    }
    fclose(file);
    vrawtex_buffer_free(output);
    return 0;
}
