#include <inttypes.h>
#define NN_IMPLEMENTATION
#include "nn.h"
#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_WIDTH 800
#define IMG_HEIGHT 600

uint32_t img_pixels[IMG_HEIGHT*IMG_WIDTH];

int main(void) {
    size_t arch[] = {2, 2, 1};

    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN_PRINT(nn);

    uint32_t neuron_color = 0xFF0000ff;
    uint32_t background_color = 0xFF181818;
    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
    olivec_fill(img, background_color);


    size_t layer_count = arch[0];
    int neuron_radius = 30;
    int layer_border_pad = 1;
    int layer_height = img.height - 2*layer_border_pad;
    int layer_vpad = layer_height / layer_count;
    int layer_x = img.width / 2;
    int layer_y = img.height / 2;
    for(size_t i = 0; i < arch[0]; ++i) {
	int cx = layer_x;
	int cy = layer_y + i*layer_vpad + layer_vpad/2;

	olivec_circle(img, cx, cy, neuron_radius, neuron_color);
    }

    const char *img_file_path = "nn.png";
    if (!stbi_write_png(img_file_path, img.width, img.height, 4, img.pixels, img.stride*sizeof(uint32_t))) {
	printf("ERROR: could not save file %s\n", img_file_path);
    }

    printf("SUCCESS: file saved in %s\n", img_file_path);



    return 0;
}



