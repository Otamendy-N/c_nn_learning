#include <time.h>
#define NN_IMPLEMENTATION
#include "nn.h"

float training_data[] = {
    0,0,0,
    0,1,1,
    1,0,1,
    1,1,0,
};

int main(void) {
    srand(time(0));

    size_t stride = 3;
    size_t n = 4;
    Matrix input = {
	.rows = n,
	.cols = 2,
	.stride = stride,
	.elems = training_data,
    };
    Matrix output = {
	.rows = n,
	.cols = 1,
	.stride = stride,
	.elems = training_data + 2, // this is a pointer so you can add.
    };

    size_t arch[] = {2, 2, 2, 1};
    NN nn = nn_alloc(arch, ARRAY_LEN(arch));
    NN g = nn_alloc(arch, ARRAY_LEN(arch));
    nn_rand(nn, 0, 1);
    float lr = 1;

    for (size_t i = 0; i < 5000; ++i) {
	nn_backprop(nn, g, input, output);
	nn_learn(nn, g, lr);
	printf("cost = %f\n", nn_cost(nn, input, output));
    }

    NN_PRINT(nn);

    for (size_t i = 0; i < 2; ++i) {
	for (size_t j = 0; j < 2; ++j) {
	    MAT_AT(NN_INPUT(nn), 0, 0) = i;
	    MAT_AT(NN_INPUT(nn), 0, 1) = j;
	    nn_forward(nn);

	    printf("%zu ^ %zu = %f\n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
	}
    }
    return 0;
}




