#ifndef NN_H
#define NN_H

#include "stddef.h"
#include "stdio.h"





#ifndef NN_MALLOC
#include "stdlib.h"
#define NN_MALLOC malloc
#endif // NN_MALLOC





#ifndef NN_ASSERT
#include "assert.h"
#include "math.h"
#define NN_ASSERT assert
#endif // NN_ASSERT


// this is goint o be an array of floats pointed by *es
// the shape of the matrix is define by rows and cols
typedef struct {
    size_t rows;
    size_t cols;
    size_t stride; // this is for cutting matrixes by cols
    float *elems;
} Matrix;

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_rand(Matrix m, float low, float high);
void matrix_fill(Matrix m, float value);
void matrix_copy(Matrix dst, Matrix src);
Matrix matrix_row(Matrix m, size_t row);
void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void matrix_print(Matrix m, const char *name, int padding);
void matrix_sig(Matrix m);

float rand_float();
float sigmoid(float x);

#define MAT_PRINT(m) matrix_print(m, #m, 0)
#define MAT_AT(m, i, j) (m).elems[(i)*(m).stride + (j)]
#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])

// this will be the Neural_Network structure
typedef struct {
    size_t count;
    Matrix *ws;
    Matrix *bs;
    Matrix *as; // this must be count + 1
} NN;

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_print(NN nn, const char * name);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Matrix input, Matrix output);
void nn_finite_diff(NN nn, NN g, float eps, Matrix input, Matrix output);
void nn_backprop(NN nn, NN g, Matrix input, Matrix output);
void nn_learn(NN nn, NN g, float lr);
void nn_fill(NN nn, float value);

#define NN_PRINT(nn) nn_print(nn, #nn)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
#endif // NN_H








#ifdef NN_IMPLEMENTATION
Matrix matrix_alloc(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.elems = NN_MALLOC(sizeof(*m.elems)*rows*cols);
    NN_ASSERT(m.elems != NULL);
    for(size_t i = 0; i < (rows * cols); ++i) {
	m.elems[i] = 0;
    }
    return m;
}
float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

void matrix_fill(Matrix m, float value) {
    for (size_t i = 0; i < (m.rows * m.cols); ++i) {
	m.elems[i] = value;
    }
}

void matrix_sig(Matrix m) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoid(MAT_AT(m, i, j));
        }
    }
}

void matrix_rand(Matrix m, float low, float high) {
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high-low) + low;
        }
    }
}

Matrix matrix_row(Matrix m, size_t row) {
    return (Matrix) {
	.rows = 1,
	.cols = m.cols,
	.stride = m.stride,
	.elems = &MAT_AT(m, row, 0),
    };
}

void matrix_copy(Matrix dst, Matrix src) {
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    for (size_t i = 0; i < dst.rows; ++i) {
	for (size_t j = 0; j < dst.cols; ++j) {
	    MAT_AT(dst, i, j) = MAT_AT(src, i, j);
	}
    }
}

void matrix_dot(Matrix dst, Matrix a, Matrix b) {
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    size_t n = a.cols;

    for (size_t i = 0; i < dst.rows; ++i) {
	for (size_t j = 0; j < dst.cols; ++j) {
	    MAT_AT(dst, i, j) = 0;
	    for (size_t k = 0; k < n; ++k) {
		MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
	    }
	}
    }
}

void matrix_sum(Matrix dst, Matrix a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for (size_t i = 0; i < dst.rows; ++i) {
	for (size_t j = 0; j < dst.cols; ++j) {
	    MAT_AT(dst, i, j) += MAT_AT(a, i, j);
	}
    }
}

void matrix_print(Matrix m, const char *name, int padding) {
    printf("%*s %s = [\n", (int) padding, "", name);
    for (size_t i = 0; i < m.rows; ++i) {
	printf("%*s", (int) padding, "");
	for (size_t j = 0; j < m.cols; ++j) {
	    printf("  %f", MAT_AT(m, i, j));
	}
	printf("\n");
    }
    printf("%*s]\n", (int) padding, "");
}







// NN from here
NN nn_alloc(size_t *arch, size_t arch_count) {
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * (nn.count));
    NN_ASSERT(nn.ws != NULL);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * (nn.count));
    NN_ASSERT(nn.bs != NULL);
    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
    NN_ASSERT(nn.as != NULL);

    nn.as[0] = matrix_alloc(1, arch[0]);
    for (size_t i = 1; i < arch_count; ++i) {
	nn.ws[i - 1] = matrix_alloc(nn.as[i - 1].cols, arch[i]);
	nn.bs[i - 1] = matrix_alloc(1, arch[i]);
	nn.as[i]     = matrix_alloc(1, arch[i]);
    }

    return nn;
}

void nn_print(NN nn, const char * name) {
    char buf[256];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; ++i) {
	snprintf(buf, sizeof(buf), "ws%zu", i);
	matrix_print(nn.ws[i], buf, 4);
	snprintf(buf, sizeof(buf), "bs%zu", i);
	matrix_print(nn.bs[i], buf, 4);
    }
    printf("]\n");
}

void nn_rand(NN nn, float low, float high) {
    for (size_t i = 0; i < nn.count; ++i) {
	matrix_rand(nn.ws[i], low, high);
	matrix_rand(nn.bs[i], low, high);
    }
}

void nn_fill(NN nn, float value) {
    for (size_t i = 0; i < nn.count; ++i) {
	matrix_fill(nn.ws[i], value);
	matrix_fill(nn.bs[i], value);
	matrix_fill(nn.as[i], value);
    }
    matrix_fill(nn.as[nn.count], value);
}

void nn_forward(NN nn) {
    for (size_t i = 0; i < nn.count; ++i) {
	matrix_dot(nn.as[i + 1], nn.as[i], nn.ws[i]); 
	matrix_sum(nn.as[i + 1], nn.bs[i]); 
	matrix_sig(nn.as[i + 1]); 
    }
}

float nn_cost(NN nn, Matrix t_input, Matrix t_output) {
    NN_ASSERT(t_input.rows == t_output.rows);
    NN_ASSERT(t_output.cols = NN_OUTPUT(nn).cols);
    size_t n = t_input.rows;

    float c = 0;
    for (size_t i = 0; i < n; ++i) {
	Matrix x = matrix_row(t_input, i);
	Matrix y = matrix_row(t_output, i);

	matrix_copy(NN_INPUT(nn), x);
	nn_forward(nn);

	size_t q = t_output.cols;
	for(size_t j = 0; j < q; ++j) {
	    float d = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j);
	    c += d*d;
	}
    }

    // here we return the average squared error
    return c/n;
}


void nn_finite_diff(NN nn, NN g, float eps, Matrix input, Matrix output) {
    float saved;
    float c = nn_cost(nn, input, output);

    for (size_t i = 0; i < nn.count; ++i) {
	for (size_t j = 0; j < nn.ws[i].rows; ++j) {
	    for (size_t k = 0; k < nn.ws[k].cols; ++k) {
		saved = MAT_AT(nn.ws[i], j, k);
		MAT_AT(nn.ws[i], j, k) += eps;
		MAT_AT(g.ws[i], j, k) = (nn_cost(nn, input, output) - c) / eps;
		MAT_AT(nn.ws[i], j, k) = saved;
	    }
	}
	for (size_t j = 0; j < nn.bs[i].rows; ++j) {
	    for (size_t k = 0; k < nn.bs[k].cols; ++k) {
		saved = MAT_AT(nn.bs[i], j, k);
		MAT_AT(nn.bs[i], j, k) += eps;
		MAT_AT(g.bs[i], j, k) = (nn_cost(nn, input, output) - c) / eps;
		MAT_AT(nn.bs[i], j, k) = saved;
	    }
	}
    }
}

void nn_backprop(NN nn, NN g, Matrix t_input, Matrix t_output) {
    NN_ASSERT(t_input.rows == t_output.rows);
    NN_ASSERT(t_output.cols = NN_OUTPUT(nn).cols);
    size_t n = t_input.rows;

    nn_fill(g, 0);

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i) {
	matrix_copy(NN_INPUT(nn), matrix_row(t_input, i));
        nn_forward(nn);

	for (size_t j = 0; j <= nn.count; ++j) {
            matrix_fill(g.as[j], 0);
        }

	// we are saving the diff on the output of the gradient descent
	// (the output is the last activation layer)
	for (size_t j = 0; j < t_output.cols; ++j) {
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(t_output, i, j);
        }

	// loop throw each activation layer from the back of the g NN
	for (size_t l = nn.count; l > 0; --l) {
	    size_t current_row_length = nn.as[l].cols;
	    // loop throw the current activation layer and update each row
	    for (size_t j = 0; j < current_row_length; ++j) {
		float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l-1], 0, j) += 2*da*a*(1 - a);
                for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
		    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l-1], k, j) += 2*da*a*(1 - a)*pa;
                    MAT_AT(g.as[l-1], 0, k) += 2*da*a*(1 - a)*w;
		}
	    }
	}
    }


    for (size_t i = 0; i < g.count; ++i) {
        for (size_t j = 0; j < g.ws[i].rows; ++j) {
            for (size_t k = 0; k < g.ws[i].cols; ++k) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for (size_t j = 0; j < g.bs[i].rows; ++j) {
            for (size_t k = 0; k < g.bs[i].cols; ++k) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}

void nn_learn(NN nn, NN g, float lr) {
    for (size_t i = 0; i < nn.count; ++i) {
	for (size_t j = 0; j < nn.ws[i].rows; ++j) {
	    for (size_t k = 0; k < nn.ws[i].cols; ++k) {
		MAT_AT(nn.ws[i], j, k) -= lr*MAT_AT(g.ws[i], j, k);
	    }
	}

	for (size_t j = 0; j < nn.bs[i].rows; ++j) {
	    for (size_t k = 0; k < nn.bs[i].cols; ++k) {
		MAT_AT(nn.bs[i], j, k) -= lr*MAT_AT(g.bs[i], j, k);
	    }
	}
    }    
}

#endif // NN_IMPLEMENTATION
