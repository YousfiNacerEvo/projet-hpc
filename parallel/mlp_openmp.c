#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double sigmoid(double x) {
    if (x >= 0.0) {
        double z = exp(-x);
        return 1.0 / (1.0 + z);
    }
    double z = exp(x);
    return z / (1.0 + z);
}

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

double relu_deriv(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double bce_loss(double y_pred, double y_true) {
    const double eps = 1e-12;
    double p = y_pred;
    if (p < eps) p = eps;
    if (p > 1.0 - eps) p = 1.0 - eps;
    return -(y_true * log(p) + (1.0 - y_true) * log(1.0 - p));
}

double forward_single(
    double* x,
    double* W1, double* b1,
    double* W2, double* b2,
    double* W3, double* b3,
    int n_input, int n_h1, int n_h2,
    double* h1_out,
    double* h2_out
) {
    for (int i = 0; i < n_h1; i++) {
        double z = b1[i];
        double* w_row = W1 + i * n_input;
        for (int j = 0; j < n_input; j++) {
            z += w_row[j] * x[j];
        }
        h1_out[i] = z;
    }

    for (int i = 0; i < n_h2; i++) {
        double z = b2[i];
        double* w_row = W2 + i * n_h1;
        for (int j = 0; j < n_h1; j++) {
            z += w_row[j] * relu(h1_out[j]);
        }
        h2_out[i] = z;
    }

    double z3 = b3[0];
    for (int j = 0; j < n_h2; j++) {
        z3 += W3[j] * relu(h2_out[j]);
    }
    return sigmoid(z3);
}

void backward_single(
    double* x,
    double* h1_out, double* h2_out,
    double y_pred, double y_true,
    double* W2, double* W3,
    int n_input, int n_h1, int n_h2,
    double* dW1, double* db1,
    double* dW2, double* db2,
    double* dW3, double* db3
) {
    double delta3 = y_pred - y_true;

    for (int j = 0; j < n_h2; j++) {
        dW3[j] += delta3 * relu(h2_out[j]);
    }
    db3[0] += delta3;

    double delta2[n_h2];
    for (int j = 0; j < n_h2; j++) {
        delta2[j] = (W3[j] * delta3) * relu_deriv(h2_out[j]);
        db2[j] += delta2[j];
        for (int k = 0; k < n_h1; k++) {
            dW2[j * n_h1 + k] += delta2[j] * relu(h1_out[k]);
        }
    }

    double delta1[n_h1];
    for (int k = 0; k < n_h1; k++) {
        double s = 0.0;
        for (int j = 0; j < n_h2; j++) {
            s += W2[j * n_h1 + k] * delta2[j];
        }
        delta1[k] = s * relu_deriv(h1_out[k]);
        db1[k] += delta1[k];
        for (int m = 0; m < n_input; m++) {
            dW1[k * n_input + m] += delta1[k] * x[m];
        }
    }
}

void parallel_minibatch_grad(
    double* X_batch,
    double* y_batch,
    int batch_size,
    int n_input, int n_h1, int n_h2,
    double* W1, double* b1,
    double* W2, double* b2,
    double* W3, double* b3,
    double* dW1, double* db1,
    double* dW2, double* db2,
    double* dW3, double* db3,
    int n_threads
) {
    int used_threads = n_threads > 0 ? n_threads : omp_get_max_threads();

    double** loc_dW1 = (double**)calloc(used_threads, sizeof(double*));
    double** loc_db1 = (double**)calloc(used_threads, sizeof(double*));
    double** loc_dW2 = (double**)calloc(used_threads, sizeof(double*));
    double** loc_db2 = (double**)calloc(used_threads, sizeof(double*));
    double** loc_dW3 = (double**)calloc(used_threads, sizeof(double*));
    double** loc_db3 = (double**)calloc(used_threads, sizeof(double*));

    for (int t = 0; t < used_threads; t++) {
        loc_dW1[t] = (double*)calloc((size_t)n_h1 * n_input, sizeof(double));
        loc_db1[t] = (double*)calloc(n_h1, sizeof(double));
        loc_dW2[t] = (double*)calloc((size_t)n_h2 * n_h1, sizeof(double));
        loc_db2[t] = (double*)calloc(n_h2, sizeof(double));
        loc_dW3[t] = (double*)calloc(n_h2, sizeof(double));
        loc_db3[t] = (double*)calloc(1, sizeof(double));
    }

#pragma omp parallel for num_threads(used_threads) schedule(static)
    for (int i = 0; i < batch_size; i++) {
        int tid = omp_get_thread_num();
        double h1[n_h1];
        double h2[n_h2];
        double* xi = X_batch + (size_t)i * n_input;

        double y_pred = forward_single(
            xi, W1, b1, W2, b2, W3, b3,
            n_input, n_h1, n_h2, h1, h2
        );

        backward_single(
            xi, h1, h2, y_pred, y_batch[i],
            W2, W3, n_input, n_h1, n_h2,
            loc_dW1[tid], loc_db1[tid],
            loc_dW2[tid], loc_db2[tid],
            loc_dW3[tid], loc_db3[tid]
        );
    }

    for (int t = 0; t < used_threads; t++) {
        for (int j = 0; j < n_h1 * n_input; j++) dW1[j] += loc_dW1[t][j];
        for (int j = 0; j < n_h1; j++) db1[j] += loc_db1[t][j];
        for (int j = 0; j < n_h2 * n_h1; j++) dW2[j] += loc_dW2[t][j];
        for (int j = 0; j < n_h2; j++) db2[j] += loc_db2[t][j];
        for (int j = 0; j < n_h2; j++) dW3[j] += loc_dW3[t][j];
        db3[0] += loc_db3[t][0];
    }

    double inv_bs = 1.0 / (double)batch_size;
    for (int j = 0; j < n_h1 * n_input; j++) dW1[j] *= inv_bs;
    for (int j = 0; j < n_h1; j++) db1[j] *= inv_bs;
    for (int j = 0; j < n_h2 * n_h1; j++) dW2[j] *= inv_bs;
    for (int j = 0; j < n_h2; j++) db2[j] *= inv_bs;
    for (int j = 0; j < n_h2; j++) dW3[j] *= inv_bs;
    db3[0] *= inv_bs;

    for (int t = 0; t < used_threads; t++) {
        free(loc_dW1[t]);
        free(loc_db1[t]);
        free(loc_dW2[t]);
        free(loc_db2[t]);
        free(loc_dW3[t]);
        free(loc_db3[t]);
    }
    free(loc_dW1);
    free(loc_db1);
    free(loc_dW2);
    free(loc_db2);
    free(loc_dW3);
    free(loc_db3);
}

void sgd_update(
    double* W, double* dW, int size_W,
    double* b, double* db, int size_b,
    double lr
) {
    for (int i = 0; i < size_W; i++) {
        W[i] -= lr * dW[i];
    }
    for (int i = 0; i < size_b; i++) {
        b[i] -= lr * db[i];
    }
}

void train_parallel(
    double* X_train, double* y_train,
    int n_samples, int n_input,
    double* W1, double* b1,
    double* W2, double* b2,
    double* W3, double* b3,
    int n_h1, int n_h2,
    int n_epochs, int batch_size, double lr,
    int n_threads
) {
    double* dW1 = (double*)calloc((size_t)n_h1 * n_input, sizeof(double));
    double* db1 = (double*)calloc(n_h1, sizeof(double));
    double* dW2 = (double*)calloc((size_t)n_h2 * n_h1, sizeof(double));
    double* db2 = (double*)calloc(n_h2, sizeof(double));
    double* dW3 = (double*)calloc(n_h2, sizeof(double));
    double* db3 = (double*)calloc(1, sizeof(double));
    double* h1_tmp = (double*)malloc((size_t)n_h1 * sizeof(double));
    double* h2_tmp = (double*)malloc((size_t)n_h2 * sizeof(double));

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        for (int start = 0; start < n_samples; start += batch_size) {
            int current_bs = batch_size;
            if (start + current_bs > n_samples) {
                current_bs = n_samples - start;
            }

            memset(dW1, 0, (size_t)n_h1 * n_input * sizeof(double));
            memset(db1, 0, n_h1 * sizeof(double));
            memset(dW2, 0, (size_t)n_h2 * n_h1 * sizeof(double));
            memset(db2, 0, n_h2 * sizeof(double));
            memset(dW3, 0, n_h2 * sizeof(double));
            memset(db3, 0, sizeof(double));

            parallel_minibatch_grad(
                X_train + (size_t)start * n_input,
                y_train + start,
                current_bs,
                n_input, n_h1, n_h2,
                W1, b1, W2, b2, W3, b3,
                dW1, db1, dW2, db2, dW3, db3,
                n_threads
            );

            sgd_update(W1, dW1, n_h1 * n_input, b1, db1, n_h1, lr);
            sgd_update(W2, dW2, n_h2 * n_h1, b2, db2, n_h2, lr);
            sgd_update(W3, dW3, n_h2, b3, db3, 1, lr);
        }

        double epoch_loss = 0.0;
        for (int i = 0; i < n_samples; i++) {
            double* xi = X_train + (size_t)i * n_input;
            double p = forward_single(
                xi, W1, b1, W2, b2, W3, b3,
                n_input, n_h1, n_h2, h1_tmp, h2_tmp
            );
            epoch_loss += bce_loss(p, y_train[i]);
        }
        epoch_loss /= (double)n_samples;
        printf("Epoch %d/%d - loss: %.6f\n", epoch + 1, n_epochs, epoch_loss);
    }

    free(dW1);
    free(db1);
    free(dW2);
    free(db2);
    free(dW3);
    free(db3);
    free(h1_tmp);
    free(h2_tmp);
}

void predict(
    double* X_test, int n_test, int n_input,
    double* W1, double* b1,
    double* W2, double* b2,
    double* W3, double* b3,
    int n_h1, int n_h2,
    double* y_pred
) {
    double* h1 = (double*)malloc((size_t)n_h1 * sizeof(double));
    double* h2 = (double*)malloc((size_t)n_h2 * sizeof(double));

    for (int i = 0; i < n_test; i++) {
        double* xi = X_test + (size_t)i * n_input;
        y_pred[i] = forward_single(
            xi, W1, b1, W2, b2, W3, b3,
            n_input, n_h1, n_h2, h1, h2
        );
    }

    free(h1);
    free(h2);
}
