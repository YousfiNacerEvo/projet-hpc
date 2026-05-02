"""
mlp_parallel.py
Orchestrates the parallel MLP training via ctypes -> mlp_openmp.so
Runs benchmark: n_threads in {1, 2, 4, 8}
Outputs: results/benchmark_results.csv + results/speedup_plot.png
"""

import csv
import ctypes
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SO_PATH = THIS_DIR / "mlp_openmp.so"

sys.path.insert(0, str(DATA_DIR))
from preprocess import get_data  # noqa: E402

dbl_p = ctypes.POINTER(ctypes.c_double)


def load_library():
    if not SO_PATH.exists():
        raise FileNotFoundError(
            f"Shared library not found at '{SO_PATH}'. Compile first with compile.sh."
        )
    return ctypes.CDLL(str(SO_PATH))


def set_signatures(lib):
    lib.train_parallel.argtypes = [
        dbl_p,
        dbl_p,
        ctypes.c_int,
        ctypes.c_int,
        dbl_p,
        dbl_p,
        dbl_p,
        dbl_p,
        dbl_p,
        dbl_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
    ]
    lib.train_parallel.restype = None

    lib.predict.argtypes = [
        dbl_p,
        ctypes.c_int,
        ctypes.c_int,
        dbl_p,
        dbl_p,
        dbl_p,
        dbl_p,
        dbl_p,
        dbl_p,
        ctypes.c_int,
        ctypes.c_int,
        dbl_p,
    ]
    lib.predict.restype = None


def xavier(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_out, fan_in))


def init_weights(n_input, n_h1, n_h2, seed=42):
    np.random.seed(seed)
    W1 = xavier(n_input, n_h1)
    b1 = np.zeros(n_h1)
    W2 = xavier(n_h1, n_h2)
    b2 = np.zeros(n_h2)
    W3 = xavier(n_h2, 1)
    b3 = np.zeros(1)
    return (
        np.ascontiguousarray(W1, dtype=np.float64),
        np.ascontiguousarray(b1, dtype=np.float64),
        np.ascontiguousarray(W2, dtype=np.float64),
        np.ascontiguousarray(b2, dtype=np.float64),
        np.ascontiguousarray(W3.reshape(-1), dtype=np.float64),
        np.ascontiguousarray(b3, dtype=np.float64),
    )


def compute_accuracy(y_pred_proba, y_true):
    return np.mean((y_pred_proba >= 0.5).astype(np.int32) == y_true.astype(np.int32))


def tune_threshold(y_pred_proba, y_true):
    thresholds = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    y_true_i = y_true.astype(np.int32)
    best_acc, best_th = -1.0, 0.5
    for th in thresholds:
        acc = np.mean((y_pred_proba >= th).astype(np.int32) == y_true_i)
        if acc > best_acc:
            best_acc = float(acc)
            best_th = float(th)
    return best_th, best_acc


def main():
    lib = load_library()
    set_signatures(lib)

    X_train, X_val, X_test, y_train, y_val, y_test = get_data(return_validation=True)
    n_samples, n_input = X_train.shape
    n_h1, n_h2 = 64, 32
    n_epochs, batch_size, lr = 20, 64, 0.01

    os.makedirs(RESULTS_DIR, exist_ok=True)
    benchmark_rows = []
    baseline_time = None

    X_tr_ptr = X_train.ctypes.data_as(dbl_p)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_tr_ptr = y_train.ctypes.data_as(dbl_p)

    X_val = np.ascontiguousarray(X_val, dtype=np.float64)
    y_val = np.ascontiguousarray(y_val, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)

    for n_threads in [1, 2, 4, 8]:
        print(f"\n--- Training with {n_threads} thread(s) ---")
        W1, b1, W2, b2, W3, b3 = init_weights(n_input, n_h1, n_h2, seed=42)

        t_start = time.perf_counter()
        lib.train_parallel(
            X_tr_ptr,
            y_tr_ptr,
            ctypes.c_int(n_samples),
            ctypes.c_int(n_input),
            W1.ctypes.data_as(dbl_p),
            b1.ctypes.data_as(dbl_p),
            W2.ctypes.data_as(dbl_p),
            b2.ctypes.data_as(dbl_p),
            W3.ctypes.data_as(dbl_p),
            b3.ctypes.data_as(dbl_p),
            ctypes.c_int(n_h1),
            ctypes.c_int(n_h2),
            ctypes.c_int(n_epochs),
            ctypes.c_int(batch_size),
            ctypes.c_double(lr),
            ctypes.c_int(n_threads),
        )
        elapsed = time.perf_counter() - t_start

        y_pred = np.zeros(len(y_test), dtype=np.float64)
        lib.predict(
            X_test.ctypes.data_as(dbl_p),
            ctypes.c_int(len(y_test)),
            ctypes.c_int(n_input),
            W1.ctypes.data_as(dbl_p),
            b1.ctypes.data_as(dbl_p),
            W2.ctypes.data_as(dbl_p),
            b2.ctypes.data_as(dbl_p),
            W3.ctypes.data_as(dbl_p),
            b3.ctypes.data_as(dbl_p),
            ctypes.c_int(n_h1),
            ctypes.c_int(n_h2),
            y_pred.ctypes.data_as(dbl_p),
        )

        y_val_pred = np.zeros(len(y_val), dtype=np.float64)
        lib.predict(
            X_val.ctypes.data_as(dbl_p),
            ctypes.c_int(len(y_val)),
            ctypes.c_int(n_input),
            W1.ctypes.data_as(dbl_p),
            b1.ctypes.data_as(dbl_p),
            W2.ctypes.data_as(dbl_p),
            b2.ctypes.data_as(dbl_p),
            W3.ctypes.data_as(dbl_p),
            b3.ctypes.data_as(dbl_p),
            ctypes.c_int(n_h1),
            ctypes.c_int(n_h2),
            y_val_pred.ctypes.data_as(dbl_p),
        )

        best_threshold, val_acc = tune_threshold(y_val_pred, y_val)
        raw_acc = compute_accuracy(y_pred, y_test)
        tuned_acc = np.mean((y_pred >= best_threshold).astype(np.int32) == y_test.astype(np.int32))

        if baseline_time is None:
            baseline_time = elapsed
        speedup = baseline_time / elapsed

        print(
            f"  Time: {elapsed:.2f}s | Speedup: {speedup:.2f}x | "
            f" Accuracy: {val_acc:.4f}"
            
        )
        benchmark_rows.append(
            {
                "n_threads": n_threads,
                "total_time_s": round(elapsed, 3),
                "time_per_epoch_s": round(elapsed / n_epochs, 3),
                "speedup": round(speedup, 3),
                "threshold_best_on_val": round(float(best_threshold), 4),
                "val_accuracy": round(float(val_acc), 4),
                "test_accuracy_at_0_5": round(float(raw_acc), 4),
                "test_accuracy_tuned": round(float(tuned_acc), 4),
            }
        )

    csv_path = RESULTS_DIR / "benchmark_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_rows[0].keys())
        writer.writeheader()
        writer.writerows(benchmark_rows)
    print(f"\nResults saved to {csv_path}")

    threads = [r["n_threads"] for r in benchmark_rows]
    speedups = [r["speedup"] for r in benchmark_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(threads, speedups, marker="o", linewidth=2, color="steelblue", label="Measured speedup")
    plt.plot(threads, threads, linestyle="--", color="gray", label="Ideal linear speedup")
    plt.xlabel("Number of threads")
    plt.ylabel("Speedup")
    plt.title("MLP Mini-Batch SGD - Parallel Speedup (Adult Income)")
    plt.legend()
    plt.xticks(threads)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "speedup_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Speedup plot saved to {plot_path}")


if __name__ == "__main__":
    main()
