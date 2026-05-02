"""
GPU bonus implementation (CUDA) using CuPy.

If CUDA driver/runtime is incompatible, script falls back to CPU (NumPy)
instead of crashing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(DATA_DIR))

from preprocess import get_data  # noqa: E402


def get_backend():
    """
    Returns (xp, to_cpu, backend_name, device_message).
    xp is cupy or numpy; to_cpu converts arrays to numpy arrays.
    """
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return np, np.asarray, "CPU", "CuPy not installed"

    try:
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        return cp, cp.asnumpy, "CUDA", device_name
    except Exception as exc:
        return np, np.asarray, "CPU", f"CUDA unavailable ({exc})"


def relu(xp, x):
    return xp.maximum(x, 0.0)


def relu_grad(xp, x):
    return (x > 0.0).astype(xp.float32)


def sigmoid(xp, x):
    return 1.0 / (1.0 + xp.exp(-x))


def bce_loss(xp, y_pred, y_true):
    eps = 1e-7
    p = xp.clip(y_pred, eps, 1.0 - eps)
    return -xp.mean(y_true * xp.log(p) + (1.0 - y_true) * xp.log(1.0 - p))


def xavier(xp, shape, rng):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return xp.asarray(rng.uniform(-limit, limit, size=shape), dtype=xp.float32)


def accuracy_with_threshold(y_prob_cpu, y_true_cpu, threshold):
    return float(
        ((y_prob_cpu >= threshold).astype(np.int32) == y_true_cpu.astype(np.int32)).mean()
    )


def best_threshold(y_prob_cpu, y_true_cpu):
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t, best_acc = 0.5, -1.0
    for t in thresholds:
        acc = accuracy_with_threshold(y_prob_cpu, y_true_cpu, t)
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def forward(xp, x, w1, b1, w2, b2, w3, b3):
    z1 = x @ w1 + b1
    a1 = relu(xp, z1)
    z2 = a1 @ w2 + b2
    a2 = relu(xp, z2)
    z3 = a2 @ w3 + b3
    y = sigmoid(xp, z3)
    return z1, a1, z2, a2, y


def to_float(x):
    return float(x.get()) if hasattr(x, "get") else float(x)


def main():
    xp, to_cpu, backend_name, device_message = get_backend()
    print(f"Backend: {backend_name} | {device_message}")

    X_train, X_val, X_test, y_train, y_val, y_test = get_data(return_validation=True)

    X_train_arr = xp.asarray(X_train, dtype=xp.float32)
    X_val_arr = xp.asarray(X_val, dtype=xp.float32)
    X_test_arr = xp.asarray(X_test, dtype=xp.float32)
    y_train_arr = xp.asarray(y_train.reshape(-1, 1), dtype=xp.float32)
    y_val_arr = xp.asarray(y_val.reshape(-1, 1), dtype=xp.float32)
    y_test_cpu = y_test.astype(np.float32)

    n_samples, n_input = X_train_arr.shape
    n_h1, n_h2 = 64, 32
    epochs, batch_size = 20, 128
    lr = 0.01

    rng = np.random.default_rng(42)
    w1 = xavier(xp, (n_input, n_h1), rng)
    b1 = xp.zeros((1, n_h1), dtype=xp.float32)
    w2 = xavier(xp, (n_h1, n_h2), rng)
    b2 = xp.zeros((1, n_h2), dtype=xp.float32)
    w3 = xavier(xp, (n_h2, 1), rng)
    b3 = xp.zeros((1, 1), dtype=xp.float32)

    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        perm = xp.random.permutation(n_samples)
        X_train_arr = X_train_arr[perm]
        y_train_arr = y_train_arr[perm]

        for i in range(0, n_samples, batch_size):
            xb = X_train_arr[i : i + batch_size]
            yb = y_train_arr[i : i + batch_size]

            z1, a1, z2, a2, y_pred = forward(xp, xb, w1, b1, w2, b2, w3, b3)
            bs = xb.shape[0]

            dz3 = (y_pred - yb) / bs
            dw3 = a2.T @ dz3
            db3 = xp.sum(dz3, axis=0, keepdims=True)

            dz2 = (dz3 @ w3.T) * relu_grad(xp, z2)
            dw2 = a1.T @ dz2
            db2 = xp.sum(dz2, axis=0, keepdims=True)

            dz1 = (dz2 @ w2.T) * relu_grad(xp, z1)
            dw1 = xb.T @ dz1
            db1 = xp.sum(dz1, axis=0, keepdims=True)

            w3 -= lr * dw3
            b3 -= lr * db3
            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1

        _, _, _, _, train_prob = forward(xp, X_train_arr, w1, b1, w2, b2, w3, b3)
        loss = to_float(bce_loss(xp, train_prob, y_train_arr))
        print(f"Epoch {epoch:02d}/{epochs} - loss: {loss:.4f}")

    elapsed = time.perf_counter() - start

    _, _, _, _, val_prob = forward(xp, X_val_arr, w1, b1, w2, b2, w3, b3)
    _, _, _, _, test_prob = forward(xp, X_test_arr, w1, b1, w2, b2, w3, b3)

    val_prob_cpu = to_cpu(val_prob).reshape(-1)
    test_prob_cpu = to_cpu(test_prob).reshape(-1)
    y_val_cpu = to_cpu(y_val_arr).reshape(-1)

    th, val_acc = best_threshold(val_prob_cpu, y_val_cpu)
    acc_05 = accuracy_with_threshold(test_prob_cpu, y_test_cpu, 0.5)
    acc_tuned = accuracy_with_threshold(test_prob_cpu, y_test_cpu, th)

    print("\n=== Bonus Results ===")
    print(f"Train time: {elapsed:.2f}s")
    print(f"Test accuracy @0.5: {acc_05:.4f}")
    print(f"Best threshold on val: {th:.3f} (val acc={val_acc:.4f})")
    print(f"Test accuracy tuned: {acc_tuned:.4f}")


if __name__ == "__main__":
    main()
