import argparse
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

K = 10


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def pca_fit(x_centered):
    """PCA: собственные векторы ковариационной матрицы."""
    n = tf.cast(tf.shape(x_centered)[0], x_centered.dtype)
    cov = tf.matmul(x_centered, x_centered, transpose_a=True) / (n - 1.0)
    eigvals, eigvecs = tf.linalg.eigh(cov)  
    eigvals = tf.reverse(eigvals, axis=[0])[:K]
    components = tf.reverse(eigvecs, axis=[1])[:, :K]
    return components, eigvals


def svd_fit(x_centered):
    """SVD: x = U * diag(s) * V^T, берём первые K сингулярных троек."""
    s, u, v = tf.linalg.svd(x_centered, full_matrices=False)  
    return v[:, :K], s[:K]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/workspace/data/dataset.csv")
    parser.add_argument("--out", default="/workspace/out")
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    print("TensorFlow:", tf.__version__, "| устройства:", [d.device_type for d in tf.config.list_logical_devices()])

    x = np.loadtxt(args.data, delimiter=",")
    n_rows = x.shape[0]
    print(f"строк: {n_rows}, признаков: {x.shape[1]}")

    x = tf.constant(x, dtype=tf.float64)
    x_centered = x - tf.reduce_mean(x, axis=0)

    pca_fit(x_centered)  
    pca_times = []
    for _ in range(args.repeats):
        (components, eigvals), t = timed(lambda: pca_fit(x_centered))
        pca_times.append(t)
    pca_proj, pca_transform = timed(lambda: tf.matmul(x_centered, components).numpy())
    total_var = float(tf.reduce_sum(tf.square(x_centered))) / (n_rows - 1)  
    pca_ev = eigvals.numpy() / total_var
    print(f"PCA: fit (среднее из {args.repeats}) = {np.mean(pca_times) * 1000:.2f} мс, "
          f"transform = {pca_transform * 1000:.2f} мс")

    svd_fit(x_centered) 
    svd_times = []
    for _ in range(args.repeats):
        (v, s), t = timed(lambda: svd_fit(x_centered))
        svd_times.append(t)
    svd_proj, svd_transform = timed(lambda: tf.matmul(x_centered, v).numpy())
    print(f"SVD: fit (среднее из {args.repeats}) = {np.mean(svd_times) * 1000:.2f} мс, "
          f"transform = {svd_transform * 1000:.2f} мс")

    np.savez(f"{args.out}/tf.npz",
             pca_proj=pca_proj, pca_components=components.numpy(), pca_ev=pca_ev,
             svd_proj=svd_proj, svd_v=v.numpy(), svd_s=s.numpy())
    result = {
        "engine": "tensorflow",
        "version": tf.__version__,
        "rows": n_rows,
        "pca": {"fit": pca_times, "transform": pca_transform},
        "svd": {"fit": svd_times, "transform": svd_transform},
    }
    with open(f"{args.out}/tf_times.json", "w") as f:
        json.dump(result, f, indent=2)

    print("PCA explained variance (10 компонент):", np.round(pca_ev, 4).tolist())
    print("SVD сингулярные числа (10 компонент):", np.round(s.numpy(), 2).tolist())


if __name__ == "__main__":
    main()
