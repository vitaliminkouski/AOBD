import argparse
import os

import numpy as np

N_SAMPLES = 10000
N_FEATURES = 50
N_CLUSTERS = 5


def random_rotation(rng, dim):
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    
    stds = 0.5 + 9.5 * np.exp(-np.arange(N_FEATURES) / 3.0)

    
    rotations = [random_rotation(rng, N_FEATURES) for _ in range(N_CLUSTERS)]
    centers = rng.normal(scale=8.0, size=(N_CLUSTERS, N_FEATURES))

    labels = rng.integers(0, N_CLUSTERS, size=N_SAMPLES)
    x = np.empty((N_SAMPLES, N_FEATURES))
    for k in range(N_CLUSTERS):
        idx = np.where(labels == k)[0]
        z = rng.standard_normal((len(idx), N_FEATURES)) * stds
        x[idx] = centers[k] + z @ rotations[k].T

    np.savetxt(os.path.join(args.out, "dataset.csv"), x, delimiter=",", fmt="%.6f")
    np.savetxt(os.path.join(args.out, "labels.csv"), labels, fmt="%d")

    print(f"dataset.csv: {x.shape[0]} строк x {x.shape[1]} признаков, кластеров: {N_CLUSTERS}")
    print("размеры кластеров:", np.bincount(labels).tolist())
    print(f"std по осям кластера: длинная {stds[0]:.1f}, короткая {stds[-1]:.1f} "
          f"(вытянутость {stds[0] / stds[-1]:.0f}:1)")


if __name__ == "__main__":
    main()
