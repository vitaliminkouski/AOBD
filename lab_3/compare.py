import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K = 10
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def align_signs(components, reference):
    """Знак собственного вектора произвольный: приводим к знаку эталона."""
    signs = np.sign(np.sum(components * reference, axis=0))
    return components * signs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/workspace/data")
    parser.add_argument("--out", default="/workspace/out")
    args = parser.parse_args()

    x = np.loadtxt(f"{args.data}/dataset.csv", delimiter=",")
    labels = np.loadtxt(f"{args.data}/labels.csv", dtype=int)
    xc = x - x.mean(axis=0)
    n = xc.shape[0]

    _, s_ref_all, vt_ref = np.linalg.svd(xc, full_matrices=False)
    v_ref = vt_ref.T[:, :K]
    ev_ref = s_ref_all**2 / np.sum(s_ref_all**2)

    spark = np.load(f"{args.out}/spark.npz")
    tf = np.load(f"{args.out}/tf.npz")
    runs = {
        "Spark PCA": (spark["pca_proj"], spark["pca_components"]),
        "Spark SVD": (spark["svd_proj"], spark["svd_v"]),
        "TF PCA": (tf["pca_proj"], tf["pca_components"]),
        "TF SVD": (tf["svd_proj"], tf["svd_v"]),
    }

    print("=" * 70)
    print("Проверка корректности (эталон: numpy.linalg.svd на центрированных данных)")
    print("=" * 70)
    print(f"{'метод':<11}{'min |cos| компонент':>22}{'max |разн. проекций|':>24}")
    aligned = {}
    for name, (proj, comps) in runs.items():
        comps = align_signs(comps, v_ref)
        proj_ref = xc @ v_ref
        cos = np.abs(np.sum(comps * v_ref, axis=0))
        proj = proj * np.sign(np.sum(proj * proj_ref, axis=0))
        aligned[name] = proj
        print(f"{name:<11}{cos.min():>22.6f}{np.abs(proj - proj_ref).max():>24.2e}")

    s_ref = s_ref_all[:K]
    print("\nСингулярные числа (первые 10):")
    print(f"{'numpy':<11}", np.round(s_ref, 2).tolist())
    print(f"{'Spark SVD':<11}", np.round(spark["svd_s"], 2).tolist())
    print(f"{'TF SVD':<11}", np.round(tf["svd_s"], 2).tolist())
    print(f"макс. отн. расхождение s: Spark {np.max(np.abs(spark['svd_s'] / s_ref - 1)):.2e}, "
          f"TF {np.max(np.abs(tf['svd_s'] / s_ref - 1)):.2e}")

    print("\nДоля объяснённой дисперсии (первые 10 компонент):")
    print(f"{'numpy':<11}", np.round(ev_ref[:K], 4).tolist(), f"сумма = {ev_ref[:K].sum():.4f}")
    print(f"{'Spark PCA':<11}", np.round(spark["pca_ev"], 4).tolist(), f"сумма = {spark['pca_ev'].sum():.4f}")
    print(f"{'TF PCA':<11}", np.round(tf["pca_ev"], 4).tolist(), f"сумма = {tf['pca_ev'].sum():.4f}")

    with open(f"{args.out}/spark_times.json") as f:
        t_spark = json.load(f)
    with open(f"{args.out}/tf_times.json") as f:
        t_tf = json.load(f)

    print("\n" + "=" * 70)
    print(f"Скорость (fit = вычисление {K} компонент, transform = проекция {n} строк)")
    print("=" * 70)
    print(f"{'метод':<6}{'движок':<12}{'fit, мс (среднее ± std)':<26}{'transform, мс':>14}{'итого, мс':>12}")
    stats = {}
    for method in ("pca", "svd"):
        for engine, t in (("Spark", t_spark), ("TensorFlow", t_tf)):
            fit = np.array(t[method]["fit"]) * 1000
            tr = t[method]["transform"] * 1000
            stats[(method, engine)] = (fit.mean(), fit.std(), tr)
            print(f"{method.upper():<6}{engine:<12}{f'{fit.mean():.2f} ± {fit.std():.2f}':<26}"
                  f"{tr:>14.2f}{fit.mean() + tr:>12.2f}")
    print()
    for method in ("pca", "svd"):
        s_fit, tf_fit = stats[(method, "Spark")][0], stats[(method, "TensorFlow")][0]
        s_tot = s_fit + stats[(method, "Spark")][2]
        tf_tot = tf_fit + stats[(method, "TensorFlow")][2]
        print(f"{method.upper()}: TensorFlow быстрее Spark в {s_fit / tf_fit:.0f} раз по fit, "
              f"в {s_tot / tf_tot:.0f} раз по fit+transform")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, proj) in zip(axes.ravel(), aligned.items()):
        for k in np.unique(labels):
            mask = labels == k
            ax.scatter(proj[mask, 0], proj[mask, 1], s=3, alpha=0.5, color=COLORS[k], label=f"кластер {k}")
        ax.set_title(name)
        ax.set_xlabel("компонента 1")
        ax.set_ylabel("компонента 2")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(markerscale=4, loc="best")
    fig.suptitle("Проекция 50-мерных данных на первые 2 компоненты")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_projections.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(1, K + 1)
    ax.bar(idx - 0.2, spark["pca_ev"], width=0.4, label="Spark PCA")
    ax.bar(idx + 0.2, tf["pca_ev"], width=0.4, label="TensorFlow PCA")
    ax.set_xlabel("номер компоненты")
    ax.set_ylabel("доля дисперсии")
    ax.set_xticks(idx)
    ax2 = ax.twinx()
    ax2.plot(idx, np.cumsum(tf["pca_ev"]), "k.-", label="накопленная доля")
    ax2.set_ylabel("накопленная доля")
    ax2.set_ylim(0, 1)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("Объяснённая дисперсия первых 10 компонент")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    for i, engine in enumerate(("Spark", "TensorFlow")):
        vals = [stats[(m, engine)][0] + stats[(m, engine)][2] for m in ("pca", "svd")]
        bars = ax.bar(np.arange(2) + (i - 0.5) * width, vals, width, label=engine)
        ax.bar_label(bars, fmt="%.1f", padding=2)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["PCA", "SVD"])
    ax.set_yscale("log")
    ax.set_ylabel("время fit + transform, мс (лог. шкала)")
    ax.set_title("Скорость: Spark (кластер) vs TensorFlow")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_time.png", dpi=150)
    plt.close(fig)
    print("\nграфики: fig_projections.png, fig_variance.png, fig_time.png")


if __name__ == "__main__":
    main()
