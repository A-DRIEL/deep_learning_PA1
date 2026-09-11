# scripts/plot_density_vs_performance.py

import json

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import apply_clean_style


def main():
    with open("outputs/instance_map_results.json") as f:
        data = json.load(f)

    per_image = data["per_image"]
    densities = np.array([r["n_gt_instances"] for r in per_image])
    maps = np.array([r["mAP"] for r in per_image])
    errors = np.array([r["count_error"] for r in per_image])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- mAP vs densidade ---
    axes[0].scatter(densities, maps, alpha=0.6, color="#4C72B0",
                     edgecolors="white", linewidths=0.5, s=50)
    z = np.polyfit(densities, maps, 1)
    trend = np.poly1d(z)
    x_line = np.linspace(densities.min(), densities.max(), 100)
    axes[0].plot(x_line, trend(x_line), color="#C44E52", linestyle="--",
                 linewidth=2, label=f"tendência (slope={z[0]:.4f})")
    axes[0].set_xlabel("Densidade (nº de instâncias reais)")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("mAP vs. densidade de objetos", fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    apply_clean_style(axes[0])

    # --- erro de contagem vs densidade ---
    axes[1].scatter(densities, errors, alpha=0.6, color="#DD8452",
                     edgecolors="white", linewidths=0.5, s=50)
    z2 = np.polyfit(densities, errors, 1)
    trend2 = np.poly1d(z2)
    axes[1].plot(x_line, trend2(x_line), color="#C44E52", linestyle="--",
                 linewidth=2, label=f"tendência (slope={z2[0]:.4f})")
    axes[1].set_xlabel("Densidade (nº de instâncias reais)")
    axes[1].set_ylabel("Erro absoluto de contagem")
    axes[1].set_title("Erro de contagem vs. densidade de objetos", fontsize=12, fontweight="bold")
    axes[1].legend(frameon=False)
    apply_clean_style(axes[1])

    plt.tight_layout()
    plt.savefig("./outputs/density_vs_performance.png", dpi=150, bbox_inches="tight")
    print("Figura salva em outputs/density_vs_performance.png")

    corr_map = np.corrcoef(densities, maps)[0, 1]
    corr_err = np.corrcoef(densities, errors)[0, 1]
    print(f"Correlação densidade vs mAP: {corr_map:.3f}")
    print(f"Correlação densidade vs erro: {corr_err:.3f}")


if __name__ == "__main__":
    main()