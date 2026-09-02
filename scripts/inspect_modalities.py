# scripts/inspect_modalities.py

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def estimate_color_variance(image_path):
    """
    Proxy simples para detectar modalidade: imagens verdadeiramente
    grayscale (fluorescência) têm variância ~0 entre canais R,G,B.
    Imagens coloridas (histologia/brightfield) têm variância alta.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    channel_variance = image.var(axis=2)
    return channel_variance.mean()


def main(threshold=50.0):
    root = Path("data/raw/stage1_train")
    results = []

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue
        image_id = sample_dir.name
        image_path = sample_dir / "images" / f"{image_id}.png"
        variance = estimate_color_variance(image_path)
        results.append(variance)

    variances = np.array(results)
    n_gray = (variances <= threshold).sum()
    n_colored = (variances > threshold).sum()

    print(f"Total de imagens: {len(variances)}")
    print(f"Grayscale (<= {threshold}): {n_gray}")
    print(f"Coloridas (> {threshold}): {n_colored}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(variances, bins=40, color="steelblue", edgecolor="black")
    axes[0].axvline(threshold, color="red", linestyle="--",
                     label=f"threshold = {threshold}")
    axes[0].set_title("Distribuição de variância de cor (escala linear)")
    axes[0].set_xlabel("Variância média entre canais RGB")
    axes[0].set_ylabel("Número de imagens")
    axes[0].legend()

    axes[1].hist(variances, bins=40, color="steelblue", edgecolor="black")
    axes[1].axvline(threshold, color="red", linestyle="--",
                     label=f"threshold = {threshold}")
    axes[1].set_yscale("log")
    axes[1].set_title("Mesma distribuição (escala log no eixo Y)")
    axes[1].set_xlabel("Variância média entre canais RGB")
    axes[1].set_ylabel("Número de imagens (log)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("modality_histogram.png", dpi=120)
    print("Figura salva em modality_histogram.png")


if __name__ == "__main__":
    main()