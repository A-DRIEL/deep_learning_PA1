# scripts/object_size_distribution.py

import json

import numpy as np
import matplotlib.pyplot as plt

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.receptive_field import compute_receptive_field, resunet_encoder_layers


def compute_diameters(dataset):
    diameters = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        mask_np = mask.numpy()
        ids = np.unique(mask_np)
        ids = ids[ids != 0]
        for obj_id in ids:
            area = (mask_np == obj_id).sum()
            diameter = 2 * np.sqrt(area / np.pi)
            diameters.append(diameter)
    return np.array(diameters)


def main():
    dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    diameters = compute_diameters(dataset)

    rf = compute_receptive_field(resunet_encoder_layers())

    pct_above_rf = 100 * (diameters > rf).mean()

    print(f"Campo receptivo teórico: {rf}px")
    print(f"Diâmetro dos objetos - min: {diameters.min():.1f}, "
          f"max: {diameters.max():.1f}, média: {diameters.mean():.1f}, "
          f"mediana: {np.median(diameters):.1f}")
    print(f"Objetos MAIORES que o campo receptivo: {pct_above_rf:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diameters, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(rf, color="red", linestyle="--", linewidth=2,
               label=f"campo receptivo teórico ({rf}px)")
    ax.set_xlabel("Diâmetro equivalente do objeto (px, em imagem 128x128)")
    ax.set_ylabel("Número de núcleos")
    ax.set_title("Distribuição de tamanho dos núcleos vs. campo receptivo")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/receptive_field_vs_object_size.png", dpi=150)
    print("\nFigura salva em outputs/receptive_field_vs_object_size.png")


if __name__ == "__main__":
    main()