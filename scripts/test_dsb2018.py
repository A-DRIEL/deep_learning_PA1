# scripts/test_dsb2018.py

import matplotlib.pyplot as plt
import numpy as np

from src.datasets.dsb2018 import DSB2018Dataset


def show_samples(dataset, indices):
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for col, idx in enumerate(indices):
        image, mask = dataset[idx]

        img_np = image.permute(1, 2, 0).numpy()  # (3,H,W) -> (H,W,3) para plot
        mask_np = mask.numpy()
        n_instances = mask_np.max()

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f"Imagem (idx={idx})")
        axes[0, col].axis("off")

        axes[1, col].imshow(mask_np, cmap="nipy_spectral")
        axes[1, col].set_title(f"Máscara ({n_instances} instâncias)")
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("dsb2018_samples.png", dpi=120)
    print("Figura salva em dsb2018_samples.png")


if __name__ == "__main__":
    dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    print(f"Total de imagens: {len(dataset)}")

    indices = [0, 100, 300, 500]
    show_samples(dataset, indices)
