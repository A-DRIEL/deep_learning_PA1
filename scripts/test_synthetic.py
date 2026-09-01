import matplotlib.pyplot as plt
import numpy as np

from src.datasets.synthetic import SyntheticEllipseDataset


def show_samples(dataset, n=4):
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i in range(n):
        image, mask = dataset[i]

        img_np = image[0].numpy()
        mask_np = mask.numpy()

        n_instances = mask_np.max()

        axes[0, i].imshow(img_np, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Imagem (idx={i})")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask_np, cmap="nipy_spectral")
        axes[1, i].set_title(f"Máscara ({n_instances} instâncias)")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("synthetic_samples.png", dpi=120)
    print("Figura salva em synthetic_samples.png")


if __name__ == "__main__":
    dataset = SyntheticEllipseDataset(
        size=128, min_ellipses=5, max_ellipses=20,
        num_samples=200, seed=42,
    )

    print(f"Tamanho do dataset: {len(dataset)}")

    image, mask = dataset[0]
    print(f"Shape da imagem: {tuple(image.shape)}")
    print(f"Shape da máscara: {tuple(mask.shape)}")
    print(f"Dtype imagem: {image.dtype}, Dtype máscara: {mask.dtype}")
    print(f"Valores únicos na máscara (idx=0): {np.unique(mask.numpy())}")
    
    show_samples(dataset, n=4)