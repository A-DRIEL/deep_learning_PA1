# scripts/effective_receptive_field.py

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet


def compute_effective_receptive_field(model, image, device):
    """
    Mede o campo receptivo EFETIVO via gradiente: qual a influência real
    de cada pixel de entrada sobre a predição no pixel central da saída.

    image: (3, H, W)
    Retorna: mapa de gradiente absoluto (H, W), normalizado.
    """
    model.eval()
    image = image.clone().unsqueeze(0).to(device)
    image.requires_grad_(True)

    logits = model(image)  # (1, 1, H, W) para o baseline binário
    _, _, H, W = logits.shape
    center_y, center_x = H // 2, W // 2

    # zera gradientes anteriores, propaga só a partir do pixel central
    model.zero_grad()
    target = logits[0, 0, center_y, center_x]
    target.backward()

    grad = image.grad[0].abs().mean(dim=0)  # média entre os 3 canais -> (H, W)
    grad_np = grad.cpu().numpy()
    return grad_np / (grad_np.max() + 1e-8)


def estimate_erf_diameter(grad_map, energy_threshold=0.9):
    """
    Estima o "diâmetro" do campo receptivo efetivo: o menor raio, a
    partir do centro, que contém `energy_threshold` da energia total
    do gradiente (soma de |grad|).
    """
    H, W = grad_map.shape
    cy, cx = H // 2, W // 2

    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    total_energy = grad_map.sum()
    sorted_idx = np.argsort(dist, axis=None)
    sorted_dist = dist.flatten()[sorted_idx]
    sorted_grad = grad_map.flatten()[sorted_idx]
    cumulative = np.cumsum(sorted_grad)

    idx = np.searchsorted(cumulative, energy_threshold * total_energy)
    radius = sorted_dist[min(idx, len(sorted_dist) - 1)]
    return 2 * radius  # diâmetro


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    image, _ = dataset[5]

    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_baseline.pt", map_location=device))

    grad_map = compute_effective_receptive_field(model, image, device)
    erf_diameter = estimate_erf_diameter(grad_map, energy_threshold=0.9)

    print(f"Campo receptivo efetivo (90% da energia): ~{erf_diameter:.1f}px")
    print(f"(campo receptivo teórico calculado antes: 140px)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title("Imagem de entrada")
    axes[0].axis("off")

    im = axes[1].imshow(grad_map, cmap="hot")
    axes[1].set_title(f"Campo receptivo efetivo (~{erf_diameter:.0f}px de diâmetro)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig("outputs/effective_receptive_field.png", dpi=150)
    print("Figura salva em outputs/effective_receptive_field.png")


if __name__ == "__main__":
    main()