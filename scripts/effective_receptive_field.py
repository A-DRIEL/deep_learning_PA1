# scripts/effective_receptive_field.py -- versão atualizada

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet


def compute_effective_receptive_field(model, image, device):
    model.eval()
    image = image.clone().unsqueeze(0).to(device)
    image.requires_grad_(True)

    logits = model(image)
    _, _, H, W = logits.shape
    center_y, center_x = H // 2, W // 2

    model.zero_grad()
    target = logits[0, 0, center_y, center_x]
    target.backward()

    grad = image.grad[0].abs().mean(dim=0)
    grad_np = grad.cpu().numpy()
    return grad_np / (grad_np.max() + 1e-8)


def estimate_erf_diameter(grad_map, energy_threshold=0.9):
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
    return 2 * radius


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_baseline.pt", map_location=device))

    indices_to_test = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450]
    erfs = []

    print("Medindo ERF em múltiplas imagens...")
    for idx in indices_to_test:
        image, _ = dataset[idx]
        grad_map = compute_effective_receptive_field(model, image, device)
        erf = estimate_erf_diameter(grad_map, energy_threshold=0.9)
        erfs.append(erf)
        print(f"  idx={idx}: ERF={erf:.1f}px")

    erfs = np.array(erfs)
    print(f"\nERF médio: {erfs.mean():.1f}px (± {erfs.std():.1f})")
    print(f"ERF mín: {erfs.min():.1f}px, máx: {erfs.max():.1f}px")
    print(f"(campo receptivo teórico: 140px)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(erfs, bins=10, color="tomato", edgecolor="black", alpha=0.8)
    ax.axvline(erfs.mean(), color="black", linestyle="--",
               label=f"média = {erfs.mean():.1f}px")
    ax.set_xlabel("Campo receptivo efetivo (px)")
    ax.set_ylabel("Número de imagens testadas")
    ax.set_title("Variabilidade do ERF entre imagens (RF teórico = 140px)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/erf_variability.png", dpi=150)
    print("\nFigura salva em outputs/erf_variability.png")


if __name__ == "__main__":
    main()