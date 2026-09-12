# scripts/failure_gallery_trilha_a.py

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset
from src.models.resunet import ResUNet
from src.postprocessing.watershed_decode import decode_watershed
from src.metrics.instance_matching import mean_average_precision, count_error


def load_split(split_path="data/splits/split.json"):
    with open(split_path) as f:
        return json.load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_instances = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    dataset_3class = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_3class.pt", map_location=device))
    model.eval()

    # os 5 casos escolhidos da galeria de falhas
    failure_indices = [65, 296, 602, 386, 244]

    fig, axes = plt.subplots(len(failure_indices), 4, figsize=(16, 4 * len(failure_indices)))

    for row, idx in enumerate(failure_indices):
        image, _ = dataset_3class[idx]
        _, gt_instances = dataset_instances[idx]
        gt_np = gt_instances.numpy()

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()  # (3, H, W)

        pred_instances = decode_watershed(probs)
        border_prob = probs[2]  # canal de borda -- o "mapa intermediário"

        score = mean_average_precision(pred_instances, gt_np)
        err = count_error(pred_instances, gt_np)

        img_np = image.permute(1, 2, 0).numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f"idx={idx}\nmAP={score:.3f}",
                                  fontsize=10, rotation=0, ha="right", va="center")
        axes[row, 1].imshow(gt_np, cmap="nipy_spectral")
        axes[row, 2].imshow(pred_instances, cmap="nipy_spectral")
        im = axes[row, 3].imshow(border_prob, cmap="hot", vmin=0, vmax=1)

        if row == 0:
            axes[row, 0].set_title("Imagem")
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].set_title("Predição (Trilha A)")
            axes[row, 3].set_title("Mapa de borda (intermediário)")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        print(f"idx={idx}: mAP={score:.4f}, erro={err}")

    plt.tight_layout()
    plt.savefig("outputs/failure_gallery_trilha_a.png", dpi=120)
    print("\nFigura salva em outputs/failure_gallery_trilha_a.png")


if __name__ == "__main__":
    main()