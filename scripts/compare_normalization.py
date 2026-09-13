# scripts/compare_normalization.py

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet
from src.postprocess.naive_instance import extract_instances_naive
from src.metrics.instance_matching import mean_average_precision, count_error


def load_split(split_path="data/splits/split.json"):
    with open(split_path) as f:
        return json.load(f)


def build_subset(dataset, allowed_ids):
    allowed_ids = set(allowed_ids)
    indices = [
        i for i, sample_dir in enumerate(dataset.sample_dirs)
        if sample_dir.name in allowed_ids
    ]
    return Subset(dataset, indices)


def evaluate_on_idx(model, dataset, idx, device):
    image, gt_mask = dataset[idx]
    gt_np = gt_mask.numpy()

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        pred_binary = (torch.sigmoid(logits).squeeze() > 0.5).cpu().numpy()

    pred_instances, _ = extract_instances_naive(pred_binary)
    score = mean_average_precision(pred_instances, gt_np)
    err = count_error(pred_instances, gt_np)
    return image, gt_np, pred_binary, pred_instances, score, err


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_fixed = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128), normalize="fixed")
    dataset_percentile = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128), normalize="percentile")

    model_fixed = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model_fixed.load_state_dict(torch.load("outputs/resunet_baseline.pt", map_location=device))
    model_fixed.eval()

    model_norm = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model_norm.load_state_dict(torch.load("outputs/resunet_normalized.pt", map_location=device))
    model_norm.eval()

    target_indices = [65, 296, 602]  # os casos de contraste/modalidade da galeria de falhas

    fig, axes = plt.subplots(len(target_indices), 4, figsize=(14, 3.5 * len(target_indices)))

    for row, idx in enumerate(target_indices):
        _, gt_np, pred_bin_old, pred_inst_old, map_old, err_old = evaluate_on_idx(
            model_fixed, dataset_fixed, idx, device
        )
        image_new, _, pred_bin_new, pred_inst_new, map_new, err_new = evaluate_on_idx(
            model_norm, dataset_percentile, idx, device
        )

        img_np = image_new.permute(1, 2, 0).numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f"idx={idx}", fontsize=10, rotation=0, ha="right", va="center")
        axes[row, 1].imshow(gt_np, cmap="nipy_spectral")
        axes[row, 2].imshow(pred_inst_old, cmap="nipy_spectral")
        axes[row, 2].set_title(f"Antes: mAP={map_old:.3f}", fontsize=10)
        axes[row, 3].imshow(pred_inst_new, cmap="nipy_spectral")
        axes[row, 3].set_title(f"Depois: mAP={map_new:.3f}", fontsize=10)

        if row == 0:
            axes[row, 0].set_title("Imagem")
            axes[row, 1].set_title("Ground truth")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        print(f"idx={idx}: antes mAP={map_old:.4f} erro={err_old}  |  "
              f"depois mAP={map_new:.4f} erro={err_new}")

    plt.tight_layout()
    plt.savefig("outputs/normalization_before_after.png", dpi=120)
    print("\nFigura salva em outputs/normalization_before_after.png")


if __name__ == "__main__":
    main()