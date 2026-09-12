# scripts/find_worst_images.py

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


def largest_diameter_in_image(mask):
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return 0.0
    areas = [(mask == i).sum() for i in ids]
    return 2 * np.sqrt(max(areas) / np.pi)


def main(top_k=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_baseline.pt", map_location=device))
    model.eval()

    split = load_split()
    full_dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    test_dataset = build_subset(full_dataset, split["test"])

    records = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, gt_mask = test_dataset[i]
            gt_np = gt_mask.numpy()

            logits = model(image.unsqueeze(0).to(device))
            pred_binary = (torch.sigmoid(logits).squeeze() > 0.5).cpu().numpy()
            pred_instances, n_pred = extract_instances_naive(pred_binary)

            score = mean_average_precision(pred_instances, gt_np)
            err = count_error(pred_instances, gt_np)
            max_diam = largest_diameter_in_image(gt_np)

            original_idx = test_dataset.indices[i]

            records.append({
                "original_idx": original_idx,
                "mAP": score,
                "count_error": err,
                "n_gt_instances": int(gt_mask.max()),
                "n_pred_instances": n_pred,
                "max_diameter": max_diam,
                "image": image,
                "gt_mask": gt_np,
                "pred_binary": pred_binary,
                "pred_instances": pred_instances,
            })

    records.sort(key=lambda r: r["mAP"])
    worst = records[:top_k]

    print(f"{'idx':>6} | {'mAP':>6} | {'erro cont.':>10} | {'nº real':>8} | {'nº pred':>8} | {'maior núcleo':>13}")
    print("-" * 70)
    for r in worst:
        print(f"{r['original_idx']:>6} | {r['mAP']:>6.3f} | {r['count_error']:>10} | "
              f"{r['n_gt_instances']:>8} | {r['n_pred_instances']:>8} | {r['max_diameter']:>13.1f}")

    fig, axes = plt.subplots(top_k, 4, figsize=(14, 3.2 * top_k))

    for row, r in enumerate(worst):
        img_np = r["image"].permute(1, 2, 0).numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f"idx={r['original_idx']}\nmAP={r['mAP']:.3f}",
                                  fontsize=9, rotation=0, ha="right", va="center")
        axes[row, 1].imshow(r["gt_mask"], cmap="nipy_spectral")
        axes[row, 2].imshow(r["pred_binary"], cmap="gray")
        axes[row, 3].imshow(r["pred_instances"], cmap="nipy_spectral")

        if row == 0:
            axes[row, 0].set_title("Imagem")
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].set_title("Predição binária")
            axes[row, 3].set_title("Instâncias extraídas")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    plt.savefig("outputs/worst_images_gallery.png", dpi=120)
    print("\nFigura salva em outputs/worst_images_gallery.png")


if __name__ == "__main__":
    main()