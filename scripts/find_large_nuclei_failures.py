# scripts/find_large_nuclei_failures.py

import json

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
    """Diâmetro equivalente do MAIOR núcleo presente na imagem."""
    ids = np.unique(mask)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return 0.0
    areas = [(mask == i).sum() for i in ids]
    max_area = max(areas)
    return 2 * np.sqrt(max_area / np.pi)


def main(erf_reference=40.7, top_k=10):
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

            max_diam = largest_diameter_in_image(gt_np)

            logits = model(image.unsqueeze(0).to(device))
            pred_binary = (torch.sigmoid(logits).squeeze() > 0.5).cpu().numpy()
            pred_instances, _ = extract_instances_naive(pred_binary)

            score = mean_average_precision(pred_instances, gt_np)
            err = count_error(pred_instances, gt_np)

            original_idx = test_dataset.indices[i]

            records.append({
                "original_idx": original_idx,
                "max_diameter": max_diam,
                "mAP": score,
                "count_error": err,
                "n_gt_instances": int(gt_mask.max()),
                "exceeds_erf": max_diam > erf_reference,
            })

    records.sort(key=lambda r: r["max_diameter"], reverse=True)

    print(f"Referência de ERF: {erf_reference}px\n")
    print(f"{'idx':>6} | {'maior núcleo':>13} | {'excede ERF':>10} | {'mAP':>6} | {'erro cont.':>10} | {'nº núcleos':>10}")
    print("-" * 75)
    for r in records[:top_k]:
        print(f"{r['original_idx']:>6} | {r['max_diameter']:>13.1f} | "
              f"{'SIM' if r['exceeds_erf'] else 'não':>10} | {r['mAP']:>6.3f} | "
              f"{r['count_error']:>10} | {r['n_gt_instances']:>10}")

    n_exceeds = sum(r["exceeds_erf"] for r in records)
    print(f"\n{n_exceeds}/{len(records)} imagens do teste têm ao menos "
          f"um núcleo excedendo o ERF de referência ({erf_reference}px).")

    maps_exceed = [r["mAP"] for r in records if r["exceeds_erf"]]
    maps_normal = [r["mAP"] for r in records if not r["exceeds_erf"]]

    if maps_exceed:
        print(f"\nmAP médio (excede ERF): {np.mean(maps_exceed):.4f} (n={len(maps_exceed)})")
    if maps_normal:
        print(f"mAP médio (dentro do ERF): {np.mean(maps_normal):.4f} (n={len(maps_normal)})")


if __name__ == "__main__":
    main()