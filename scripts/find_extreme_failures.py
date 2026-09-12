# scripts/find_extreme_failures.py

import json

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


def build_indices(sample_dirs, allowed_ids):
    allowed_ids = set(allowed_ids)
    return [i for i, d in enumerate(sample_dirs) if d.name in allowed_ids]


def main(coverage_threshold=0.8):
    """
    coverage_threshold: fração da área total de foreground (GT) que uma
    ÚNICA instância prevista precisa cobrir para contar como "achou que
    tudo é um núcleo só".
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_split()
    dataset_instances = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    dataset_3class = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))

    test_indices = build_indices(dataset_instances.sample_dirs, split["test"])

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_3class.pt", map_location=device))
    model.eval()

    total_misses = []
    total_merges = []

    with torch.no_grad():
        for idx in test_indices:
            image, _ = dataset_3class[idx]
            _, gt_instances = dataset_instances[idx]
            gt_np = gt_instances.numpy()
            n_gt = int(gt_instances.max())

            if n_gt == 0:
                continue

            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_instances = decode_watershed(probs)

            pred_ids = np.unique(pred_instances)
            pred_ids = pred_ids[pred_ids != 0]
            n_pred = len(pred_ids)

            score = mean_average_precision(pred_instances, gt_np)
            err = count_error(pred_instances, gt_np)

            if n_pred == 0:
                total_misses.append({
                    "idx": idx, "n_gt": n_gt, "n_pred": n_pred,
                    "mAP": score, "count_error": err,
                })

            if n_pred >= 1:
                gt_foreground_area = (gt_np > 0).sum()
                largest_pred_area = max((pred_instances == pid).sum() for pid in pred_ids)
                coverage = largest_pred_area / gt_foreground_area if gt_foreground_area > 0 else 0

                if n_pred <= 2 and coverage > coverage_threshold and n_gt >= 5:
                    total_merges.append({
                        "idx": idx, "n_gt": n_gt, "n_pred": n_pred,
                        "coverage": coverage, "mAP": score, "count_error": err,
                    })

    print(f"=== MODO 1: Falso negativo total (n_pred=0, n_gt>0) ===")
    print(f"Encontrados: {len(total_misses)} casos")
    for m in sorted(total_misses, key=lambda r: -r["n_gt"])[:5]:
        print(f"  idx={m['idx']:>4}  n_gt={m['n_gt']:>3}  mAP={m['mAP']:.3f}  erro={m['count_error']}")

    print(f"\n=== MODO 2: Fusão total (poucas instâncias cobrindo >{coverage_threshold*100:.0f}% do foreground) ===")
    print(f"Encontrados: {len(total_merges)} casos")
    for m in sorted(total_merges, key=lambda r: -r["n_gt"])[:5]:
        print(f"  idx={m['idx']:>4}  n_gt={m['n_gt']:>3}  n_pred={m['n_pred']}  "
              f"cobertura={m['coverage']:.2f}  mAP={m['mAP']:.3f}  erro={m['count_error']}")


if __name__ == "__main__":
    main()