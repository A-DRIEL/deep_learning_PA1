# scripts/evaluate_instance_map.py

import json
from pathlib import Path

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


def evaluate_image(model, image, gt_mask, device):
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
        pred_binary = (torch.sigmoid(logits).squeeze() > 0.5).cpu().numpy()

    pred_instances, _ = extract_instances_naive(pred_binary)
    gt_np = gt_mask.numpy()

    map_score = mean_average_precision(pred_instances, gt_np)
    err = count_error(pred_instances, gt_np)
    n_gt_instances = int(gt_mask.max().item())

    return {
        "mAP": map_score,
        "count_error": err,
        "n_gt_instances": n_gt_instances,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_baseline.pt", map_location=device))
    model.eval()

    split = load_split()
    full_dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    test_dataset = build_subset(full_dataset, split["test"])

    print(f"Avaliando em {len(test_dataset)} imagens de teste...\n")

    results = []
    for i in range(len(test_dataset)):
        image, gt_mask = test_dataset[i]
        result = evaluate_image(model, image, gt_mask, device)
        results.append(result)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(test_dataset)} imagens avaliadas...")

    # --- agregação ---
    maps = np.array([r["mAP"] for r in results])
    errors = np.array([r["count_error"] for r in results])
    densities = np.array([r["n_gt_instances"] for r in results])

    print(f"\nmAP médio (teste): {maps.mean():.4f} (± {maps.std():.4f})")
    print(f"Erro de contagem médio: {errors.mean():.2f} (± {errors.std():.2f})")
    print(f"Densidade (nº instâncias reais) - min: {densities.min()}, "
          f"max: {densities.max()}, média: {densities.mean():.1f}")

    # --- salva resultados brutos para o gráfico e para reprodutibilidade ---
    Path("outputs").mkdir(exist_ok=True)
    output = {
        "per_image": [
            {
                "mAP": r["mAP"],
                "count_error": r["count_error"],
                "n_gt_instances": r["n_gt_instances"],
            }
            for r in results
        ],
        "summary": {
            "mAP_mean": float(maps.mean()),
            "mAP_std": float(maps.std()),
            "count_error_mean": float(errors.mean()),
            "count_error_std": float(errors.std()),
        },
    }
    with open("outputs/instance_map_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResultados salvos em outputs/instance_map_results.json")


if __name__ == "__main__":
    main()