import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet
from src.models.segnet_style import ResUNetPoolIndices
from src.metrics.segmentation import iou_score, dice_score
from src.postprocess.naive_instance import extract_instances_naive
from src.metrics.instance_matching import mean_average_precision, count_error

SEEDS = [0, 1]
NUM_EPOCHS = 2  # ablação: reduzido em relação ao treino final (50)

MODELS = {
    "skip_connections": ResUNet,
    "pool_indices": ResUNetPoolIndices,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(path="data/splits/split.json"):
    with open(path) as f:
        return json.load(f)


def build_subset(dataset, allowed_ids):
    allowed_ids = set(allowed_ids)
    idxs = [i for i, d in enumerate(dataset.sample_dirs) if d.name in allowed_ids]
    return Subset(dataset, idxs)


def train_one(model_name, model_cls, seed, device):
    set_seed(seed)

    split = load_split()
    full_dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    train_ds = build_subset(full_dataset, split["train"])
    val_ds = build_subset(full_dataset, split["val"])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = model_cls(in_channels=3, num_classes=1, base_channels=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for images, instance_masks in train_loader:
            images = images.to(device)
            binary_masks = (instance_masks > 0).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, binary_masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[{model_name} | seed {seed}] epoch {epoch+1}/{NUM_EPOCHS} "
              f"- loss {epoch_loss/len(train_loader):.4f}")

    # --- avaliação: IoU/Dice (semântico) + mAP/erro de contagem (instância) ---
    model.eval()
    total_iou, total_dice, n_batches = 0.0, 0.0, 0
    maps, errs = [], []

    with torch.no_grad():
        for images, instance_masks in val_loader:
            images = images.to(device)
            binary_masks_gt = (instance_masks > 0).to(device)

            logits = model(images)
            preds = (torch.sigmoid(logits).squeeze(1) > 0.5)

            total_iou += iou_score(preds, binary_masks_gt)
            total_dice += dice_score(preds, binary_masks_gt)
            n_batches += 1

            for b in range(preds.shape[0]):
                pred_np = preds[b].cpu().numpy()
                gt_np = instance_masks[b].numpy()
                pred_instances, _ = extract_instances_naive(pred_np)
                maps.append(mean_average_precision(pred_instances, gt_np))
                errs.append(count_error(pred_instances, gt_np))

    return {
        "iou_mean": total_iou / n_batches,
        "dice_mean": total_dice / n_batches,
        "mAP_mean": float(np.mean(maps)),
        "count_error_mean": float(np.mean(errs)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for name, model_cls in MODELS.items():
        seed_results = [train_one(name, model_cls, seed, device) for seed in SEEDS]

        agg = {}
        for key in seed_results[0]:
            values = [r[key] for r in seed_results]
            agg[f"{key}_avg"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        results[name] = agg

        print(f"\n=== {name} ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/ablation_upsampling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em outputs/ablation_upsampling_results.json")


if __name__ == "__main__":
    main()