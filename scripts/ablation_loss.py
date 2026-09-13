import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset
from src.models.resunet import ResUNet
from src.losses.focal import FocalLoss
from src.postprocessing.watershed_decode import decode_watershed
from src.metrics.instance_matching import mean_average_precision, count_error

SEEDS = [0, 1]
NUM_EPOCHS = 20 

# ---- as 6 configurações que cobrem CE -> CE bal. -> focal -> focal bal. ----
CONFIGS = {
    "ce":              {"gamma": 0.0, "weighted": False},
    "ce_balanceada":   {"gamma": 0.0, "weighted": True},
    "focal_g1":        {"gamma": 1.0, "weighted": False},
    "focal_g2":        {"gamma": 2.0, "weighted": False},
    "focal_g5":        {"gamma": 5.0, "weighted": False},
    "focal_balanceada":{"gamma": 2.0, "weighted": True},
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


def load_class_weights(path="data/class_weights_3class.json", device="cpu"):
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data["weights"], dtype=torch.float32, device=device)


def border_pixel_recall(logits, target, border_class=2):
    """Recall pixel-a-pixel só da classe borda -- mostra o efeito do
    desbalanceamento de forma direta, sem precisar do watershed."""
    pred = logits.argmax(dim=1)
    is_border_gt = target == border_class
    if is_border_gt.sum() == 0:
        return None
    tp = ((pred == border_class) & is_border_gt).sum().item()
    return tp / is_border_gt.sum().item()


def train_one_config(config_name, cfg, seed, device):
    set_seed(seed)

    split = load_split()
    ds_3class = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))
    ds_instances = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))

    train_ds = build_subset(ds_3class, split["train"])
    val_ds_3class = build_subset(ds_3class, split["val"])
    val_ids = set(split["val"])
    val_idx_instances = [i for i, d in enumerate(ds_instances.sample_dirs) if d.name in val_ids]

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)

    alpha = load_class_weights(device=device) if cfg["weighted"] else None
    criterion = FocalLoss(gamma=cfg["gamma"], alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"[{config_name} | seed {seed}] epoch {epoch+1}/{NUM_EPOCHS} "
              f"- loss {epoch_loss/len(train_loader):.4f}")

    # --- avaliação em validação: mAP de instância + recall de borda ---
    model.eval()
    maps, errs, recalls = [], [], []
    with torch.no_grad():
        for i in val_idx_instances:
            image, _ = ds_3class[i]
            _, gt_instances = ds_instances[i]
            gt_instances = gt_instances.numpy()

            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_instances = decode_watershed(probs)

            maps.append(mean_average_precision(pred_instances, gt_instances))
            errs.append(count_error(pred_instances, gt_instances))

            label_3class = val_ds_3class[val_idx_instances.index(i)][1] \
                if False else None  # (não usado; recall calculado abaixo)

    # recall de borda calculado num segundo passe (mais simples de ler)
    with torch.no_grad():
        for images, labels in DataLoader(val_ds_3class, batch_size=8):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            r = border_pixel_recall(logits, labels)
            if r is not None:
                recalls.append(r)

    return {
        "mAP_mean": float(np.mean(maps)),
        "count_error_mean": float(np.mean(errs)),
        "border_recall_mean": float(np.mean(recalls)) if recalls else None,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for name, cfg in CONFIGS.items():
        seed_results = [train_one_config(name, cfg, seed, device) for seed in SEEDS]

        agg = {}
        for key in seed_results[0]:
            values = [r[key] for r in seed_results if r[key] is not None]
            agg[f"{key}_avg"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        results[name] = agg

        print(f"\n=== {name} ===")
        for k, v in agg.items():
            print(f"  {k}: {v:.4f}")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/ablation_loss_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em outputs/ablation_loss_results.json")


if __name__ == "__main__":
    main()