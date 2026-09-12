# scripts/train_dsb2018_normalized.py

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet
from src.metrics.segmentation import iou_score, dice_score


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    split = load_split()
    full_dataset = DSB2018Dataset(
        "data/raw/stage1_train", target_size=(128, 128), normalize="percentile"
    )

    train_dataset = build_subset(full_dataset, split["train"])
    val_dataset = build_subset(full_dataset, split["val"])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
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

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTreino concluído em {elapsed:.1f}s ({elapsed/60:.2f} min)")

    model.eval()
    total_iou, total_dice, n_batches = 0.0, 0.0, 0

    with torch.no_grad():
        for images, instance_masks in val_loader:
            images = images.to(device)
            binary_masks = (instance_masks > 0).to(device)

            logits = model(images)
            preds = (torch.sigmoid(logits).squeeze(1) > 0.5)

            total_iou += iou_score(preds, binary_masks)
            total_dice += dice_score(preds, binary_masks)
            n_batches += 1

    print(f"IoU médio (val): {total_iou / n_batches:.4f}")
    print(f"Dice médio (val): {total_dice / n_batches:.4f}")

    Path("outputs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "outputs/resunet_normalized.pt")
    print("Checkpoint salvo em outputs/resunet_normalized.pt")


if __name__ == "__main__":
    main()