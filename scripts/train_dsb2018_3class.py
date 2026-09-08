import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset
from src.models.resunet import ResUNet


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


def load_class_weights(path="data/class_weights_3class.json", device="cpu"):
    with open(path) as f:
        data = json.load(f)
    return torch.tensor(data["weights"], dtype=torch.float32, device=device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    split = load_split()
    full_dataset = DSB2018ThreeClassDataset(
        "data/raw/stage1_train", target_size=(128, 128), border_width=2
    )

    train_dataset = build_subset(full_dataset, split["train"])
    val_dataset = build_subset(full_dataset, split["val"])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # única mudança arquitetural: num_classes=3
    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)

    class_weights = load_class_weights(device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)  # espera target (B,H,W) long
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    start_time = time.time()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)  # (B, H, W), valores em {0,1,2}

            optimizer.zero_grad()
            logits = model(images)  # (B, 3, H, W)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTreino concluído em {elapsed:.1f}s ({elapsed/60:.2f} min)")

    Path("outputs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "outputs/resunet_3class.pt")
    print("Checkpoint salvo em outputs/resunet_3class.pt")


if __name__ == "__main__":
    main()