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


@torch.no_grad()
def evaluate_val_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        total_loss += criterion(model(images), labels).item()
    model.train()
    return total_loss / len(val_loader)


def train_resunet_3class(
    split_path="data/splits/split.json",
    checkpoint_path="outputs/resunet_3class.pt",
    seed=42,
    num_epochs=40,
    patience=3,
    batch_size=8,
    num_workers=4,
    extra_train_ids=None,        # lista adicional de image_ids no treino
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device} | seed: {seed} | split: {split_path}")

    split = load_split(split_path)
    full_dataset = DSB2018ThreeClassDataset(
        "data/raw/stage1_train", target_size=(128, 128), border_width=2
    )

    train_ids = list(split["train"])
    if extra_train_ids:
        # evita duplicar se algum id aparecer nos dois
        train_ids = train_ids + [i for i in extra_train_ids
                                 if i not in set(split["train"])]
    print(f"Train size efetivo: {len(train_ids)}")

    train_dataset = build_subset(full_dataset, train_ids)
    val_dataset   = build_subset(full_dataset, split["val"])

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)

    class_weights = load_class_weights(device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
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

        avg_train_loss = epoch_loss / len(train_loader)

        if patience is None:
            print(f"Epoch {epoch+1}/{num_epochs} - loss: {avg_train_loss:.4f}")
            continue

        val_loss = evaluate_val_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - loss treino: {avg_train_loss:.4f} - loss val: {val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nSem melhora na loss de val há {patience} épocas -- parando antecipadamente.")
                break

    elapsed = time.time() - start_time
    print(f"\nTreino concluído em {elapsed:.1f}s ({elapsed/60:.2f} min)")

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint salvo em {checkpoint_path}")


def main():
    train_resunet_3class(
        split_path="data/splits/split.json",
        checkpoint_path="outputs/resunet_3class.pt",
        seed=42,
    )


if __name__ == "__main__":
    main()