import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.synthetic import SyntheticEllipseDataset
from src.models.resunet import ResUNet
from src.metrics.segmentation import iou_score, dice_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # --- dados ---
    train_dataset = SyntheticEllipseDataset(
        size=128, min_ellipses=5, max_ellipses=20,
        num_samples=200, seed=42,
    )
    val_dataset = SyntheticEllipseDataset(
        size=128, min_ellipses=5, max_ellipses=20,
        num_samples=40, seed=999,  # seed diferente: amostras não vistas no treino
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # --- modelo, loss, otimizador ---
    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- treino ---
    num_epochs = 15
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

        print(f"Epoch {epoch+1}/{num_epochs} - loss: {epoch_loss / len(train_loader):.6f}")

    elapsed = time.time() - start_time
    print(f"\nTreino concluído em {elapsed:.1f}s")

    # --- avaliação ---
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

    if elapsed > 300:
        print("\nATENÇÃO: treino excedeu 5 minutos, ajustar epochs/batch_size.")


if __name__ == "__main__":
    main()