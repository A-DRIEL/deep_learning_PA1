# src/datasets/split.py

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def estimate_color_variance(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return image.var(axis=2).mean()


def classify_modality(variance, threshold=50.0):
    return "colored" if variance > threshold else "grayscale"


def build_stratified_split(root_dir, seed=42, train_frac=0.7, val_frac=0.15):
    root = Path(root_dir)
    records = []

    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue
        image_id = sample_dir.name
        image_path = sample_dir / "images" / f"{image_id}.png"
        variance = estimate_color_variance(image_path)
        modality = classify_modality(variance)
        records.append({"image_id": image_id, "modality": modality})

    image_ids = [r["image_id"] for r in records]
    modalities = [r["modality"] for r in records]

    # relatório de contagem por modalidade
    n_gray = modalities.count("grayscale")
    n_colored = modalities.count("colored")
    print(f"Grayscale: {n_gray}, Colorido: {n_colored}, Total: {len(records)}")

    # split estratificado: primeiro separa teste, depois divide o resto em treino/val
    test_frac = 1.0 - train_frac - val_frac

    train_val_ids, test_ids, train_val_mod, _ = train_test_split(
        image_ids, modalities,
        test_size=test_frac, stratify=modalities, random_state=seed,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_frac / (train_frac + val_frac),
        stratify=train_val_mod, random_state=seed,
    )

    split = {"train": train_ids, "val": val_ids, "test": test_ids}

    out_path = Path("data") / "splits"/ "split.json"
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Treino: {len(train_ids)}, Val: {len(val_ids)}, Teste: {len(test_ids)}")
    print(f"Split salvo em {out_path}")

    return split


if __name__ == "__main__":
    build_stratified_split("data/raw/stage1_train")