# scripts/validate_split.py

import json
from pathlib import Path

import cv2
import numpy as np


def estimate_color_variance(image_path):
    """
    Proxy simples para detectar modalidade: imagens verdadeiramente
    grayscale (fluorescência) têm variância ~0 entre canais R,G,B.
    Imagens coloridas (histologia/brightfield) têm variância alta.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return image.var(axis=2).mean()


def load_split(split_path="data/splits/split.json"):
    with open(split_path) as f:
        return json.load(f)


def check_no_leakage(split):
    train_ids = set(split["train"])
    val_ids = set(split["val"])
    test_ids = set(split["test"])

    overlaps = {
        "train∩val": train_ids & val_ids,
        "train∩test": train_ids & test_ids,
        "val∩test": val_ids & test_ids,
    }

    ok = True
    for name, overlap in overlaps.items():
        if overlap:
            print(f"FALHA: {name} tem {len(overlap)} IDs em comum!")
            ok = False

    if ok:
        print("OK: nenhum vazamento entre treino/val/teste.")

    return ok


def check_total_coverage(split, root_dir="data/raw/stage1_train"):
    root = Path(root_dir)
    all_ids = {d.name for d in root.iterdir() if d.is_dir()}

    split_ids = set(split["train"]) | set(split["val"]) | set(split["test"])

    missing = all_ids - split_ids
    extra = split_ids - all_ids

    if missing:
        print(f"AVISO: {len(missing)} imagens do dataset não estão em nenhum split.")
    if extra:
        print(f"FALHA: {len(extra)} IDs no split não existem no dataset!")

    if not missing and not extra:
        print(f"OK: split cobre exatamente as {len(all_ids)} imagens do dataset.")


def check_stratification(split, root_dir="data/raw/stage1_train", threshold=50.0):
    root = Path(root_dir)

    print("\nProporção de imagens coloridas por conjunto:")
    for split_name, ids in split.items():
        n_colored = 0
        for image_id in ids:
            image_path = root / image_id / "images" / f"{image_id}.png"
            variance = estimate_color_variance(image_path)
            if variance > threshold:
                n_colored += 1

        pct = 100 * n_colored / len(ids)
        print(f"  {split_name:6s}: {n_colored:3d}/{len(ids):3d} coloridas ({pct:.1f}%)")


def main():
    split = load_split()

    print(f"Tamanhos: treino={len(split['train'])}, "
          f"val={len(split['val'])}, teste={len(split['test'])}\n")

    check_no_leakage(split)
    check_total_coverage(split)
    check_stratification(split)


if __name__ == "__main__":
    main()