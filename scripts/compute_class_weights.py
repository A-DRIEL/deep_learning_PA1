import json
from pathlib import Path

import numpy as np

from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset


def compute_class_weights(dataset, num_classes=3):
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        _, label = dataset[i]
        for c in range(num_classes):
            counts[c] += (label == c).sum().item()

    total = counts.sum()
    weights = total / (num_classes * counts)  # peso_c ≈ N_total / (3 * N_c)
    return weights, counts


if __name__ == "__main__":
    ds = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))
    weights, counts = compute_class_weights(ds)

    print(f"Contagem por classe (fundo, interior, borda): {counts}")
    print(f"Pesos sugeridos: {weights}")

    Path("data").mkdir(exist_ok=True)
    with open("data/class_weights_3class.json", "w") as f:
        json.dump({"counts": counts.tolist(), "weights": weights.tolist()}, f, indent=2)