import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset
from src.models.resunet import ResUNet
from src.postprocessing.watershed_decode import decode_watershed
from src.metrics.instance_matching import mean_average_precision, count_error


def load_split(split_path="data/splits/split.json"):
    with open(split_path) as f:
        return json.load(f)


def build_val_indices(sample_dirs, val_ids):
    val_ids = set(val_ids)
    return [i for i, d in enumerate(sample_dirs) if d.name in val_ids]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = load_split()

    # dataset "3 classes" só serve pra alimentar o modelo (imagem + rótulo de treino)
    dataset_3class = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))
    # dataset "clássico" da Parte 1 é quem tem a instance_mask de verdade (GT)
    dataset_instances = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))

    val_indices = build_val_indices(dataset_3class.sample_dirs, split["val"])

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_3class.pt", map_location=device))
    model.eval()

    maps, count_errors = [], []
    n_gt_per_image = []  # pra depois segmentar por densidade (item 6 do enunciado)

    with torch.no_grad():
        for idx in val_indices:
            image, _ = dataset_3class[idx]              # imagem já pré-processada
            _, gt_instances = dataset_instances[idx]      # (H, W) long, GT de instâncias

            gt_instances = gt_instances.numpy()

            logits = model(image.unsqueeze(0).to(device))       # (1, 3, H, W)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()   # (3, H, W)

            pred_instances = decode_watershed(probs)

            score = mean_average_precision(pred_instances, gt_instances)
            err = count_error(pred_instances, gt_instances)

            maps.append(score)
            count_errors.append(err)
            n_gt_per_image.append(int(gt_instances.max()))

    maps = np.array(maps)
    count_errors = np.array(count_errors)
    n_gt_per_image = np.array(n_gt_per_image)

    print(f"mAP médio (0.50:0.95): {maps.mean():.4f}")
    print(f"Erro de contagem médio: {count_errors.mean():.4f}")

    # quebra por densidade -- é o que a apresentação (item 6) pede
    low_density = n_gt_per_image < 10
    high_density = n_gt_per_image >= 30

    if low_density.any():
        print(f"\nBaixa densidade (<10 núcleos, n={low_density.sum()}):")
        print(f"  mAP médio: {maps[low_density].mean():.4f}")
        print(f"  Erro contagem médio: {count_errors[low_density].mean():.4f}")

    if high_density.any():
        print(f"\nAlta densidade (>=30 núcleos, n={high_density.sum()}):")
        print(f"  mAP médio: {maps[high_density].mean():.4f}")
        print(f"  Erro contagem médio: {count_errors[high_density].mean():.4f}")


if __name__ == "__main__":
    main()