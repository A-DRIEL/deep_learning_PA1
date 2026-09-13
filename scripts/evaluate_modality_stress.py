# scripts/evaluate_modality_stress.py
"""
Teste de estresse da Parte 6: mudança de modalidade.

Compara dois modelos idênticos em arquitetura, loss e hiperparâmetros,
diferindo SÓ em quais imagens eles viram durante o treino:

    - "com_grayscale_no_treino": checkpoints de scripts/train_dsb2018_3class.py
      (split.json normal, estratificado -- viu as duas modalidades)
    - "sem_grayscale_no_treino": checkpoints de scripts/train_modality_holdout.py
      (split_modality_holdout.json -- nunca viu uma imagem grayscale)

Ambos são avaliados EXCLUSIVAMENTE nas imagens grayscale reservadas em
data/splits/split_modality_holdout.json ("test"), usando o mesmo
pipeline de decodificação (watershed) e as mesmas métricas (mAP
0.50:0.95 + erro de contagem) já usadas em evaluate_dsb2018_3class.py.

A pergunta que isso responde: quanto do desempenho em grayscale vem de
"ter memorizado a aparência dessa modalidade" vs. "ter aprendido a
separar núcleos encostados de forma geral"? Se a queda de mAP do
"sem_grayscale" para o "com_grayscale" for grande, o modelo está
dependendo de pistas específicas de modalidade (cor, textura de fundo)
e não generaliza para o problema de fato.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset
from src.models.resunet import ResUNet
from src.postprocessing.watershed_decode import decode_watershed
from src.metrics.instance_matching import mean_average_precision, count_error


def load_modality_split(path="data/splits/split_modality_holdout.json"):
    with open(path) as f:
        return json.load(f)


def evaluate_checkpoint_on_ids(ckpt_path, image_ids, device):
    """Avalia um checkpoint 3-class + watershed exatamente nas imagens de `image_ids`."""
    dataset_3class = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))
    dataset_instances = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))

    image_ids = set(image_ids)
    indices = [
        i for i, d in enumerate(dataset_3class.sample_dirs)
        if d.name in image_ids
    ]

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    maps, errors = [], []
    with torch.no_grad():
        for idx in indices:
            image, _ = dataset_3class[idx]                 # imagem pré-processada
            _, gt_instances = dataset_instances[idx]         # GT de instâncias real
            gt_instances = gt_instances.numpy()

            logits = model(image.unsqueeze(0).to(device))          # (1, 3, H, W)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()     # (3, H, W)
            pred_instances = decode_watershed(probs)

            maps.append(mean_average_precision(pred_instances, gt_instances))
            errors.append(count_error(pred_instances, gt_instances))

    return np.array(maps), np.array(errors)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modality_split = load_modality_split()
    grayscale_test_ids = modality_split["test"]

    print(f"Avaliando em {len(grayscale_test_ids)} imagens grayscale "
          f"(modalidade '{modality_split['held_out_modality']}')\n")

    # ajuste os nomes de checkpoint conforme os arquivos que vocês
    # realmente salvaram para cada seed
    configs = {
        "com_grayscale_no_treino": [
            "outputs/resunet_3class_with_grayscale.pt"
        ],
        "sem_grayscale_no_treino": [
            "outputs/resunet_3class_no_grayscale.pt"
        ],
    }

    results = {}
    for name, ckpts in configs.items():
        seed_maps, seed_errors = [], []
        for ckpt in ckpts:
            if not Path(ckpt).exists():
                print(f"[aviso] checkpoint não encontrado, pulando: {ckpt}")
                continue
            maps, errors = evaluate_checkpoint_on_ids(ckpt, grayscale_test_ids, device)
            seed_maps.append(maps.mean())
            seed_errors.append(errors.mean())
            print(f"{name} ({ckpt}): mAP={maps.mean():.4f}, "
                  f"erro_contagem_medio={errors.mean():.2f}")

        results[name] = {
            "mAP_mean": float(np.mean(seed_maps)) if seed_maps else None,
            #"mAP_std": float(np.std(seed_maps)) if seed_maps else None,
            "count_error_mean": float(np.mean(seed_errors)) if seed_errors else None,
            #"count_error_std": float(np.std(seed_errors)) if seed_errors else None,
            "n_seeds": len(seed_maps),
        }

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/modality_stress_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResumo:")
    print(json.dumps(results, indent=2))
    print("\nResultados salvos em outputs/modality_stress_results.json")


if __name__ == "__main__":
    main()