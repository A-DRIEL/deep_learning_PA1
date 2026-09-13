# scripts/train_baseline_with_grayscale.py
"""
Baseline justo para o teste de estresse (Parte 6).

Treina o MESMO ResUNet-3class, mesmo split, mesmos hiperparâmetros que
o holdout -- mas inclui as 91 imagens 'grayscale_train' no treino.
Assim, a ÚNICA diferença entre os dois modelos é ter visto grayscale
no treino ou não. A avaliação (evaluate_modality_stress.py) roda em
grayscale_test, que nenhum dos dois viu.
"""
import json
from scripts.train_dsb2018_3class import train_resunet_3class


def load_split(path="data/splits/split_modality_holdout.json"):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    split = load_split()
    train_resunet_3class(
        split_path="data/splits/split_modality_holdout.json",
        checkpoint_path="outputs/resunet_3class_with_grayscale.pt",
        seed=42,
        extra_train_ids=split["grayscale_train"],
    )