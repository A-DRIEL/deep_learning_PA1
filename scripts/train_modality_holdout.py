"""
Treina o modelo da Parte 2 usando o split de holdout de modalidade.
Reaproveita train_resunet_3class de scripts/train_dsb2018_3class.py --
a única coisa que muda é qual split e qual checkpoint são usados.
"""

from scripts.train_dsb2018_3class import train_resunet_3class

if __name__ == "__main__":
    train_resunet_3class(
        split_path="data/splits/split_modality_holdout.json",
        checkpoint_path="outputs/resunet_3class_no_grayscale.pt",
        seed=42,
    )