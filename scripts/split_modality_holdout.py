"""
Split para o teste de estresse da Parte 6 ("mudança de modalidade").

Reaproveita a MESMA heurística de modalidade já usada no split
estratificado original (scripts/split_dataset.py): variância média
entre canais RGB. A diferença é o que se faz com essa classificação:

    - no split original (data/splits/split.json), as duas modalidades
      são ESTRATIFICADAS -- aparecem proporcionalmente em treino,
      val e teste;
    - aqui, uma modalidade é SEGREGADA -- some do treino e da
      validação por completo, e vira o único conteúdo do teste.

    "colored"   (H&E / brightfield, alta variância entre canais)
                -> usada inteira para treino/validação
    "grayscale" (fluorescência, variância ~0 entre canais)
                -> nunca aparece no treino nem na validação; é o
                   conjunto de teste do estresse

Isso simula o cenário real de mudar de coloração/protocolo de
microscopia entre o dado de desenvolvimento e o dado de produção.
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split


from scripts.split_dataset import estimate_color_variance, classify_modality


def build_modality_holdout_split(root_dir, seed=42, val_frac=0.15,
                                 threshold=50.0, grayscale_train_frac=0.15):
    root = Path(root_dir)
    records = []
    for sample_dir in sorted(root.iterdir()):
        if not sample_dir.is_dir():
            continue
        image_id = sample_dir.name
        image_path = sample_dir / "images" / f"{image_id}.png"
        variance = estimate_color_variance(image_path)
        modality = classify_modality(variance, threshold=threshold)
        records.append({"image_id": image_id, "modality": modality})

    colored_ids = [r["image_id"] for r in records if r["modality"] == "colored"]
    grayscale_ids = [r["image_id"] for r in records if r["modality"] == "grayscale"]

    print(f"Coloridas (viram treino/val): {len(colored_ids)}")
    print(f"Grayscale (holdout): {len(grayscale_ids)}")

    if not colored_ids or not grayscale_ids:
        raise ValueError("Uma modalidade ficou vazia — revise inspect_modalities.py.")


    train_ids, val_ids = train_test_split(
        colored_ids, test_size=val_frac, random_state=seed,
    )

    # Divide grayscale em "grayscale_train" (só baseline vê)
    # e "grayscale_test" (avaliação de ambos)
    grayscale_train_ids, grayscale_test_ids = train_test_split(
        grayscale_ids,
        test_size=(1.0 - grayscale_train_frac),
        random_state=seed,
    )

    split = {
        "train": train_ids,                 # colored train -- usado pelos DOIS
        "val":   val_ids,                   # colored val  -- usado pelos DOIS
        "test":  grayscale_test_ids,        # avaliação de ambos (471 imgs)
        "grayscale_train": grayscale_train_ids,  # baseline adiciona isso (91 imgs)
        "held_out_modality": "grayscale",
        "seen_modality": "colored",
        "threshold": threshold,
    }

    out_path = Path("data") / "splits" / "split_modality_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nTreino colored: {len(train_ids)}, Val colored: {len(val_ids)}")
    print(f"Grayscale_train (só baseline): {len(grayscale_train_ids)}")
    print(f"Grayscale_test (avaliação): {len(grayscale_test_ids)}")
    print(f"Split salvo em {out_path}")
    return split


if __name__ == "__main__":
    build_modality_holdout_split("data/raw/stage1_train")