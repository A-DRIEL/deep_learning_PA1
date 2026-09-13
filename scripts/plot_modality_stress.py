import json

import matplotlib.pyplot as plt

from src.viz.style import apply_clean_style


def main():
    with open("outputs/modality_stress_results.json") as f:
        results = json.load(f)

    labels = ["Treinou COM\ngrayscale", "Treinou SEM\ngrayscale\n(holdout)"]
    keys   = ["com_grayscale_no_treino", "sem_grayscale_no_treino"]

    valid = []
    for label, key in zip(labels, keys):
        r = results.get(key, {})
        mean = r.get("mAP_mean")
        if mean is None:
            print(f"[aviso] sem resultados para '{key}', pulando do gráfico.")
            continue
        valid.append((label, mean))

    if not valid:
        print("Nenhum resultado disponível para plotar.")
        return

    labels_v = [v[0] for v in valid]
    means_v  = [v[1] for v in valid]
    colors   = ["#4C72B0", "#C44E52"][:len(labels_v)]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels_v, means_v, color=colors, width=0.5)
    ax.set_ylabel("mAP (0.50:0.95) -- avaliado só em grayscale")
    ax.set_title(
        "Teste de estresse: mudança de modalidade\n"
        "(1 seed; teste disjunto do treino dos dois modelos)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylim(0, 1)
    apply_clean_style(ax)

    for bar, mean in zip(bars, means_v):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                f"{mean:.3f}", ha="center", fontsize=11)

    plt.tight_layout()
    plt.savefig("outputs/modality_stress.png", dpi=150)
    print("Figura salva em outputs/modality_stress.png")


if __name__ == "__main__":
    main()