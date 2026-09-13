# scripts/failure_gallery_final.py

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.dsb2018_3class import DSB2018ThreeClassDataset
from src.models.resunet import ResUNet
from src.postprocessing.watershed_decode import decode_watershed
from src.metrics.instance_matching import mean_average_precision, count_error


FAILURE_CASES = {
    65:  "Modalidade: citologia (fundo claro, coloração invertida)",
    296: "Modalidade: histologia colunar (textura de tecido denso)",
    602: "Núcleo minúsculo, baixo contraste, ruído de fundo alto",
    386: "Fusão de núcleos mesmo em baixíssima densidade",
    244: "Fusão densa 'clássica' (aglomerado de núcleos)",
}


def plot_single_case(idx, img_np, gt_np, pred_instances, border_prob, score, err, out_dir):
    """Gera uma figura separada (1x4) para uma única imagem de falha -- pronta para um slide."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

    axes[0].imshow(img_np)
    axes[0].set_title("Imagem")
    axes[1].imshow(gt_np, cmap="nipy_spectral")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_instances, cmap="nipy_spectral")
    axes[2].set_title("Predição (Trilha A)")
    axes[3].imshow(border_prob, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title("Mapa de borda (intermediário)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"idx={idx}  |  mAP={score:.3f}  |  erro de contagem={err}", fontsize=12)
    plt.tight_layout()

    out_path = out_dir / f"failure_case_{idx}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_instances = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    dataset_3class = DSB2018ThreeClassDataset("data/raw/stage1_train", target_size=(128, 128))

    model = ResUNet(in_channels=3, num_classes=3, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_3class.pt", map_location=device))
    model.eval()

    indices = list(FAILURE_CASES.keys())

    out_dir = Path("outputs/failure_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)))

    print(f"{'idx':>6} | {'mAP':>6} | {'erro cont.':>10} | causa hipotetizada")
    print("-" * 80)

    for row, idx in enumerate(indices):
        image, _ = dataset_3class[idx]
        _, gt_instances = dataset_instances[idx]
        gt_np = gt_instances.numpy()

        with torch.no_grad():
            logits = model(image.unsqueeze(0).to(device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_instances = decode_watershed(probs)
        border_prob = probs[2]

        score = mean_average_precision(pred_instances, gt_np)
        err = count_error(pred_instances, gt_np)

        img_np = image.permute(1, 2, 0).numpy()

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f"idx={idx}\nmAP={score:.3f}",
                                  fontsize=10, rotation=0, ha="right", va="center")
        axes[row, 1].imshow(gt_np, cmap="nipy_spectral")
        axes[row, 2].imshow(pred_instances, cmap="nipy_spectral")
        axes[row, 3].imshow(border_prob, cmap="hot", vmin=0, vmax=1)

        if row == 0:
            axes[row, 0].set_title("Imagem")
            axes[row, 1].set_title("Ground truth")
            axes[row, 2].set_title("Predição (Trilha A)")
            axes[row, 3].set_title("Mapa de borda (intermediário)")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        out_path = plot_single_case(idx, img_np, gt_np, pred_instances, border_prob, score, err, out_dir)

        print(f"{idx:>6} | {score:>6.3f} | {err:>10} | {FAILURE_CASES[idx]}")
        print(f"         -> figura individual salva em {out_path}")

    plt.figure(fig.number)
    plt.tight_layout()
    plt.savefig("outputs/failure_gallery_final.png", dpi=120)
    print("\nFigura combinada salva em outputs/failure_gallery_final.png")
    print(f"Figuras individuais salvas em {out_dir}/")


if __name__ == "__main__":
    main()