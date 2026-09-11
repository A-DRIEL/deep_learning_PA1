# scripts/demonstrate_tiling_problem.py

import torch
import matplotlib.pyplot as plt

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet
from src.postprocess.mosaic import build_mosaic
from src.postprocess.tiled_inference import tiled_inference
from src.postprocess.tiled_instance_naive import stitch_naive
from src.metrics.instance_matching import mean_average_precision, count_error


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DSB2018Dataset("data/raw/stage1_train", target_size=(128, 128))
    mosaic_image, mosaic_gt = build_mosaic(dataset, indices=[0, 100, 300, 500], grid=(2, 2))

    model = ResUNet(in_channels=3, num_classes=1, base_channels=32).to(device)
    model.load_state_dict(torch.load("outputs/resunet_baseline.pt", map_location=device))
    model.eval()

    tiles = tiled_inference(model, mosaic_image, tile_size=128, overlap=32, device=device)
    stitched_naive = stitch_naive(tiles, mosaic_shape=(256, 256))

    gt_np = mosaic_gt.numpy()
    map_naive = mean_average_precision(stitched_naive, gt_np)
    err_naive = count_error(stitched_naive, gt_np)

    print(f"Ground truth: {int(mosaic_gt.max())} instâncias")
    print(f"Costura ingênua: {int(stitched_naive.max())} instâncias, mAP={map_naive:.4f}, erro={err_naive}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(gt_np, cmap="nipy_spectral")
    axes[0].set_title(f"Ground truth ({int(mosaic_gt.max())} instâncias)")
    axes[1].imshow(stitched_naive, cmap="nipy_spectral")
    axes[1].set_title(f"Costura ingênua ({int(stitched_naive.max())}, mAP={map_naive:.3f})")

    # marca as linhas de fronteira dos tiles, pra facilitar ver onde o problema ocorre
    for ax in axes:
        ax.axhline(128, color="white", linestyle="--", linewidth=1)
        ax.axvline(128, color="white", linestyle="--", linewidth=1)
        ax.axis("off")

    
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    # recorte ao redor do cruzamento das fronteiras (região mais crítica)
    y_slice = slice(90, 170)
    x_slice = slice(90, 170)

    axes2[0].imshow(gt_np[y_slice, x_slice], cmap="nipy_spectral")
    axes2[0].set_title("Ground truth (zoom na fronteira)")
    axes2[1].imshow(stitched_naive[y_slice, x_slice], cmap="nipy_spectral")
    axes2[1].set_title("Costura ingênua (zoom na fronteira)")
    for ax in axes2:
        ax.axhline(128 - 90, color="white", linestyle="--", linewidth=1)
        ax.axvline(128 - 90, color="white", linestyle="--", linewidth=1)
        ax.axis("off")
    plt.tight_layout()

    plt.savefig("outputs/tiling_problem.png", dpi=150)
    print("\nFigura salva em outputs/tiling_problem.png")


if __name__ == "__main__":
    main()