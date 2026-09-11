import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.datasets.dsb2018 import DSB2018Dataset
from src.models.resunet import ResUNet
from src.postprocess.mosaic import build_mosaic
from src.postprocess.tiled_inference import tiled_inference
from src.postprocess.tiled_instance_naive import stitch_naive
from src.postprocess.tiled_instance_fused import stitch_with_fusion
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
    stitched_fused = stitch_with_fusion(tiles, mosaic_shape=(256, 256))

    gt_np = mosaic_gt.numpy()
    n_gt = int(mosaic_gt.max())

    map_naive = mean_average_precision(stitched_naive, gt_np)
    err_naive = count_error(stitched_naive, gt_np)

    map_fused = mean_average_precision(stitched_fused, gt_np)
    err_fused = count_error(stitched_fused, gt_np)

    print(f"Ground truth: {n_gt} instâncias")
    print(f"Ingênua: {int(stitched_naive.max())} instâncias, mAP={map_naive:.4f}, erro={err_naive}")
    print(f"Fundida: {int(stitched_fused.max())} instâncias, mAP={map_fused:.4f}, erro={err_fused}")

    # --- visão geral, com a região de overlap destacada (sem cobrir pixels) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(gt_np, cmap="nipy_spectral")
    axes[0].set_title(f"Ground truth ({n_gt} instâncias)")
    axes[1].imshow(stitched_naive, cmap="nipy_spectral")
    axes[1].set_title(f"Ingênua ({int(stitched_naive.max())}, mAP={map_naive:.3f})")
    axes[2].imshow(stitched_fused, cmap="nipy_spectral")
    axes[2].set_title(f"Fundida ({int(stitched_fused.max())}, mAP={map_fused:.3f})")

    for ax in axes:
        rect_h = Rectangle((0, 96), 256, 64, linewidth=1.5,
                            edgecolor="red", facecolor="none", linestyle="--")
        rect_v = Rectangle((96, 0), 64, 256, linewidth=1.5,
                            edgecolor="red", facecolor="none", linestyle="--")
        ax.add_patch(rect_h)
        ax.add_patch(rect_v)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/tiling_problem.png", dpi=150)
    print("\nFigura geral salva em outputs/tiling_problem.png")

    # --- zoom na fronteira, comparando os 3 ---
    y_slice = slice(90, 170)
    x_slice = slice(90, 170)

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    axes2[0].imshow(gt_np[y_slice, x_slice], cmap="nipy_spectral")
    axes2[0].set_title("Ground truth")
    axes2[1].imshow(stitched_naive[y_slice, x_slice], cmap="nipy_spectral")
    axes2[1].set_title("Ingênua")
    axes2[2].imshow(stitched_fused[y_slice, x_slice], cmap="nipy_spectral")
    axes2[2].set_title("Fundida")
    for ax in axes2:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/tiling_fusion_comparison.png", dpi=150)
    print("Figura de zoom salva em outputs/tiling_fusion_comparison.png")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()