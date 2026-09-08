from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_dilation


def load_individual_masks(masks_dir):
    """Carrega cada máscara individual de um núcleo como array booleano (H, W)."""
    mask_paths = sorted(Path(masks_dir).glob("*.png"))
    return [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0 for p in mask_paths]


def generate_three_class_label(masks_dir, border_width=2):
    """
    Gera um mapa de rótulos (H, W) com:
        0 = fundo
        1 = interior do núcleo
        2 = borda (contato entre núcleos VIZINHOS)

    Método: dilata cada máscara individualmente por `border_width` pixels.
    Onde duas dilatações de núcleos DIFERENTES se sobrepõem, e o pixel
    pertence de fato a algum núcleo, é marcado como borda.
    """
    masks = load_individual_masks(masks_dir)
    if not masks:
        raise ValueError(f"Nenhuma máscara encontrada em {masks_dir}")

    H, W = masks[0].shape
    struct = np.ones((3, 3), dtype=bool)  # 1 iteração ≈ 1 px de raio

    mask_union = np.zeros((H, W), dtype=bool)
    overlap_count = np.zeros((H, W), dtype=np.int32)

    for m in masks:
        mask_union |= m
        dilated = binary_dilation(m, structure=struct, iterations=border_width)
        overlap_count += dilated.astype(np.int32)

    # pixel de borda: contido em >=2 dilatações de núcleos distintos
    # E é pixel que realmente pertence a algum núcleo (não é só "espaço vazio" entre eles)
    border = (overlap_count >= 2) & mask_union
    interior = mask_union & (~border)

    label = np.zeros((H, W), dtype=np.int64)
    label[interior] = 1
    label[border] = 2
    return label