# src/postprocess/mosaic.py

import numpy as np
import torch


def build_mosaic(dataset, indices, grid=(2, 2)):
    """
    Monta um mosaico colando `len(indices)` imagens do dataset em uma
    grade (rows, cols). Cada imagem já deve ter o mesmo tamanho (H, W).

    Retorna:
        mosaic_image: (3, H*rows, W*cols) float
        mosaic_gt: (H*rows, W*cols) long -- IDs de instância re-numerados
                   para não colidir entre as imagens originais
    """
    rows, cols = grid
    assert len(indices) == rows * cols

    images, masks = [], []
    for idx in indices:
        img, mask = dataset[idx]
        images.append(img)
        masks.append(mask)

    _, H, W = images[0].shape
    mosaic_image = torch.zeros(3, H * rows, W * cols)
    mosaic_gt = torch.zeros(H * rows, W * cols, dtype=torch.long)

    next_id = 1
    for i, (img, mask) in enumerate(zip(images, masks)):
        r, c = divmod(i, cols)
        y0, y1 = r * H, (r + 1) * H
        x0, x1 = c * W, (c + 1) * W

        mosaic_image[:, y0:y1, x0:x1] = img

        # re-numera IDs desta sub-imagem para não colidir com as outras
        local_mask = mask.clone()
        local_ids = torch.unique(local_mask)
        local_ids = local_ids[local_ids != 0]
        remapped = torch.zeros_like(local_mask)
        for lid in local_ids:
            remapped[local_mask == lid] = next_id
            next_id += 1
        mosaic_gt[y0:y1, x0:x1] = remapped

    return mosaic_image, mosaic_gt