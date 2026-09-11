# src/postprocess/tiled_inference.py

import numpy as np
import torch


def tiled_inference(model, mosaic_image, tile_size=128, overlap=32, device="cpu"):
    """
    Roda o modelo em tiles sobrepostos sobre uma imagem grande, e devolve
    a predição binária de CADA tile separadamente, com suas coordenadas --
    sem tentar fundir ainda (isso é o passo seguinte).

    mosaic_image: (3, H, W)

    Retorna: lista de dicts {"y0","y1","x0","x1","pred_binary"}
    """
    _, H, W = mosaic_image.shape
    stride = tile_size - overlap

    tiles = []
    y0 = 0
    while y0 < H:
        y1 = min(y0 + tile_size, H)
        y0_adj = max(0, y1 - tile_size)  # garante tile de tamanho fixo na borda

        x0 = 0
        while x0 < W:
            x1 = min(x0 + tile_size, W)
            x0_adj = max(0, x1 - tile_size)

            tile = mosaic_image[:, y0_adj:y1, x0_adj:x1].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tile)
                pred_binary = (torch.sigmoid(logits).squeeze() > 0.5).cpu().numpy()

            tiles.append({
                "y0": y0_adj, "y1": y1, "x0": x0_adj, "x1": x1,
                "pred_binary": pred_binary,
            })

            if x1 >= W:
                break
            x0 += stride
        if y1 >= H:
            break
        y0 += stride

    return tiles