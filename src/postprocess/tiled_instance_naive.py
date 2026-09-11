# src/postprocess/tiled_instance_naive.py

import numpy as np

from src.postprocess.naive_instance import extract_instances_naive


def stitch_naive(tiles, mosaic_shape):
    """
    'Costura' ingênua: cada tile extrai suas próprias instâncias e
    escreve no mosaico final, com IDs re-numerados globalmente --
    SEM nenhuma tentativa de unir fragmentos do mesmo objeto entre
    tiles. É isso que demonstra o problema pedido pelo enunciado.
    """
    H, W = mosaic_shape
    stitched = np.zeros((H, W), dtype=np.int32)
    next_id = 1

    for tile in tiles:
        y0, y1, x0, x1 = tile["y0"], tile["y1"], tile["x0"], tile["x1"]
        pred_instances, n = extract_instances_naive(tile["pred_binary"])

        # onde já foi escrito por um tile anterior (sobreposição),
        # a versão ingênua simplesmente SOBRESCREVE -- last-write-wins
        remapped = pred_instances.copy()
        for local_id in range(1, n + 1):
            remapped[pred_instances == local_id] = next_id + local_id - 1
        next_id += n

        stitched[y0:y1, x0:x1] = np.where(remapped > 0, remapped, stitched[y0:y1, x0:x1])

    return stitched