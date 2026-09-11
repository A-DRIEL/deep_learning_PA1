import numpy as np

from src.postprocess.naive_instance import extract_instances_naive


def stitch_with_fusion(tiles, mosaic_shape, iou_threshold=0.3):
    """
    Costura com fusão: ao escrever um novo tile, verifica se algum
    fragmento novo tem sobreposição significativa (IoU) com um
    fragmento JÁ escrito na região de overlap. Se tiver, reutiliza o
    ID existente em vez de criar um novo -- unindo o objeto partido.
    """
    H, W = mosaic_shape
    stitched = np.zeros((H, W), dtype=np.int32)
    next_id = 1

    for tile in tiles:
        y0, y1, x0, x1 = tile["y0"], tile["y1"], tile["x0"], tile["x1"]
        pred_instances, n = extract_instances_naive(tile["pred_binary"])

        existing_region = stitched[y0:y1, x0:x1]  # o que já foi escrito ali antes
        remapped = np.zeros_like(pred_instances)

        for local_id in range(1, n + 1):
            new_fragment = (pred_instances == local_id)

            # verifica sobreposição com QUALQUER instância já existente nessa região
            overlapping_ids = np.unique(existing_region[new_fragment])
            overlapping_ids = overlapping_ids[overlapping_ids != 0]

            best_id, best_iou = None, 0.0
            for existing_id in overlapping_ids:
                existing_fragment_global = (stitched == existing_id)
                intersection = np.logical_and(
                    new_fragment, existing_fragment_global[y0:y1, x0:x1]
                ).sum()
                union = np.logical_or(
                    new_fragment, existing_fragment_global[y0:y1, x0:x1]
                ).sum()
                iou = intersection / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_id = iou, existing_id

            if best_id is not None and best_iou >= iou_threshold:
                remapped[new_fragment] = best_id  # funde: reusa ID existente
            else:
                remapped[new_fragment] = next_id  # objeto novo, nunca visto antes
                next_id += 1

        write_mask = remapped > 0
        stitched[y0:y1, x0:x1][write_mask] = remapped[write_mask]

    return stitched