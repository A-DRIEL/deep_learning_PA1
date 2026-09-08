import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed


def decode_watershed(probs, interior_thresh=0.5, foreground_thresh=0.5):
    """
    Converte a predição de 3 classes (após softmax) em uma máscara de
    instâncias, separando núcleos encostados.

    probs: array (3, H, W) com probabilidades [fundo, interior, borda].
    Retorna: instance_mask (H, W) int32, 0 = fundo, 1..N = instâncias.
    """
    interior_prob, border_prob = probs[1], probs[2]

    # 1) marcadores = componentes conexos do canal 'interior' binarizado
    interior_bin = interior_prob > interior_thresh
    markers, n_markers = ndi.label(interior_bin)
    if n_markers == 0:
        return np.zeros_like(interior_bin, dtype=np.int32)

    # 2) mapa de elevação a partir da borda:
    #    quanto mais 'borda', mais alto -> os núcleos (vales) ficam separados
    elevation = border_prob

    # 3) restringe ao foreground (interior + borda, excluindo fundo)
    foreground = (interior_prob + border_prob) > foreground_thresh

    instance_mask = watershed(elevation, markers=markers, mask=foreground)
    return instance_mask.astype(np.int32)