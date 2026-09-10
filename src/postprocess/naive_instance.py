# src/postprocess/naive_instance.py

import numpy as np
from scipy import ndimage


def extract_instances_naive(binary_mask):
    """
    Extração de instância "ingênua": threshold + componentes conexos.


    binary_mask: array (H, W) booleano ou {0,1} 

    Retorna: array (H, W) int32 com IDs de instância (0 = fundo)
    """
    
    instance_mask, num_instances = ndimage.label(binary_mask)
    return instance_mask.astype(np.int32), num_instances