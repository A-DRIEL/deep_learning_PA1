import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import ellipse


class SyntheticEllipseDataset(Dataset):
    """
    Dataset sintético de elipses para o teste unitário (Parte 0).

    Cada amostra contém 5-20 elipses de tamanhos/posições/rotações
    aleatórias, muitas delas adjacentes.

    Imagem de saída: (3, H, W) float em [0, 1]

    Máscara de saída: (H, W) long = ID da
    instância (0 = fundo, 1..N = instâncias).
    """

    def __init__(self, size=128, min_ellipses=5, max_ellipses=20,
                 num_samples=200, seed=None):
        self.size = size
        self.min_ellipses = min_ellipses
        self.max_ellipses = max_ellipses
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        base_seed = self.seed if self.seed is not None else 0
        rng = np.random.default_rng(base_seed + idx)

        image, instance_mask = self._generate_sample(rng)

        # (H, W) -> (3, H, W), replicando o canal
        image_t = torch.from_numpy(image).float().unsqueeze(0).repeat(3, 1, 1)
        mask_t = torch.from_numpy(instance_mask).long()
        return image_t, mask_t

    def _generate_sample(self, rng):
        H = W = self.size

        background_level = rng.uniform(0.05, 0.25)
        image = np.full((H, W), background_level, dtype=np.float32)

        instance_mask = np.zeros((H, W), dtype=np.int32)

        n_ellipses = rng.integers(self.min_ellipses, self.max_ellipses + 1)

        next_id = 1
        for _ in range(n_ellipses):
            cy = rng.uniform(0, H)
            cx = rng.uniform(0, W)
            r_major = rng.uniform(6, 18)
            r_minor = rng.uniform(4, 14)
            rotation = rng.uniform(0, np.pi)

            rr, cc = ellipse(
                cy, cx, r_major, r_minor,
                shape=(H, W), rotation=rotation
            )

            if rr.size == 0:
                continue

            
            free = instance_mask[rr, cc] == 0
            rr, cc = rr[free], cc[free]

            if rr.size < 10:
                continue

            instance_mask[rr, cc] = next_id

            object_level = rng.uniform(0.55, 0.95)
            image[rr, cc] = object_level

            next_id += 1

        noise_sigma = rng.uniform(0.02, 0.08)
        noise = rng.normal(0, noise_sigma, size=(H, W)).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)

        return image, instance_mask