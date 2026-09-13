# src/datasets/dsb2018.py

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DSB2018Dataset(Dataset):
    """
    Dataset DSB2018 núcleos de microscopia com
    máscara individual por instância.

    Espera a estrutura padrão do Kaggle:
        root_dir/
            <image_id>/
                images/<image_id>.png
                masks/<mask_hash_1>.png
                masks/<mask_hash_2>.png
                ...

    Imagem de saída: (3, H, W) float em [0, 1].
    Máscara de saída: (H, W) long -- IDs de instância (0 = fundo).

    Ambas redimensionadas para `target_size`, com interpolação
    apropriada para cada caso (bilinear para imagem, nearest para
    máscara.
    """

    def __init__(self, root_dir, target_size=(128, 128), normalize="fixed"):
        """
        normalize:
            "fixed"      -- comportamento original: divide por 255.0
            "percentile" -- estica os percentis 1-99 de CADA imagem
                            individualmente para a faixa [0, 1]. Ajuda
                            com imagens de baixo contraste ou modalidade
                            muito diferente do padrão do dataset.
        """
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.normalize = normalize
        self.sample_dirs = sorted(
            d for d in self.root_dir.iterdir() if d.is_dir()
        )
        if len(self.sample_dirs) == 0:
            raise ValueError(f"Nenhuma subpasta de imagem encontrada em {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        image_id = sample_dir.name

        image = self._load_image(sample_dir, image_id)
        instance_mask = self._load_instance_mask(sample_dir)

        image_resized = self._resize_image(image)
        mask_resized = self._resize_mask(instance_mask)

        image_t = torch.from_numpy(image_resized).float().permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_resized).long()
        return image_t, mask_t

    def _load_image(self, sample_dir, image_id):
        image_path = sample_dir / "images" / f"{image_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32)

        if self.normalize == "percentile":
            p1, p99 = np.percentile(image, [1, 99])
            if p99 > p1:  # evita divisão por zero em imagens totalmente uniformes
                image = np.clip((image - p1) / (p99 - p1), 0.0, 1.0)
            else:
                image = image / 255.0
        else:  # "fixed", comportamento original
            image = image / 255.0

        return image

    def _load_instance_mask(self, sample_dir):
        masks_dir = sample_dir / "masks"
        mask_paths = sorted(masks_dir.glob("*.png"))  

        # descobre dimensões a partir do primeiro arquivo de máscara
        first_mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
        H, W = first_mask.shape

        instance_mask = np.zeros((H, W), dtype=np.int32)

        for next_id, mask_path in enumerate(mask_paths, start=1):
            binary_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            instance_mask[binary_mask > 0] = next_id

        return instance_mask

    def _resize_image(self, image):
        target_h, target_w = self.target_size
        return cv2.resize(
            image, (target_w, target_h), interpolation=cv2.INTER_AREA
        )

    def _resize_mask(self, mask):
        target_h, target_w = self.target_size
        return cv2.resize(
            mask.astype(np.int32), (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )