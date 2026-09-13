import numpy as np
import torch
from pathlib import Path
from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.labels_3class import generate_three_class_label


class DSB2018ThreeClassDataset(DSB2018Dataset):
    def __init__(self, root_dir, target_size=(128, 128), border_width=2, cache_dir="data/cache/3class"):
        super().__init__(root_dir, target_size=target_size)
        self.border_width = border_width
        self.cache_dir = Path(cache_dir) / f"{target_size[0]}x{target_size[1]}_bw{border_width}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        image_id = sample_dir.name
        cache_path = self.cache_dir / f"{image_id}.npz"

        if cache_path.exists():
            data = np.load(cache_path)
            image_resized, label_resized = data["image"], data["label"]
        else:
            image = self._load_image(sample_dir, image_id)
            label = generate_three_class_label(sample_dir / "masks", border_width=self.border_width)
            image_resized = self._resize_image(image)
            label_resized = self._resize_mask(label)
            np.savez_compressed(cache_path, image=image_resized, label=label_resized)

        image_t = torch.from_numpy(image_resized).float().permute(2, 0, 1)
        label_t = torch.from_numpy(label_resized).long()
        return image_t, label_t