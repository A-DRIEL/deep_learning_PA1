import torch

from src.datasets.dsb2018 import DSB2018Dataset
from src.datasets.labels_3class import generate_three_class_label


class DSB2018ThreeClassDataset(DSB2018Dataset):
    """
    Reaproveita o carregamento/resize de imagem do DSB2018Dataset (Parte 1),
    mas troca o alvo pela máscara de 3 classes (fundo/interior/borda).

    Saída: (imagem (3,H,W) float, rótulo (H,W) long em {0,1,2})
    """

    def __init__(self, root_dir, target_size=(128, 128), border_width=2):
        super().__init__(root_dir, target_size=target_size)
        self.border_width = border_width

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        image_id = sample_dir.name

        image = self._load_image(sample_dir, image_id)
        label = generate_three_class_label(
            sample_dir / "masks", border_width=self.border_width
        )

        image_resized = self._resize_image(image)
        label_resized = self._resize_mask(label)  # já usa INTER_NEAREST — não mistura classes

        image_t = torch.from_numpy(image_resized).float().permute(2, 0, 1)
        label_t = torch.from_numpy(label_resized).long()
        return image_t, label_t