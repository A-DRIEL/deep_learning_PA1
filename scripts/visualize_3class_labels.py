import matplotlib.pyplot as plt
from pathlib import Path

from src.datasets.labels_3class import generate_three_class_label

root = Path("data/raw/stage1_train")
sample_dirs = sorted(root.iterdir())[:4]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, sample_dir in zip(axes, sample_dirs):
    label = generate_three_class_label(sample_dir / "masks", border_width=1)
    ax.imshow(label, cmap="viridis", vmin=0, vmax=2)
    ax.set_title(sample_dir.name[:8])
    ax.axis("off")
plt.tight_layout()
plt.show()
#plt.savefig("three_class_labels_preview.png", dpi=120)