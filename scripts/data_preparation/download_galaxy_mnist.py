import os
from pathlib import Path
from galaxy_datasets import galaxy_mnist

script_dir = Path(__file__)
root = script_dir.parent.parent.parent  # Go up to the root of the project
data_path = root / 'basicsr' / 'datasets' / 'galaxy_mnist' / 'train'
data_path.mkdir(parents=True, exist_ok=True)

catalog, label_cols = galaxy_mnist(
    root=data_path,
    train=True,
    download=True
)

images_dir = data_path / 'images'
for file_path in images_dir.iterdir():
    file_path.rename(data_path / file_path.name)

os.rmdir(images_dir)  # Remove the empty images directory