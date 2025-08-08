import os
from pathlib import Path
from galaxy_datasets import galaxy_mnist

script_dir = Path(__file__)
root = script_dir.parent.parent.parent  # Go up to the root of the project

def download_data(phase='train'):
    """Download Galaxy MNIST dataset."""
    data_path = root / 'basicsr' / 'datasets' / 'galaxy_mnist' / phase
    data_path.mkdir(parents=True, exist_ok=True)

    # if the dataset is already downloaded, skip
    if not data_path.exists():
        catalog, label_cols = galaxy_mnist(
            root=data_path,
            train=(phase == 'train'),
            download=True
        )

        images_dir = data_path / 'images'
        for file_path in images_dir.iterdir():
            file_path.rename(data_path / file_path.name)

        os.rmdir(images_dir)  # Remove the empty images directory

if __name__ == '__main__':
    # download_data('train')
    download_data('gt')