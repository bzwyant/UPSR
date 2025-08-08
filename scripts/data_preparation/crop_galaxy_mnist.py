#!/usr/bin/env python3

import os
import argparse
from pathlib import Path
import torch
from torchvision.transforms import Compose, ToTensor, CenterCrop, ToPILImage
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image


def setup_gpu():
    """Initialize GPU if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device


def center_crop(tensor_batch, target_size=128):
    """
    Center crop a batch of images on the GPU
    
    Args: 
        tensor_batch (torch.Tensor): Batch of images with shape (N, C, H, W)
        target_size (int): Size of the cropped image (target_size x target_size)
    
    Returns:
        torch.Tensor: Cropped batch of images
    """
    _, _, h, w = tensor_batch.shape
    start = (h - target_size) // 2
    return tensor_batch[..., start:start + target_size, start:start + target_size]


def process_batch(tensor_batch, device):
    tensor_batch = tensor_batch.to(device)
    cropped_128 = center_crop(tensor_batch, target_size=128)
    downsampled_64 = F.avg_pool2d(cropped_128, kernel_size=2, stride=2)
    return cropped_128, downsampled_64


def load_image_as_tensor(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return ToTensor()(img)


def save_tensor_as_image(tensor, save_path):
    img = ToPILImage()(tensor)
    img.save(save_path)


def process_images(input_dir, output_dir_128, output_dir_64, batch_size=32):
    """
    Process images in batches and save cropped and downsampled versions.

    Args:
        input_dir (str): Directory containing input images.
        output_dir_128 (str): Directory to save cropped images.
        output_dir_64 (str): Directory to save downsampled images.
        batch_size (int): Number of images to process in each batch.
    """
    device = setup_gpu()

    output_dir_128 = Path(output_dir_128)
    output_dir_64 = Path(output_dir_64)
    output_dir_128.mkdir(parents=True, exist_ok=True)
    output_dir_64.mkdir(parents=True, exist_ok=True)

    # Get all image files - FIXED: glob returns generator, need list()
    input_path = Path(input_dir)
    image_files = list(input_path.glob("*.jpg"))

    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # FIXED: Use integer division for total_batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size

    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch_files = image_files[start_idx:end_idx]

            # Load batch of images as tensors
            batch_tensors = [load_image_as_tensor(str(img_path)) for img_path in batch_files]
            batch_tensors = torch.stack(batch_tensors)

            # Process entire batch on GPU
            cropped_128, downsampled_64 = process_batch(batch_tensors, device)

            cropped_128_cpu = cropped_128.cpu()
            downsampled_64_cpu = downsampled_64.cpu()

            # Save processed images
            for idx, img_path in enumerate(batch_files):
                save_path_128 = output_dir_128 / img_path.name
                save_tensor_as_image(cropped_128_cpu[idx], save_path_128)

                save_path_64 = output_dir_64 / img_path.name
                save_tensor_as_image(downsampled_64_cpu[idx], save_path_64)
            
            pbar.update(len(batch_files))  # FIXED: Update by batch size

    print(f"\nSaved {len(image_files)} cropped images (128x128) to {output_dir_128}")
    print(f"Saved {len(image_files)} downsampled images (64x64) to {output_dir_64}")
    
    # Final GPU memory status
    print(f"GPU Memory Used: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Process Galaxy MNIST images.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir_128', type=str, required=True, help='Directory to save cropped images (128x128).')
    parser.add_argument('--output_dir_64', type=str, required=True, help='Directory to save downsampled images (64x64).')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images to process in each batch.')

    args = parser.parse_args()

    try:
        process_images(args.input_dir, args.output_dir_128, args.output_dir_64, args.batch_size)
        print("\nProcessing completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()