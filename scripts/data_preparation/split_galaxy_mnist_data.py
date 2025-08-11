#!/usr/bin/env python3
import shutil
import random
from pathlib import Path

script_dir = Path(__file__).parent
root = script_dir.parent.parent.parent
data_path = root / "basicsr" / "datasets" / "galaxy_mnist"

def split_dataset():
    # Path to train/gt folder
    print(f"Data path: {data_path}")
    train_gt_path = data_path / "train" / "gt"
    train_lq_path = data_path / "train" / "lq"
    
    # Create val and test directories
    val_gt_path = data_path / "val" / "gt"
    val_lq_path = data_path / "val" / "lq"
    test_gt_path = data_path / "test" / "gt"
    test_lq_path = data_path / "test" / "lq"
    
    val_gt_path.mkdir(parents=True, exist_ok=True)
    val_lq_path.mkdir(parents=True, exist_ok=True)
    test_gt_path.mkdir(parents=True, exist_ok=True)
    test_lq_path.mkdir(parents=True, exist_ok=True)
    
    # Get all files in train/gt
    all_files = list(train_gt_path.glob("*.jpg"))
    
    print(f"Found {len(all_files)} files in train/gt")
    
    if len(all_files) < 1000:
        print(f"Warning: Only {len(all_files)} files available, using all of them")
        selected_files = all_files
    else:
        selected_files = random.sample(all_files, 1000)
    
    val_files = selected_files[:500]
    test_files = selected_files[500:]
    
    # Move files to val
    print(f"Moving {len(val_files)} files to val/gt and val/lq...")
    for file in val_files:
        # Move GT file
        shutil.move(str(file), str(val_gt_path / file.name))
        # Move corresponding LQ file
        lq_file = train_lq_path / file.name
        if lq_file.exists():
            shutil.move(str(lq_file), str(val_lq_path / lq_file.name))
        else:
            print(f"Warning: LQ file not found for {file.name}")
    
    # Move files to test
    print(f"Moving {len(test_files)} files to test/gt and test/lq...")
    for file in test_files:
        # Move GT file
        shutil.move(str(file), str(test_gt_path / file.name))
        # Move corresponding LQ file
        lq_file = train_lq_path / file.name
        if lq_file.exists():
            shutil.move(str(lq_file), str(test_lq_path / lq_file.name))
        else:
            print(f"Warning: LQ file not found for {file.name}")
    
    # Count remaining files
    remaining_train_gt = len(list(train_gt_path.glob("*.jpg")))
    remaining_train_lq = len(list(train_lq_path.glob("*.jpg")))
    val_count_gt = len(list(val_gt_path.glob("*.jpg")))
    val_count_lq = len(list(val_lq_path.glob("*.jpg")))
    test_count_gt = len(list(test_gt_path.glob("*.jpg")))
    test_count_lq = len(list(test_lq_path.glob("*.jpg")))
    
    print("\nFinal counts:")
    print(f"Train GT: {remaining_train_gt}, Train LQ: {remaining_train_lq}")
    print(f"Val GT: {val_count_gt}, Val LQ: {val_count_lq}")
    print(f"Test GT: {test_count_gt}, Test LQ: {test_count_lq}")
    print("Done!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    split_dataset()