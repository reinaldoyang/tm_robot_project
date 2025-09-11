import os
import shutil
import random


def split_dataset():
    # Paths
    dataset_dir = "rlds_dataset_npy"      # your current folder with all .npy episodes
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # Create folders if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get all .npy files
    all_files = [f for f in os.listdir(dataset_dir) if f.endswith(".npy")]

    # Shuffle randomly
    random.shuffle(all_files)

    # Split index
    split_idx = int(len(all_files) * 0.8)  # 80% train

    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # Move files to respective folders
    for f in train_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(train_dir, f))

    for f in val_files:
        shutil.move(os.path.join(dataset_dir, f), os.path.join(val_dir, f))

    print(f"Total episodes: {len(all_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
