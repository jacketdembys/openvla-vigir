import os
import numpy as np
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
# Directory containing the episode folders and .npy files
SOURCE_DIR = '/home/Desktop/Saved_Episodes'
# Output directory (edit this as needed)
OUTPUT_DIR = '/home/Desktop/Saved_Episodes_RLDS_compatible'
# Train/val split ratio
TRAIN_RATIO = 0.8

# === SCRIPT ===
def collect_npy_files(root_dir):
    """Recursively collect all .npy files under root_dir."""
    npy_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.npy'):
                npy_files.append(os.path.join(dirpath, fname))
    return npy_files

def split_files(files, train_ratio):
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]

def copy_and_rename(files, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for idx, src in tqdm(enumerate(files)):
        dst = os.path.join(out_dir, f'episode_{idx}.npy')
        shutil.copy2(src, dst)

def main():
    npy_files = collect_npy_files(SOURCE_DIR)
    print(f"Found {len(npy_files)} .npy episode files.")
    train_files, val_files = split_files(npy_files, TRAIN_RATIO)
    print(f"Splitting into {len(train_files)} train and {len(val_files)} val episodes.")

    train_dir = os.path.join(OUTPUT_DIR, 'train')
    val_dir = os.path.join(OUTPUT_DIR, 'val')
    copy_and_rename(train_files, train_dir)
    copy_and_rename(val_files, val_dir)
    print(f"Dataset created at {OUTPUT_DIR}.")
    print(f"Train episodes: {train_dir}")
    print(f"Val episodes: {val_dir}")

if __name__ == '__main__':
    main()
