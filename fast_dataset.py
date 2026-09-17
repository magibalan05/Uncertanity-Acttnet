import os
import glob
import torch
from torch.utils.data import Dataset

class FastCachedBraTSDataset(Dataset):
    """
    Ultra-Fast In-Memory / Pre-cached PyTorch Dataset.
    Loads pre-processed `.pt` tensor files instantly into RAM / GPU VRAM.
    """
    def __init__(self, cache_dir="c:/final year project/actt net/processed_cache"):
        self.cache_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if len(self.cache_files) == 0:
            raise RuntimeError(f"No cached tensor files found in {cache_dir}. Run `python preprocess.py` first.")

    def __len__(self):
        return len(self.cache_files)

    def __getitem__(self, idx):
        data = torch.load(self.cache_files[idx], weights_only=True)
        return data['image'], data['mask'], data['pid']
