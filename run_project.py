import os
import sys

# Test pure python dataset loading and execution
print("=" * 60)
print("Executing HQU-MSANet Training Runner via Python Direct Invocation")
print("=" * 60)

# Import dataset, model, metrics
from dataset import BraTS2023Dataset
from model import HQUMSNet
from train import train_hqu_msanet

if __name__ == "__main__":
    train_hqu_msanet()
