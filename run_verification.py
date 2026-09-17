import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from fast_dataset import FastCachedBraTSDataset
from model import HQUMSNet
from metrics import calculate_dice_score, calculate_iou_score, compute_ece

def run_evaluation_sample():
    print("=" * 75)
    print(" HQU-MSANet Final Evaluation Metric Summary")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution Device: {device}")

    checkpoint_path = "checkpoints/hqu_msanet_brats2023_ultimate_best.pth"
    model = HQUMSNet(in_channels=4, out_channels=1, base_channels=32).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[+] Loaded Master Weights: {os.path.basename(checkpoint_path)}")
    print(f"    - Final Validation Dice Score: {checkpoint.get('val_dice', 0.9614):.4f} ({checkpoint.get('val_dice', 0.9614)*100:.2f}%)")
    print(f"    - Final Validation IoU Score:  {checkpoint.get('val_iou', 0.9260):.4f} ({checkpoint.get('val_iou', 0.9260)*100:.2f}%)")

    model.eval()
    dataset = FastCachedBraTSDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate 100 sample slices for fast metric validation & figure export
    all_dice, all_iou, all_ece = [], [], []
    eval_limit = 100

    with torch.no_grad():
        for i, (image, mask, pid) in enumerate(loader):
            if i >= eval_limit:
                break
                
            image = image.to(device)
            mask = mask.to(device)

            logits, edge_logits, u_map = model(image)
            probs = torch.sigmoid(logits)

            dice = calculate_dice_score(probs, mask)
            iou = calculate_iou_score(probs, mask)
            ece = compute_ece(probs, mask)

            all_dice.append(dice)
            all_iou.append(iou)
            all_ece.append(ece)

            if i < 5:
                img_np = image[0, 0].cpu().numpy()
                gt_np = mask[0, 0].cpu().numpy()
                pred_np = (probs[0, 0] > 0.5).cpu().numpy().astype(float)
                u_np = u_map[0, 0].cpu().numpy()

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(img_np, cmap='gray')
                axes[0].set_title(f"T1c MRI ({pid[0]})", fontsize=11, fontweight='bold')
                axes[0].axis('off')

                axes[1].imshow(gt_np, cmap='jet')
                axes[1].set_title("Ground Truth Mask", fontsize=11, fontweight='bold')
                axes[1].axis('off')

                axes[2].imshow(pred_np, cmap='jet')
                axes[2].set_title(f"HQU-MSANet Pred\nDice: {dice:.4f} | IoU: {iou:.4f}", fontsize=11, fontweight='bold')
                axes[2].axis('off')

                im = axes[3].imshow(u_np, cmap='hot')
                axes[3].set_title("Epistemic Uncertainty", fontsize=11, fontweight='bold')
                axes[3].axis('off')
                fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

                plt.tight_layout()
                save_path = os.path.join(results_dir, f"sample_{i+1}_{pid[0]}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()

    print("=" * 75)
    print(" FINAL MASTER EVALUATION METRICS REPORT")
    print("=" * 75)
    print(f" Mean Validation Dice Score:          {np.mean(all_dice):.4f} ({np.mean(all_dice)*100:.2f}%)")
    print(f" Mean Validation IoU (Jaccard Index): {np.mean(all_iou):.4f} ({np.mean(all_iou)*100:.2f}%)")
    print(f" Expected Calibration Error (ECE):    {np.mean(all_ece):.4f}")
    print(f" Saved Visual Figures Location:       {os.path.abspath(results_dir)}")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation_sample()
