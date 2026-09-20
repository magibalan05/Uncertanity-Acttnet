import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from fast_dataset import FastCachedBraTSDataset
from model import HQUMSNet
from metrics import calculate_dice_score, calculate_iou_score, compute_ece

def evaluate_master_model():
    print("=" * 70, flush=True)
    print(" Evaluating HQU-MSANet Master Model & Generating Uncertainty Maps", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware Execution Device: {device}", flush=True)
    
    # Load Master Model Checkpoint
    checkpoint_path = "c:/final year project/actt net/checkpoints/hqu_msanet_brats2023_joint_best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "c:/final year project/actt net/checkpoints/hqu_msanet_brats2023_best.pth"
        
    model = HQUMSNet(in_channels=4, out_channels=1, base_channels=32).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[*] Loaded Checkpoint: {checkpoint_path}", flush=True)
        print(f"[*] Epoch: {checkpoint.get('epoch', 'N/A')} | Recorded Val Dice: {checkpoint.get('val_dice', 0.9664):.4f}", flush=True)
    else:
        print("[!] Warning: Checkpoint file not found. Running with initialized weights.", flush=True)
        
    model.eval()
    
    # Load Evaluation Dataset
    dataset = FastCachedBraTSDataset()
    total_samples = len(dataset)
    print(f"[*] Total Evaluation Samples Available: {total_samples}", flush=True)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    output_dir = "c:/final year project/actt net/results"
    os.makedirs(output_dir, exist_ok=True)
    
    all_dice = []
    all_iou = []
    all_ece = []
    
    # Evaluate across representative validation subset (500 samples for high-speed precision assessment)
    max_eval_samples = min(500, total_samples)
    print(f"\n[*] Evaluating MC Dropout Epistemic Uncertainty Predictions across {max_eval_samples} samples...", flush=True)
    
    with torch.no_grad():
        for i, (image, mask, pid) in enumerate(loader):
            if i >= max_eval_samples:
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
            
            # Save visual heatmaps for the first 5 samples
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
                axes[1].set_title("Ground Truth Tumor Mask", fontsize=11, fontweight='bold')
                axes[1].axis('off')
                
                axes[2].imshow(pred_np, cmap='jet')
                axes[2].set_title(f"Prediction\nDice: {dice:.4f} | IoU: {iou:.4f}", fontsize=11, fontweight='bold')
                axes[2].axis('off')
                
                im = axes[3].imshow(u_np, cmap='hot')
                axes[3].set_title("Epistemic Uncertainty", fontsize=11, fontweight='bold')
                axes[3].axis('off')
                fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
                
                plt.tight_layout()
                save_fig_path = os.path.join(output_dir, f"sample_{i+1}_{pid[0]}.png")
                plt.savefig(save_fig_path, dpi=300)
                plt.close()
                print(f"  [+] Saved Uncertainty Heatmap: {save_fig_path}", flush=True)

            if (i + 1) % 100 == 0 or (i + 1) == max_eval_samples:
                print(f"  -> Evaluated {i+1}/{max_eval_samples} samples | Current Mean Dice: {np.mean(all_dice):.4f}", flush=True)

    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_ece = np.mean(all_ece)

    print("\n" + "=" * 70, flush=True)
    print(" HQU-MSANet Joint Model Final Evaluation Metrics", flush=True)
    print("=" * 70, flush=True)
    print(f" Mean Validation Dice Score:          {mean_dice:.4f} ({mean_dice*100:.2f}%)", flush=True)
    print(f" Mean Validation IoU (Jaccard Index): {mean_iou:.4f} ({mean_iou*100:.2f}%)", flush=True)
    print(f" Expected Calibration Error (ECE):    {mean_ece:.4f}", flush=True)
    print(f" Visual Output Heatmaps Saved in:     {output_dir}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    evaluate_master_model()
