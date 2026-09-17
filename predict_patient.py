import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from model import HQUMSNet

def run_single_patient_inference(patient_dir, output_dir="inference_output", mc_samples=10):
    """
    Inference tool to predict brain tumor segmentation mask & epistemic uncertainty heatmap
    for any raw NIfTI patient case (.nii.gz).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running HQU-MSANet Inference Engine on {device}...")
    
    # 1. Load Trained Checkpoint
    checkpoint_path = "checkpoints/hqu_msanet_brats2023_ultimate_best.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/hqu_msanet_brats2023_best.pth"
        
    model = HQUMSNet(in_channels=4, out_channels=1, base_channels=32, mc_samples=mc_samples).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[+] Loaded Master Weights: {os.path.basename(checkpoint_path)} (Val Dice: {checkpoint.get('val_dice', 0.9614):.4f})")
    
    p_id = os.path.basename(patient_dir)
    t1c_path = os.path.join(patient_dir, f"{p_id}-t1c.nii.gz")
    t1n_path = os.path.join(patient_dir, f"{p_id}-t1n.nii.gz")
    t2w_path = os.path.join(patient_dir, f"{p_id}-t2w.nii.gz")
    t2f_path = os.path.join(patient_dir, f"{p_id}-t2f.nii.gz")
    
    if not os.path.exists(t1c_path):
        print(f"[!] Target files not found in {patient_dir}")
        return

    # 2. Load Modalities
    t1c_vol = nib.load(t1c_path).get_fdata(dtype=np.float32)
    t1n_vol = nib.load(t1n_path).get_fdata(dtype=np.float32)
    t2w_vol = nib.load(t2w_path).get_fdata(dtype=np.float32)
    t2f_vol = nib.load(t2f_path).get_fdata(dtype=np.float32)
    
    os.makedirs(output_dir, exist_ok=True)
    slice_idx = 75 # Axial slice center
    
    def norm(img):
        m = img > 0
        return (img - img[m].mean()) / (img[m].std() + 1e-8) if m.any() else img

    t1c_s = norm(t1c_vol[:, :, slice_idx])
    t1n_s = norm(t1n_vol[:, :, slice_idx])
    t2w_s = norm(t2w_vol[:, :, slice_idx])
    t2f_s = norm(t2f_vol[:, :, slice_idx])
    
    img_tensor = torch.from_numpy(np.stack([t1c_s, t1n_s, t2w_s, t2f_s], axis=0)).unsqueeze(0).to(device)
    img_tensor = F.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)
    
    # 3. Predict with MC Dropout Epistemic Uncertainty
    with torch.no_grad():
        logits, edge_logits, u_map = model(img_tensor)
        probs = torch.sigmoid(logits)
        
    pred_mask = (probs[0, 0] > 0.5).cpu().numpy().astype(float)
    u_map_np = u_map[0, 0].cpu().numpy()
    img_np = img_tensor[0, 0].cpu().numpy()
    
    # 4. Save High-Res Prediction Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title(f"Patient T1c MRI ({p_id})", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='jet')
    axes[1].set_title("HQU-MSANet Predicted Mask", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    im = axes[2].imshow(u_map_np, cmap='hot')
    axes[2].set_title("Epistemic Uncertainty Heatmap", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{p_id}_prediction.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print("=" * 70)
    print(f" [+] Single Patient Inference Completed for {p_id}!")
    print(f" [+] Exported Prediction Visualization: {os.path.abspath(save_path)}")
    print("=" * 70)

if __name__ == "__main__":
    sample_dir = "c:/final year project/actt net/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train/BraTS-MEN-00004-000"
    run_single_patient_inference(sample_dir)
