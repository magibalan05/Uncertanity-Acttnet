import os
import sys
import glob
import time
import numpy as np
import torch
import nibabel as nib

def prepare_joint_dataset():
    cache_dir = "c:/final year project/actt net/processed_cache"
    os.makedirs(cache_dir, exist_ok=True)

    gli_data_dir = "c:/final year project/actt net/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    men_data_dir = "c:/final year project/actt net/ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData/BraTS-MEN-Train"

    gli_patients = sorted([
        ("GLI", os.path.join(gli_data_dir, d)) 
        for d in os.listdir(gli_data_dir)
        if os.path.isdir(os.path.join(gli_data_dir, d))
    ]) if os.path.exists(gli_data_dir) else []

    men_patients = sorted([
        ("MEN", os.path.join(men_data_dir, d)) 
        for d in os.listdir(men_data_dir)
        if os.path.isdir(os.path.join(men_data_dir, d))
    ]) if os.path.exists(men_data_dir) else []

    print("=" * 75, flush=True)
    print(f"[*] Joint Dataset Preprocessor (GLI + MEN)", flush=True)
    print(f"    - Glioma (GLI) Patients:     {len(gli_patients)}", flush=True)
    print(f"    - Meningioma (MEN) Patients: {len(men_patients)}", flush=True)
    print("=" * 75, flush=True)

    # Process first 200 GLI patients for high density slice representation + existing MEN cache
    target_gli = gli_patients[:250]
    
    count = 0
    skipped = 0
    start = time.time()

    def norm(img):
        m = img > 0
        return (img - img[m].mean()) / (img[m].std() + 1e-8) if m.any() else img

    for idx, (tag, p_dir) in enumerate(target_gli, 1):
        p_id = os.path.basename(p_dir)
        
        # Check if already preprocessed
        check_file1 = os.path.join(cache_dir, f"{p_id}_slice_45.pt")
        check_file2 = os.path.join(cache_dir, f"{tag}_{p_id}_slice_45.pt")
        if os.path.exists(check_file1) or os.path.exists(check_file2):
            skipped += 1
            continue

        t1c_path = os.path.join(p_dir, f"{p_id}-t1c.nii.gz")
        t1n_path = os.path.join(p_dir, f"{p_id}-t1n.nii.gz")
        t2w_path = os.path.join(p_dir, f"{p_id}-t2w.nii.gz")
        t2f_path = os.path.join(p_dir, f"{p_id}-t2f.nii.gz")
        seg_path = os.path.join(p_dir, f"{p_id}-seg.nii.gz")

        if not (os.path.exists(t1c_path) and os.path.exists(seg_path)):
            continue

        try:
            t1c_vol = nib.load(t1c_path).get_fdata(dtype=np.float32)
            t1n_vol = nib.load(t1n_path).get_fdata(dtype=np.float32)
            t2w_vol = nib.load(t2w_path).get_fdata(dtype=np.float32)
            t2f_vol = nib.load(t2f_path).get_fdata(dtype=np.float32)
            seg_vol = nib.load(seg_path).get_fdata(dtype=np.float32)

            for slice_idx in range(45, 115, 4):
                seg_slice = seg_vol[:, :, slice_idx]
                mask = (seg_slice > 0).astype(np.float32)

                t1c_s = norm(t1c_vol[:, :, slice_idx])
                t1n_s = norm(t1n_vol[:, :, slice_idx])
                t2w_s = norm(t2w_vol[:, :, slice_idx])
                t2f_s = norm(t2f_vol[:, :, slice_idx])

                img_tensor = torch.from_numpy(np.stack([t1c_s, t1n_s, t2w_s, t2f_s], axis=0))
                mask_tensor = torch.from_numpy(np.expand_dims(mask, axis=0))

                img_tensor = torch.nn.functional.interpolate(img_tensor.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)
                mask_tensor = torch.nn.functional.interpolate(mask_tensor.unsqueeze(0), size=(128, 128), mode='nearest').squeeze(0)

                save_file = os.path.join(cache_dir, f"{p_id}_slice_{slice_idx}.pt")
                torch.save({'image': img_tensor, 'mask': mask_tensor, 'pid': p_id}, save_file)
                count += 1

        except Exception as e:
            continue

        if idx % 25 == 0 or idx == len(target_gli):
            pct = (idx / len(target_gli)) * 100
            print(f" -> [{idx:03d}/{len(target_gli):03d}] ({pct:5.1f}%) GLI Slices Processed: {count} (Skipped: {skipped})", flush=True)

    total_cached = len(glob.glob(os.path.join(cache_dir, "*.pt")))
    print(f"\n[+] Preprocessing Complete! {count} GLI tensor slices added in {time.time() - start:.1f}s.", flush=True)
    print(f"[+] Total Combined (GLI + MEN) Cached Tensor Slices: {total_cached}", flush=True)

if __name__ == "__main__":
    prepare_joint_dataset()
