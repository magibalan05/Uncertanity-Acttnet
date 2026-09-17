import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from fast_dataset import FastCachedBraTSDataset
from model import HQUMSNet
from metrics import CombinedBoundaryLoss, calculate_dice_score, calculate_iou_score

def train_phase2():
    print("=" * 70)
    print(" HQU-MSANet Phase 2 High-Precision Fine-Tuning Pipeline")
    print(" Advanced Boundary-Aware Refinement & Calibration Tuning")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("[!] ERROR: CUDA/GPU is not available.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"[*] GPU TARGET: {torch.cuda.get_device_name(0)}")
    print(f"[*] CUDA Version: {torch.version.cuda}")

    # Load cached dataset
    dataset = FastCachedBraTSDataset()
    total_samples = len(dataset)
    print(f"[*] Total High-Density Pre-cached Tensor Slices: {total_samples}")

    train_size = int(0.85 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"[*] Training Samples: {len(train_dataset)} | Validation Samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # Model Initialization
    model = HQUMSNet(in_channels=4, out_channels=1, base_channels=32).to(device)
    
    # Load Phase 1 Best Weights if available
    checkpoint_dir = "checkpoints"
    best_model_path = os.path.join(checkpoint_dir, "hqu_msanet_brats2023_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[*] Loaded Phase 1 Weights (Val Dice: {checkpoint['val_dice']:.4f})")
    
    epochs = 15
    # Lower learning rate for fine-tuning stability
    learning_rate = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CombinedBoundaryLoss(lambda_edge=0.35) # Increased boundary edge weighting
    scaler = torch.amp.GradScaler('cuda')

    best_val_dice = checkpoint.get('val_dice', 0.90) if os.path.exists(best_model_path) else 0.0

    print("\n" + "-" * 70)
    print(f" Starting Phase 2 High-Precision Fine-Tuning Loop ({epochs} Epochs)...")
    print("-" * 70)

    start_time = time.time()
    total_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_dice_acc = 0.0
        train_iou_acc = 0.0

        for step, (images, masks, _) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, edge_logits = model(images)
                loss, l_dice, l_bce, l_edge = criterion(logits, edge_logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            probs = torch.sigmoid(logits)
            train_dice_acc += calculate_dice_score(probs, masks)
            train_iou_acc += calculate_iou_score(probs, masks)

            batch_pct = (step / total_batches) * 100
            print(f"\r  [Phase 2 - Epoch {epoch:02d}/{epochs:02d}] Step {step:02d}/{total_batches:02d} ({batch_pct:5.1f}%) | Batch Loss: {loss.item():.4f}", end="", flush=True)

        scheduler.step()

        avg_train_loss = train_loss / total_batches
        avg_train_dice = train_dice_acc / total_batches
        avg_train_iou = train_iou_acc / total_batches

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_dice_acc = 0.0
        val_iou_acc = 0.0
        val_batches = len(val_loader)

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits, edge_logits, u_map = model(images)
                    loss, _, _, _ = criterion(logits, edge_logits, masks)

                val_loss += loss.item()
                probs = torch.sigmoid(logits)
                val_dice_acc += calculate_dice_score(probs, masks)
                val_iou_acc += calculate_iou_score(probs, masks)

        avg_val_loss = val_loss / val_batches
        avg_val_dice = val_dice_acc / val_batches
        avg_val_iou = val_iou_acc / val_batches

        epoch_time = time.time() - epoch_start
        overall_pct = (epoch / epochs) * 100
        print(f"\n[+] Phase 2 Epoch [{epoch:02d}/{epochs:02d}] Completed in {epoch_time:.2f}s | Overall Progress: {overall_pct:5.1f}%")
        print(f"    Train Loss: {avg_train_loss:.4f} | Train Dice: {avg_train_dice:.4f} | Train IoU: {avg_train_iou:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f} | Val Dice:   {avg_val_dice:.4f} | Val IoU:   {avg_val_iou:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice,
                'val_iou': avg_val_iou
            }, best_model_path)
            print(f"    --> [SAVED HIGH ACCURACY MODEL CHECKPOINT] Val Dice: {best_val_dice:.4f} -> {best_model_path}")
        print("-" * 70)

    total_duration = time.time() - start_time
    print("=" * 70)
    print(f" Phase 2 GPU Fine-Tuning Completed Successfully in {total_duration/60:.2f} minutes!")
    print(f" High Precision Validation Dice Coefficient: {best_val_dice:.4f}")
    print(f" Saved Model Checkpoint: {os.path.abspath(best_model_path)}")
    print("=" * 70)

if __name__ == "__main__":
    train_phase2()
