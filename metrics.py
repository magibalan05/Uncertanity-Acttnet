import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    """
    Dice Loss for binary tumor segmentation.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedBoundaryLoss(nn.Module):
    """
    Hybrid Loss: Dice Loss + Binary Cross Entropy + Boundary Loss
    """
    def __init__(self, alpha=0.5, beta=0.5, lambda_edge=0.25):
        super(CombinedBoundaryLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.lambda_edge = lambda_edge

    def _extract_edges(self, targets):
        """Laplacian kernel to compute boundary edge target ground truth"""
        laplacian_kernel = torch.tensor([[-1., -1., -1.],
                                         [-1.,  8., -1.],
                                         [-1., -1., -1.]], device=targets.device).view(1, 1, 3, 3)
        edges = F.conv2d(targets, laplacian_kernel, padding=1)
        return torch.abs(edges) > 0.1

    def forward(self, logits, edge_logits, targets):
        l_dice = self.dice_loss(logits, targets)
        l_bce = self.bce_loss(logits, targets)
        
        edge_targets = self._extract_edges(targets).float()
        l_edge = self.bce_loss(edge_logits, edge_targets)
        
        total_loss = l_dice + l_bce + self.lambda_edge * l_edge
        return total_loss, l_dice, l_bce, l_edge


def calculate_dice_score(preds, targets, threshold=0.5):
    """Computes Dice similarity coefficient"""
    preds_binary = (preds > threshold).float()
    intersection = (preds_binary * targets).sum()
    total = preds_binary.sum() + targets.sum()
    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()


def calculate_iou_score(preds, targets, threshold=0.5):
    """Computes Intersection over Union (IoU) / Jaccard Index"""
    preds_binary = (preds > threshold).float()
    intersection = (preds_binary * targets).sum()
    union = preds_binary.sum() + targets.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_ece(probs, targets, n_bins=10):
    """
    Expected Calibration Error (ECE) for Confidence Calibration evaluation.
    """
    probs_flat = probs.detach().cpu().view(-1).numpy()
    targets_flat = targets.detach().cpu().view(-1).numpy()
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        in_bin = (probs_flat > bin_lower) & (probs_flat <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets_flat[in_bin].mean()
            avg_confidence_in_bin = probs_flat[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)
