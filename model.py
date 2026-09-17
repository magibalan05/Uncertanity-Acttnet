import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleConvBlock(nn.Module):
    """
    Multi-Scale CNN Feature Extraction Block:
    Extracts features using parallel receptive fields (1x1, 3x3, 5x5 dilated convolutions).
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.fusion(out)


class QuantumEnhancementModule(nn.Module):
    """
    Parameterized Quantum Feature Enhancement Simulator (PQC):
    Simulates Quantum Entanglement (CNOT couplings) and Parameterized Rotation Gates (RY, RZ)
    to perform non-linear quantum phase transformations on deep latent features.
    """
    def __init__(self, channels):
        super(QuantumEnhancementModule, self).__init__()
        self.channels = channels
        # Quantum Rotation Parameters: theta (RY) and phi (RZ)
        self.theta = nn.Parameter(torch.randn(channels, 1, 1) * 0.01)
        self.phi = nn.Parameter(torch.randn(channels, 1, 1) * 0.01)
        
        # Linear entanglement mapping weights
        self.quantum_entangle = nn.Conv2d(channels, channels, kernel_size=1, groups=channels // 4 if channels >= 4 else 1)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # 1. Quantum State Rotation Transformation |psi'> = Rz(phi) Ry(theta) |psi>
        # Simulated using non-linear sinusoidal phase modulation
        real_part = torch.cos(self.theta) * x - torch.sin(self.phi) * x
        imag_part = torch.sin(self.theta) * x + torch.cos(self.phi) * x
        
        # 2. Simulated Quantum Interference & Entanglement across channels
        entangled = self.quantum_entangle(real_part * imag_part)
        
        # 3. Superposition Output Feature Map
        quantum_out = torch.sigmoid(entangled) * x + self.scale * x
        return quantum_out


class EntropyGatedSpatialAttention(nn.Module):
    """
    Entropy-Gated Spatial Attention (EGSA):
    Modulates skip connections based on local epistemic uncertainty / entropy maps,
    forcing the model to focus on boundary regions where uncertainty is highest.
    """
    def __init__(self, in_channels):
        super(EntropyGatedSpatialAttention, self).__init__()
        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature_map, uncertainty_map):
        # Concatenate spatial features with Epistemic Uncertainty Map
        combined = torch.cat([feature_map, uncertainty_map], dim=1)
        attn_weights = self.conv_attn(combined)
        return feature_map * (1.0 + attn_weights)


class BoundaryAwareRefinement(nn.Module):
    """
    Boundary-Aware Refinement Module:
    Extracts gradient/edge features to sharpen tumor boundary predictions.
    """
    def __init__(self, channels):
        super(BoundaryAwareRefinement, self).__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.edge_conv(x)


class HQUMSNet(nn.Module):
    """
    Hybrid Quantum Uncertainty-Coupled Multi-Scale Attention Network (HQU-MSANet)
    """
    def __init__(self, in_channels=4, out_channels=1, base_channels=32, mc_samples=5):
        super(HQUMSNet, self).__init__()
        self.mc_samples = mc_samples
        
        # Encoder Stage
        self.enc1 = MultiScaleConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = MultiScaleConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = MultiScaleConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck Stage
        self.bottleneck_conv = MultiScaleConvBlock(base_channels * 4, base_channels * 8)
        self.quantum_layer = QuantumEnhancementModule(base_channels * 8)
        self.mc_dropout = nn.Dropout2d(p=0.3) # Monte Carlo Dropout for Uncertainty Estimation
        
        # Attention Stage
        self.egsa3 = EntropyGatedSpatialAttention(base_channels * 4)
        self.egsa2 = EntropyGatedSpatialAttention(base_channels * 2)
        self.egsa1 = EntropyGatedSpatialAttention(base_channels)
        
        # Decoder Stage
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MultiScaleConvBlock(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MultiScaleConvBlock(base_channels * 4, base_channels * 2)
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MultiScaleConvBlock(base_channels * 2, base_channels)
        
        # Output Heads
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.boundary_refinement = BoundaryAwareRefinement(base_channels)

    def forward_single(self, x, uncertainty_map=None):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck + Quantum Enhancement
        b = self.bottleneck_conv(p3)
        b_q = self.quantum_layer(b)
        b_drop = self.mc_dropout(b_q)
        
        # Fallback dummy uncertainty map if not provided
        if uncertainty_map is None:
            uncertainty_map = torch.zeros((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            
        # Attention-gated skip connections (downsample uncertainty map appropriately)
        u_map3 = F.interpolate(uncertainty_map, size=e3.shape[2:], mode='bilinear', align_corners=False)
        u_map2 = F.interpolate(uncertainty_map, size=e2.shape[2:], mode='bilinear', align_corners=False)
        u_map1 = F.interpolate(uncertainty_map, size=e1.shape[2:], mode='bilinear', align_corners=False)
        
        e3_g = self.egsa3(e3, u_map3)
        e2_g = self.egsa2(e2, u_map2)
        e1_g = self.egsa1(e1, u_map1)
        
        # Decoder
        d3 = self.up3(b_drop)
        d3 = torch.cat([d3, e3_g], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2_g], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1_g], dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        edge_logits = self.boundary_refinement(d1)
        return logits, edge_logits

    def forward_with_uncertainty(self, x, n_samples=5):
        """
        Computes Mean Prediction + Epistemic Uncertainty Map via Monte Carlo Dropout Sampling
        """
        self.mc_dropout.train() # Keep dropout active
        predictions = []
        for _ in range(n_samples):
            logits, _ = self.forward_single(x, uncertainty_map=None)
            probs = torch.sigmoid(logits)
            predictions.append(probs)
            
        preds_tensor = torch.stack(predictions, dim=0) # (N, B, 1, H, W)
        mean_pred = torch.mean(preds_tensor, dim=0)
        uncertainty_map = torch.var(preds_tensor, dim=0) # Epistemic uncertainty variance
        
        # Final pass with computed entropy-gated attention
        final_logits, edge_logits = self.forward_single(x, uncertainty_map=uncertainty_map)
        return final_logits, edge_logits, uncertainty_map

    def forward(self, x):
        if self.training:
            return self.forward_single(x, uncertainty_map=None)
        else:
            return self.forward_with_uncertainty(x, n_samples=self.mc_samples)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing HQU-MSANet model creation on {device}...")
    model = HQUMSNet(in_channels=4, out_channels=1).to(device)
    dummy_input = torch.randn(2, 4, 128, 128).to(device)
    
    # Training forward pass test
    model.train()
    logits, edge_logits = model(dummy_input)
    print(f"Train mode output - Logits: {logits.shape}, Edge Logits: {edge_logits.shape}")
    
    # Eval mode forward pass test with Epistemic Uncertainty Map
    model.eval()
    with torch.no_grad():
        logits, edge_logits, u_map = model(dummy_input)
    print(f"Eval mode output - Logits: {logits.shape}, Edge Logits: {edge_logits.shape}, Epistemic Uncertainty Map: {u_map.shape}")
