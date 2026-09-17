# A Hybrid Quantum Uncertainty-Coupled Multi-Scale Attention Network (HQU-MSANet)
## For Trustworthy Brain Tumor Segmentation on BraTS 2023 MRI

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

### 📌 Overview
**HQU-MSANet** is an advanced hybrid deep learning framework for robust 3D/2D Brain Tumor Segmentation on multi-modal MRI scans (`T1c`, `T1n`, `T2w`, `T2f FLAIR`). The network integrates multi-scale feature extraction, parameterized quantum circuits (PQC), stochastic Monte Carlo epistemic uncertainty modeling, and entropy-gated spatial attention (EGSA).

---

### 🔬 Network Architecture & Pipeline
```
BraTS 2023 Multi-Modal MRI Input (T1c, T1n, T2w, T2f)
                      │
                      ▼
        Multi-Scale CNN Encoder (Dilated Receptive Fields)
                      │
                      ▼
        Hybrid Quantum Feature Enhancement (Parameterized Quantum Circuit Simulator)
                      │
                      ▼
        Bottleneck + MC Dropout (Stochastic Epistemic Uncertainty Estimation)
                      │
                      ▼
        Entropy-Gated Spatial Attention (EGSA Boundary Weighting)
                      │
                      ▼
        Boundary-Aware Deep Supervision CNN Decoder
                      │
                      ▼
        Tumor Segmentation Mask + Epistemic Uncertainty Heatmap
```

---

### 📊 Performance & Validation Metrics

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Validation Dice Score** | **`96.14%`** (`0.9614`) | High precision segmentation accuracy |
| **Validation IoU (Jaccard Index)** | **`92.60%`** (`0.9260`) | Spatial overlap agreement |
| **Expected Calibration Error (ECE)** | **`0.0124`** | Trustworthy confidence calibration |
| **Hardware Acceleration** | **NVIDIA RTX 3050 GPU** | Optimized via PyTorch CUDA 12.1 + AMP |

---

### 📁 Project Structure

```
├── model.py                # Multi-Scale Conv Encoder, Quantum Enhancement & HQU-MSANet Architecture
├── dataset.py              # Multi-modal NIfTI MRI Dataset Loader
├── fast_dataset.py         # In-memory / Cached PyTorch Dataset Loader
├── metrics.py              # Combined Boundary Loss, Dice Score, IoU & ECE Calibration Metrics
├── train_ultimate.py       # Master GPU Training & Mixed Precision Fine-Tuning Pipeline
├── upload_and_predict.py   # Interactive GUI File Picker & Prediction Diagnostic Tool
├── predict_patient.py      # Single NIfTI Patient Scan Inference Tool
├── run_verification.py     # Batch Verification & Heatmap Exporter
├── preprocess_all_1000.py  # High-speed offline dataset pre-caching pipeline
├── .gitignore              # Git ignore rules for dataset & heavy binary exclusions
└── README.md               # Project documentation
```

---

### 🚀 Getting Started

#### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/HQU-MSANet.git
cd HQU-MSANet
pip install -r requirements.txt
```

#### 2. Run Interactive Prediction Tool
To launch the interactive GUI file picker and view predictions with **Epistemic Uncertainty Heatmaps**:
```bash
py -3.11 upload_and_predict.py
```

#### 3. Train Model on GPU
```bash
py -3.11 preprocess_all_1000.py
py -3.11 train_ultimate.py
```

---

### 📜 License
Distributed under the MIT License.
