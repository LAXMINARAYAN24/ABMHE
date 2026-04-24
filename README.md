# ABMHE — Adjacent-Blocks-Based Modification for Local Histogram Equalization

A GPU-accelerated image-enhancement library and interactive Streamlit demo that implements the **ABMHE** algorithm alongside four classical histogram-equalization baselines.

---

## Table of Contents

1. [Overview](#overview)
2. [Algorithm](#algorithm)
3. [Methods](#methods)
4. [Quality Metrics](#quality-metrics)
5. [Project Structure](#project-structure)
6. [Requirements](#requirements)
7. [Installation](#installation)
   - [0 — Prerequisites](#0--prerequisites)
   - [1 — Clone the repository](#1--clone-the-repository)
   - [2 — Create a virtual environment](#2--create-and-activate-a-virtual-environment-recommended)
   - [3 — Install dependencies](#3--install-dependencies)
   - [4 — Verify the installation](#4--verify-the-installation)
8. [Running the Demo](#running-the-demo)
   - [Start the app](#start-the-streamlit-app)
   - [Step-by-step walkthrough](#step-by-step-walkthrough)
   - [Troubleshooting](#troubleshooting)
9. [Usage (API)](#usage-api)
10. [How It Works — Step by Step](#how-it-works--step-by-step)
11. [RGB Support](#rgb-support)

---

## Overview

ABMHE enhances image contrast by dividing the image into **overlapping blocks**, classifying each block as *active*, *normal*, or *inactive* based on its local edge content, assigning a guided histogram to each block type, and blending the equalized blocks back together with a **Hanning window** to avoid block-boundary artefacts.

Key properties:

| Property | Value |
|---|---|
| Block size | 64 × 64 px (fixed) |
| Block step | block\_size / 4 = 16 px |
| Colour modes | Grayscale · RGB per-channel · RGB via YCbCr |
| Compute backend | PyTorch (CPU or CUDA) |
| Frontend | Streamlit |

---

## Algorithm

### Block classification

For every overlapping block the **Sobel gradient magnitude** is computed. A global gradient threshold *T* is derived as the median of the gradient-magnitude CDF (Eq. 1 of the paper). The fraction of pixels whose gradient exceeds *T* determines the block type:

| Fraction of high-gradient pixels | Block type |
|---|---|
| ≥ upper\_thresh (default 0.50) | **Active** — rich in edges/detail |
| < lower\_thresh (default 0.20) | **Inactive** — smooth / flat region |
| otherwise | **Normal** |

### Histogram assignment

* **Active blocks** — use the histogram of their own high-gradient pixels directly.
* **Normal blocks** — average of the block's own high-gradient histogram and a *guided* histogram interpolated from the four nearest active neighbours (weighted by 1 / distance).
* **Inactive blocks** — guided histogram from the nearest active/normal neighbours, corrected by the block's low-gradient histogram to avoid over-equalisation of flat regions.

### Blending

Each block is equalised with its assigned histogram via CDF mapping, then multiplied by a 2-D Hanning window and accumulated into the output canvas using **weighted-average blending**. This produces smooth, artefact-free transitions between blocks.

---

## Methods

Five histogram-equalisation methods are available for comparison:

| Method | Description |
|---|---|
| **GHE** | Global Histogram Equalization (OpenCV `equalizeHist`) |
| **BBHE** | Brightness-Preserving Bi-Histogram Equalization — splits at the mean and equalises each half independently |
| **CLAHE** | Contrast-Limited Adaptive Histogram Equalization (OpenCV, clip limit 3.0, 32 × 32 tiles) |
| **POSHE** | Partially Overlapped Sub-block Histogram Equalization — sliding window with Hanning blend, no block-type guidance |
| **ABMHE** | This work — edge-guided adaptive histogram equalization |

---

## Quality Metrics

Two no-reference metrics are computed on the enhanced images:

| Metric | Description |
|---|---|
| **NRSS** | No-Reference Sharpness Score — `1 − mean_SSIM(sharp_blocks, blurred_blocks)`. Higher is sharper. |
| **Local Contrast** | Mean of `(A_max − A_min) / (A_max + A_min)` over every 3 × 3 neighbourhood (Eq. 10). Higher means more contrast. |

---

## Project Structure

```
ABMHE/
├── app.py          # Core library: algorithm, baselines, metrics
└── frontend.py     # Streamlit web application
```

### `app.py` — public API

| Symbol | Type | Description |
|---|---|---|
| `abmhe` | function | Main ABMHE algorithm (grayscale) |
| `abmhe_rgb_perchannel` | function | Apply ABMHE to each R/G/B channel independently |
| `abmhe_rgb_ycbcr` | function | Apply ABMHE to Y channel in YCbCr space |
| `ghe` | function | Global HE baseline |
| `bbhe` | function | BBHE baseline |
| `clahe` | function | CLAHE baseline |
| `poshe` | function | POSHE baseline |
| `compute_metrics` | function | Returns (NRSS, local\_contrast) for an image tensor |
| `pil_to_tensor` | function | PIL → float32 PyTorch tensor |
| `is_effectively_grayscale` | function | Detects near-grayscale colour images |

---

## Requirements

| Dependency | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | `python --version` to check |
| PyTorch | 2.0 | CPU or CUDA wheel |
| torchvision | 0.15 | Installed alongside PyTorch |
| opencv-python | 4.7 | Used for GHE and CLAHE |
| Pillow | 9.0 | Image I/O |
| NumPy | 1.23 | Array utilities |
| Streamlit | 1.30 | Web frontend |

---

## Installation

### 0 — Prerequisites

Make sure Python 3.10+ is installed:

```bash
python --version   # must print Python 3.10.x or higher
pip --version
```

If Python is missing, download it from https://www.python.org/downloads/ or use your OS package manager (e.g. `sudo apt install python3.11`).

### 1 — Clone the repository

```bash
git clone https://github.com/LAXMINARAYAN24/ABMHE.git
cd ABMHE
```

### 2 — Create and activate a virtual environment (recommended)

```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate

# Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)`.

### 3 — Install dependencies

**CPU-only (works on any machine):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**CUDA (GPU acceleration):**

Find the correct install command for your CUDA version at https://pytorch.org/get-started/locally/, for example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4 — Verify the installation

```bash
python - <<'EOF'
import torch, cv2, streamlit, PIL
print("PyTorch :", torch.__version__)
print("OpenCV  :", cv2.__version__)
print("Streamlit:", streamlit.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

Expected output (versions may differ):

```
PyTorch : 2.x.x
OpenCV  : 4.x.x
Streamlit: 1.x.x
CUDA available: False   # True if a CUDA GPU is present
```

---

## Running the Demo

### Start the Streamlit app

```bash
streamlit run frontend.py
```

Streamlit will print something like:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open **http://localhost:8501** in your browser. If the page does not open automatically, copy the URL from the terminal.

> **Note:** Keep the terminal open while using the app. Press `Ctrl+C` to stop the server.

### Step-by-step walkthrough

| Step | What to do |
|---|---|
| 1 | Click **Browse files** (or drag and drop) to upload a PNG, JPEG, BMP, or TIFF image. |
| 2 | The sidebar on the left shows a **Settings** panel once an image is loaded. |
| 3 | Use the **Single method** dropdown to pick one algorithm: ABMHE, GHE, BBHE, CLAHE, or POSHE. For colour images, two additional RGB modes also appear. |
| 4 | Click **Run single** (blue button) to enhance the image with the chosen method. A side-by-side view of the original and enhanced image is shown, along with NRSS and Local Contrast scores. |
| 5 | Click **Run all** (grey button) to run every available method at once. The result is a comparison grid and a metrics table so you can evaluate all methods side by side. |

### Custom port or host

```bash
streamlit run frontend.py --server.port 8080
streamlit run frontend.py --server.address 0.0.0.0   # expose on the local network
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | Dependencies not installed | Re-run the install step and ensure the virtualenv is activated |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV missing | `pip install opencv-python` |
| App loads but processing is very slow | Running on CPU with a large image | Resize image to ≤ 1 MP before uploading, or use a CUDA GPU |
| `RuntimeError: CUDA out of memory` | Image too large for GPU VRAM | Reduce image resolution or process on CPU |
| Browser shows blank page | Streamlit not fully started yet | Wait a few seconds and refresh |
| Port 8501 already in use | Another Streamlit instance running | Use `--server.port 8080` or kill the other process |

---

## Usage (API)

```python
from PIL import Image
from app import pil_to_tensor, is_effectively_grayscale, abmhe, compute_metrics

pil_img = Image.open("photo.jpg")

# Convert to tensor
gray_t = pil_to_tensor(pil_img.convert("L"))   # float32 (H,W)

# Enhance
enhanced = abmhe(gray_t, block_size=64)         # uint8 (H,W) tensor

# Quality metrics
nrss, contrast = compute_metrics(enhanced)
print(f"NRSS={nrss:.4f}  Contrast={contrast:.4f}")

# Save result
import numpy as np
from PIL import Image as PILImage
PILImage.fromarray(enhanced.cpu().numpy()).save("enhanced.png")
```

### RGB enhancement

```python
from app import pil_to_tensor, abmhe_rgb_ycbcr

rgb_t = pil_to_tensor(pil_img.convert("RGB"))  # float32 (3,H,W)
enhanced_rgb = abmhe_rgb_ycbcr(rgb_t)          # uint8  (3,H,W)
```

---

## How It Works — Step by Step

```
Input image (H × W)
       │
       ▼
1. Compute Sobel gradient magnitude G(i,j)
       │
       ▼
2. Derive threshold T  =  median of CDF(G)
       │
       ▼
3. Slide 64×64 window with step=16 over the image
   For each block:
     ├─ Classify as active / normal / inactive
     ├─ Build hist_high_gradient  (pixels where G > T)
     └─ Build hist_low_gradient   (pixels where G ≤ T)
       │
       ▼
4. Assign extended histogram per block type
     Active   → hist_high_gradient
     Normal   → 0.5 × (hist_high + guided_from_active_neighbours)
     Inactive → correction_factor × guided_from_active+normal_neighbours
       │
       ▼
5. Equalise each block via CDF of its extended histogram
       │
       ▼
6. Blend with 2-D Hanning window (weighted average)
       │
       ▼
Output enhanced image (H × W, uint8)
```

---

## RGB Support

| Mode | Description |
|---|---|
| **Per-channel** | ABMHE applied independently to R, G, and B. Fast but may introduce hue shifts. |
| **YCbCr-Y** | Image converted to YCbCr (BT.601); ABMHE applied to the Y (luminance) channel only; Cb/Cr left unchanged; converted back to RGB. Preserves colours accurately. |
