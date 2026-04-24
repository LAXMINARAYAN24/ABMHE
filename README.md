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
8. [Running the Demo](#running-the-demo)
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

* Python ≥ 3.10
* [PyTorch](https://pytorch.org/) ≥ 2.0 (CPU or CUDA)
* OpenCV (`opencv-python`)
* Pillow
* NumPy
* Streamlit ≥ 1.30

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/LAXMINARAYAN24/ABMHE.git
cd ABMHE

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy streamlit
```

For GPU acceleration install the CUDA-enabled PyTorch wheel matching your CUDA version from https://pytorch.org/get-started/locally/.

---

## Running the Demo

```bash
streamlit run frontend.py
```

Then open the URL printed in the terminal (default: http://localhost:8501).

### Workflow

1. **Upload** a PNG, JPEG, BMP, or TIFF image using the file uploader.
2. In the **sidebar**, choose a single enhancement method from the dropdown.
3. Click **Run single** to see a side-by-side comparison with NRSS and Local Contrast scores.
4. Click **Run all** to run every method at once and display a comparison grid plus a metrics table.

For colour images, two additional RGB modes appear: *ABMHE RGB (per-channel)* and *ABMHE RGB (YCbCr-Y)*.

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
