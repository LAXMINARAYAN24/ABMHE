"""
ABMHE: Adjacent-Blocks-Based Modification for Local Histogram Equalization
Core image-enhancement functions used by the Streamlit frontend.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#  Low-level PyTorch helpers

def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL image  →  float32 tensor [0,255], shape (H,W) or (C,H,W)."""
    arr = np.array(pil_img, dtype=np.float32)
    t = torch.from_numpy(arr).to(DEVICE)
    if t.ndim == 2:
        return t                        # grayscale (H,W)
    return t.permute(2, 0, 1)          # RGB → (C,H,W)


def is_effectively_grayscale(pil_img: Image.Image, tolerance: int = 1) -> bool:
    """
    Return True if an image should be treated as grayscale.

    """
    image_np = np.array(pil_img)

    if image_np.ndim == 2:
        return True

    if image_np.ndim == 3 and image_np.shape[2] >= 3:
        red = image_np[..., 0].astype(np.int16)
        green = image_np[..., 1].astype(np.int16)
        blue = image_np[..., 2].astype(np.int16)
        max_channel_delta = max(
            np.abs(red - green).max(),
            np.abs(green - blue).max(),
        )
        return max_channel_delta <= tolerance

    return pil_img.mode in ("L", "1", "P")


def tensor_to_uint8(t: torch.Tensor) -> torch.Tensor:
    return t.clamp(0, 255).round().to(torch.uint8)


def sobel_gradients(img: torch.Tensor) -> torch.Tensor:
    """
    Sobel gradient magnitude. img: float32 (H,W).
    Returns G: float32 (H,W) in [0,255].
    """
    image_batch = img.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    kx = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], device=DEVICE).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]], device=DEVICE).view(1, 1, 3, 3)

    padded_image = F.pad(image_batch, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded_image, kx).squeeze()  # conv expects (b,c,h,w)
    grad_y = F.conv2d(padded_image, ky).squeeze()
    gradient_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalise to [0,255]
    grad_min, grad_max = gradient_mag.min(), gradient_mag.max()
    if grad_max > grad_min:
        gradient_mag = (gradient_mag - grad_min) / (grad_max - grad_min) * 255.0
    return gradient_mag


def gradient_threshold(img: torch.Tensor, L: int = 256) -> tuple[float, torch.Tensor]:
    """
    Eq.(1): T = median of the gradient-magnitude CDF.
    Returns (T_float, G_float32 in [0,255]).
    """
    gradient_image = sobel_gradients(img)
    flat_gradient = gradient_image.flatten()
    gradient_hist = torch.zeros(L, device=DEVICE)
    bin_indices = flat_gradient.long().clamp(0, L - 1)
    gradient_hist.scatter_add_(0, bin_indices, torch.ones_like(flat_gradient))
    gradient_cdf = torch.cumsum(gradient_hist / (flat_gradient.numel() + 1e-8), dim=0)
    threshold = int((gradient_cdf >= 0.5).nonzero(as_tuple=False)[0].item())
    return float(threshold), gradient_image


def torch_histogram(values: torch.Tensor, L: int = 256,
                    normalise: bool = True) -> torch.Tensor:
    """1-D histogram of a 1-D float32 tensor, binned into [0,L)."""
    bin_indices = values.long().clamp(0, L - 1)
    hist = torch.zeros(L, device=DEVICE)
    hist.scatter_add_(0, bin_indices, torch.ones(bin_indices.numel(), device=DEVICE))
    if normalise:
        total = hist.sum()
        hist = hist / (total + 1e-8)
    return hist


def equalize_with_hist(block: torch.Tensor,
                       hist_norm: torch.Tensor,
                       L: int = 256) -> torch.Tensor:
    """
    Map pixel values through the CDF of hist_norm.
    block   : float32 (m,n)
    hist_norm: float32 (L,)  normalised histogram
    Returns float32 (m,n) in [0, L-1].
    """
    cdf = torch.cumsum(hist_norm, dim=0) * (L - 1)
    cdf = cdf.clamp(0, L - 1)
    pixel_indices = block.long().clamp(0, L - 1).flatten()
    equalized = cdf[pixel_indices].reshape(block.shape)  # replace block values with CDF values
    return equalized

# to make smooth transitions between blocks in ABMHE
def hanning_2d(height: int, width: int, device) -> torch.Tensor:
    """2-D Hanning window with shape (height, width)."""
    window_y = torch.hann_window(height, periodic=False, device=device)
    window_x = torch.hann_window(width, periodic=False, device=device)
    return torch.outer(window_y, window_x)


# ABMHE algorithm

def abmhe(img_float: torch.Tensor,
          block_size: int = 64,
          L: int = 256,
          lower_thresh: float = 0.20,
          upper_thresh: float = 0.50) -> torch.Tensor:
    """
    Returns

    uint8 tensor (H,W) enhanced image
    """
    image_height, image_width = img_float.shape
    block_height = block_width = block_size
    step_y = step_x = block_height // 4  # paper: hstep = m/4, vstep = n/4

    gradient_threshold_value, gradient_image = gradient_threshold(img_float, L)

    top_positions = list(range(0, image_height - block_height + 1, step_y))
    left_positions = list(range(0, image_width - block_width + 1, step_x))
    num_rows, num_cols = len(top_positions), len(left_positions)

    block_grid = [[None] * num_cols for _ in range(num_rows)]

    for row_idx, top in enumerate(top_positions):
        for col_idx, left in enumerate(left_positions):
            block = img_float[top:top + block_height, left:left + block_width]
            block_gradient = gradient_image[top:top + block_height, left:left + block_width]

            # block -> actual pixel
            # block_gradient -> edge information

            high_gradient_mask = block_gradient > gradient_threshold_value
            high_gradient_ratio = high_gradient_mask.float().mean().item()

            if high_gradient_ratio >= upper_thresh:
                block_type = "active"
            elif high_gradient_ratio < lower_thresh:
                block_type = "inactive"
            else:
                block_type = "normal"

            high_gradient_pixels = block[high_gradient_mask]
            if high_gradient_pixels.numel() == 0:
                high_gradient_pixels = block.flatten()
            hist_high_gradient = torch_histogram(high_gradient_pixels, L)

            low_gradient_pixels = block[~high_gradient_mask]
            if low_gradient_pixels.numel() == 0:
                low_gradient_pixels = block.flatten()
            hist_low_gradient, _ = torch.histogram(
                low_gradient_pixels.cpu(),
                torch.arange(L + 1, dtype=torch.float32)
            )
            hist_low_gradient = hist_low_gradient.to(DEVICE)

            block_grid[row_idx][col_idx] = {
                "block_type": block_type,
                "hist_high_gradient": hist_high_gradient,
                "hist_low_gradient": hist_low_gradient,
                "top": top,
                "left": left,
                "extended_hist": None,
            }

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            block_info = block_grid[row_idx][col_idx]
            if block_info["block_type"] == "active":
                block_info["extended_hist"] = block_info["hist_high_gradient"].clone()

    # finds nearest neighbour block having valid histogram
    # returns histogram and distance
    def nearest_hist_in_direction(row_idx, col_idx, delta_row, delta_col, allowed_types):
        current_row, current_col = row_idx + delta_row, col_idx + delta_col
        distance = 1
        while 0 <= current_row < num_rows and 0 <= current_col < num_cols:
            neighbor = block_grid[current_row][current_col]
            if neighbor["block_type"] in allowed_types and neighbor["extended_hist"] is not None:
                return neighbor["extended_hist"], distance
            current_row += delta_row
            current_col += delta_col
            distance += 1
        return None, None

    def guided_histogram(row_idx, col_idx, allowed_types):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        weighted_sum = torch.zeros(L, device=DEVICE)
        total_weight = 0.0

        for delta_row, delta_col in directions:
            neighbor_hist, distance = nearest_hist_in_direction(
                row_idx, col_idx, delta_row, delta_col, allowed_types
            )
            if neighbor_hist is not None and distance > 0:
                weight = 1.0 / distance
                weighted_sum += weight * neighbor_hist
                total_weight += weight

        if total_weight < 1e-8:
            return None
        return weighted_sum / total_weight

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            block_info = block_grid[row_idx][col_idx]
            if block_info["block_type"] != "normal":
                continue

            guide_hist = guided_histogram(row_idx, col_idx, {"active"})
            if guide_hist is None:
                guide_hist = block_info["hist_high_gradient"].clone()

            normal_hist = 0.5 * (block_info["hist_high_gradient"] + guide_hist)
            block_info["extended_hist"] = normal_hist / (normal_hist.sum() + 1e-8)

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            block_info = block_grid[row_idx][col_idx]
            if block_info["block_type"] != "inactive":
                continue

            guide_hist = guided_histogram(row_idx, col_idx, {"active", "normal"})
            if guide_hist is None:
                guide_hist = torch.ones(L, device=DEVICE) / L

            low_grad_hist = block_info["hist_low_gradient"]
            max_low_grad = low_grad_hist.max()
            if max_low_grad > 0:
                correction_factor = 1.0 - low_grad_hist / (max_low_grad + 1e-8)
            else:
                correction_factor = torch.ones(L, device=DEVICE)

            inactive_hist = correction_factor * guide_hist
            block_info["extended_hist"] = inactive_hist / (inactive_hist.sum() + 1e-8)

    blending_window = hanning_2d(block_height, block_width, DEVICE)   # center high weight and edges low weight and used for smooth blending
    weighted_output_sum = torch.zeros(image_height, image_width, device=DEVICE)
    weighted_output_count = torch.zeros(image_height, image_width, device=DEVICE)



    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            block_info = block_grid[row_idx][col_idx]
            top, left = block_info["top"], block_info["left"]
            block = img_float[top:top + block_height, left:left + block_width]

            equalized_block = equalize_with_hist(block, block_info["extended_hist"], L)
            weighted_output_sum[top:top + block_height, left:left + block_width] += equalized_block * blending_window
            weighted_output_count[top:top + block_height, left:left + block_width] += blending_window

    covered_pixels = weighted_output_count > 1e-8
    enhanced = torch.zeros(image_height, image_width, device=DEVICE)
    enhanced[covered_pixels] = (
        weighted_output_sum[covered_pixels] / weighted_output_count[covered_pixels]
    )
    enhanced[~covered_pixels] = img_float[~covered_pixels]

    return tensor_to_uint8(enhanced)


#  RGB extension


def abmhe_rgb_perchannel(rgb_tensor: torch.Tensor,
                          block_size: int = 64) -> torch.Tensor:
    """
    Apply ABMHE independently on each R, G, B channel.
    rgb_tensor: float32 (3,H,W) in [0,255]
    Returns   : uint8  (3,H,W)
    """
    enhanced_channels = [abmhe(rgb_tensor[channel], block_size=block_size) for channel in range(3)]
    return torch.stack(enhanced_channels, dim=0)


def rgb_to_ycbcr(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB tensor (3,H,W) in [0,255] to YCbCr using BT.601.
    """
    red, green, blue = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
    y_channel = 0.299 * red + 0.587 * green + 0.114 * blue
    cb_channel = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    cr_channel = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    return torch.stack([y_channel, cb_channel, cr_channel], dim=0)


def ycbcr_to_rgb(ycbcr_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert YCbCr tensor (3,H,W) to RGB tensor in [0,255].
    """
    y_channel, cb_channel, cr_channel = ycbcr_tensor[0], ycbcr_tensor[1], ycbcr_tensor[2]
    red = y_channel + 1.402 * (cr_channel - 128.0)
    green = y_channel - 0.344136 * (cb_channel - 128.0) - 0.714136 * (cr_channel - 128.0)
    blue = y_channel + 1.772 * (cb_channel - 128.0)
    return torch.stack([red, green, blue], dim=0).clamp(0, 255)


def abmhe_rgb_ycbcr(rgb_tensor: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Apply ABMHE on luminance (Y) channel only, then convert back to RGB.
    """
    ycbcr_tensor = rgb_to_ycbcr(rgb_tensor)
    y_enhanced = abmhe(ycbcr_tensor[0], block_size=block_size).float()
    ycbcr_enhanced = torch.stack([y_enhanced, ycbcr_tensor[1], ycbcr_tensor[2]], dim=0)
    rgb_enhanced = ycbcr_to_rgb(ycbcr_enhanced)
    return tensor_to_uint8(rgb_enhanced)


#  Baseline method

def ghe(img: torch.Tensor, L: int = 256) -> torch.Tensor:
    """Global Histogram Equalization using OpenCV."""
    img_u8 = tensor_to_uint8(img).cpu().numpy()
    enhanced_u8 = cv2.equalizeHist(img_u8)
    return torch.from_numpy(enhanced_u8).to(DEVICE)


def bbhe(img: torch.Tensor, L: int = 256) -> torch.Tensor:
    """Brightness-Preserving Bi-Histogram Equalization."""
    mean_val = img.mean().item()
    lo_mask  = img <= mean_val
    hi_mask  = ~lo_mask

    def equalize_part(pixels, lo, hi):
        h = torch_histogram(pixels, L)
        cdf = torch.cumsum(h, dim=0)
        # Map CDF to [lo, hi]
        cdf_scaled = lo + (hi - lo) * cdf
        return cdf_scaled.clamp(lo, hi)

    cdf_lo = equalize_part(img[lo_mask], 0.0, mean_val)
    cdf_hi = equalize_part(img[hi_mask], mean_val + 1, L - 1.0)

    out = torch.zeros_like(img)
    out[lo_mask] = cdf_lo[img[lo_mask].long().clamp(0, L - 1)]
    out[hi_mask] = cdf_hi[img[hi_mask].long().clamp(0, L - 1)]
    return tensor_to_uint8(out)


def clahe(img: torch.Tensor, clip_limit: float = 3.0,
          tile: int = 32, L: int = 256) -> torch.Tensor:
    """Contrast-Limited Adaptive Histogram Equalization using OpenCV."""
    img_u8 = tensor_to_uint8(img).cpu().numpy()
    clahe_op = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile), int(tile)),
    )
    enhanced_u8 = clahe_op.apply(img_u8)
    return torch.from_numpy(enhanced_u8).to(DEVICE)


def poshe(img: torch.Tensor, block_size: int = 64,
          step: int = 16, L: int = 256) -> torch.Tensor:
    """Partially Overlapped Sub-block Histogram Equalization."""
    H, W  = img.shape
    m     = block_size
    win   = hanning_2d(m, m, DEVICE)
    out   = torch.zeros(H, W, device=DEVICE)
    count = torch.zeros(H, W, device=DEVICE)

    for y in range(0, H - m + 1, step):
        for x in range(0, W - m + 1, step):
            blk    = img[y:y+m, x:x+m]
            hist_n = torch_histogram(blk.flatten(), L)
            eq_blk = equalize_with_hist(blk, hist_n, L)
            out  [y:y+m, x:x+m] += eq_blk * win
            count[y:y+m, x:x+m] += win

    covered = count > 1e-8
    result  = torch.zeros_like(img)
    result[covered]  = out[covered] / count[covered]
    result[~covered] = img[~covered]
    return tensor_to_uint8(result)


#  Quality metrics

def gaussian_kernel(size: int = 7, sigma: float = 1.5,
                    device=DEVICE) -> torch.Tensor:
    """1-D Gaussian → outer product for 2-D kernel."""
    x     = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    gauss /= gauss.sum()  # normalize
    return torch.outer(gauss, gauss)


def ssim_pair(img_a: torch.Tensor, img_b: torch.Tensor,
              k1: float = 0.01, k2: float = 0.03,
              L: int = 255) -> float:
    """SSIM between two (H,W) float tensors."""
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    kern   = gaussian_kernel(11, 1.5, img_a.device).view(1, 1, 11, 11)
    pad    = 5

    def mu(x):
        return F.conv2d(F.pad(x.unsqueeze(0).unsqueeze(0),
                              (pad,)*4, mode="reflect"), kern).squeeze() 

    mu_a, mu_b = mu(img_a), mu(img_b)
    mu_aa, mu_bb, mu_ab = mu_a**2, mu_b**2, mu_a * mu_b

    def sig(x, mu_x_sq):
        return F.conv2d(F.pad((x**2).unsqueeze(0).unsqueeze(0),
                              (pad,)*4, mode="reflect"), kern).squeeze() - mu_x_sq

    sig_aa = sig(img_a, mu_aa)
    sig_bb = sig(img_b, mu_bb)
    sig_ab = (F.conv2d(F.pad((img_a * img_b).unsqueeze(0).unsqueeze(0),
                             (pad,)*4, mode="reflect"), kern).squeeze()
              - mu_ab)

    num   = (2 * mu_ab + C1) * (2 * sig_ab + C2)
    denom = (mu_aa + mu_bb + C1) * (sig_aa + sig_bb + C2)
    return (num / (denom + 1e-8)).mean().item()


def compute_nrss(img: torch.Tensor, K: int = 16, L: int = 256) -> float:
    
    img_f  = img.float()
    kern   = gaussian_kernel(7, 1.5, img.device).view(1, 1, 7, 7)
    blurred = F.conv2d(F.pad(img_f.unsqueeze(0).unsqueeze(0),
                             (3,)*4, mode="reflect"), kern).squeeze()

    H, W  = img.shape
    bsize = 16
    blocks = []
    for y in range(0, H - bsize + 1, bsize):
        for x in range(0, W - bsize + 1, bsize):
            v = img_f[y:y+bsize, x:x+bsize].var().item()
            blocks.append((v, y, x))
    blocks.sort(key=lambda t: -t[0])

    ssim_sum = 0.0
    for _, y, x in blocks[:K]:
        b1 = img_f[y:y+bsize, x:x+bsize]
        b2 = blurred[y:y+bsize, x:x+bsize]
        ssim_sum += ssim_pair(b1, b2)
    return 1.0 - ssim_sum / min(K, len(blocks))


def compute_local_contrast(img: torch.Tensor) -> float:
    """
    Eq.(10): C(i,j) = (Amax - Amin) / (Amax + Amin)  over 3x3, averaged.
    """
    img_f = img.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    padded = F.pad(img_f, (1,) * 4, mode="reflect")
    local_max = F.max_pool2d(padded, 3, stride=1, padding=0).squeeze()
    local_min = -F.max_pool2d(-padded, 3, stride=1, padding=0).squeeze()
    local_contrast = (local_max - local_min) / (local_max + local_min + 1e-8)
    return local_contrast.mean().item()


def compute_metrics(img: torch.Tensor) -> tuple[float, float]:
    img_u8 = img if img.dtype == torch.uint8 else tensor_to_uint8(img)
    return compute_nrss(img_u8), compute_local_contrast(img_u8)