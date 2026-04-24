import numpy as np
import streamlit as st
import torch
from PIL import Image
from typing import Callable

from app import (
    pil_to_tensor,
    is_effectively_grayscale,
    abmhe,
    ghe,
    bbhe,
    clahe,
    poshe,
    abmhe_rgb_perchannel,
    abmhe_rgb_ycbcr,
    compute_metrics,
)

st.set_page_config(page_title="ABMHE Demo", layout="wide")
st.title("ABMHE Image Enhancement Demo")

st.write("Upload an image, choose a method, and preview results.")

FIXED_BLOCK_SIZE = 64


def to_pil_gray(image_tensor: torch.Tensor) -> Image.Image:
    arr = image_tensor.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(arr, mode="L")


def to_pil_rgb(image_tensor: torch.Tensor) -> Image.Image:
    arr = image_tensor.clamp(0, 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def compute_luminance_metrics(image_pil: Image.Image, device: torch.device) -> tuple[float, float]:
    gray_np = np.array(image_pil.convert("L"), dtype=np.uint8)
    gray_tensor = torch.from_numpy(gray_np).to(device)
    return compute_metrics(gray_tensor)


def run_all_grayscale_methods(gray_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "Original": gray_tensor,
        "GHE": ghe(gray_tensor),
        "BBHE": bbhe(gray_tensor),
        "CLAHE": clahe(gray_tensor),
        "POSHE": poshe(gray_tensor, block_size=FIXED_BLOCK_SIZE),
        "ABMHE": abmhe(gray_tensor, block_size=FIXED_BLOCK_SIZE),
    }


def render_image_grid(
    results: dict[str, torch.Tensor],
    columns: int,
    tensor_to_pil: Callable[[torch.Tensor], Image.Image],
) -> None:
    grid_columns = st.columns(columns)
    for idx, (name, image_tensor) in enumerate(results.items()):
        image_pil = tensor_to_pil(image_tensor)
        with grid_columns[idx % columns]:
            st.image(image_pil, caption=name, use_container_width=True)


def metrics_table_from_results(
    results: dict[str, torch.Tensor],
    tensor_to_pil: Callable[[torch.Tensor], Image.Image],
    device: torch.device,
    include_original: bool = False,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for name, image_tensor in results.items():
        if not include_original and name == "Original":
            continue
        image_pil = tensor_to_pil(image_tensor)
        nrss, contrast = compute_luminance_metrics(image_pil, device)
        rows.append({"Method": name, "NRSS": round(nrss, 4), "Contrast": round(contrast, 4)})
    return rows


def run_single_method(
    method_name: str,
    gray_tensor: torch.Tensor,
    rgb_tensor: torch.Tensor | None,
) -> tuple[Image.Image, str]:
    gray_method_map: dict[str, Callable[[], torch.Tensor]] = {
        "ABMHE": lambda: abmhe(gray_tensor, block_size=FIXED_BLOCK_SIZE),
        "GHE": lambda: ghe(gray_tensor),
        "BBHE": lambda: bbhe(gray_tensor),
        "CLAHE": lambda: clahe(gray_tensor),
        "POSHE": lambda: poshe(gray_tensor, block_size=FIXED_BLOCK_SIZE),
    }

    if method_name in gray_method_map:
        return to_pil_gray(gray_method_map[method_name]()), "gray"

    if method_name == "ABMHE RGB (per-channel)":
        if rgb_tensor is None:
            raise ValueError("RGB enhancement is not available for grayscale images.")
        enhanced_rgb = abmhe_rgb_perchannel(rgb_tensor, block_size=FIXED_BLOCK_SIZE)
        return to_pil_rgb(enhanced_rgb), "rgb"

    if method_name == "ABMHE RGB (YCbCr-Y)":
        if rgb_tensor is None:
            raise ValueError("RGB enhancement is not available for grayscale images.")
        enhanced_rgb = abmhe_rgb_ycbcr(rgb_tensor, block_size=FIXED_BLOCK_SIZE)
        return to_pil_rgb(enhanced_rgb), "rgb"

    raise ValueError(f"Unknown or invalid method: {method_name}")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is None:
    st.stop()

pil_img = Image.open(uploaded)
is_color = not is_effectively_grayscale(pil_img)

gray_t = pil_to_tensor(pil_img.convert("L"))
rgb_t = pil_to_tensor(pil_img.convert("RGB")) if is_color else None
device = gray_t.device

single_method_options = ["ABMHE", "GHE", "BBHE", "CLAHE", "POSHE"]
if is_color:
    single_method_options += ["ABMHE RGB (per-channel)", "ABMHE RGB (YCbCr-Y)"]

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Single method", single_method_options, index=0)
    c1, c2 = st.columns(2)
    with c1:
        run_single = st.button("Run single", type="primary", use_container_width=True)
    with c2:
        run_all = st.button("Run all", use_container_width=True)

if not (run_single or run_all):
    st.stop()

if run_single:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(pil_img, use_container_width=True)

    try:
        out_pil, _ = run_single_method(method, gray_t, rgb_t)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with col2:
        st.subheader(f"Enhanced: {method}")
        st.image(out_pil, use_container_width=True)

    nrss, contrast = compute_luminance_metrics(out_pil, device)
    st.markdown(f"**NRSS:** `{nrss:.4f}`  |  **Local Contrast:** `{contrast:.4f}`")

if run_all:
    st.subheader("Input")
    st.image(pil_img, use_container_width=True)

    st.subheader("Grayscale — all algorithms")
    gray_results = run_all_grayscale_methods(gray_t)
    render_image_grid(gray_results, columns=3, tensor_to_pil=to_pil_gray)
    gray_metrics_rows = metrics_table_from_results(
        gray_results,
        tensor_to_pil=to_pil_gray,
        device=device,
        include_original=False,
    )
    st.dataframe(gray_metrics_rows, use_container_width=True)

    if is_color:
        st.subheader("RGB")
        rgb_results = {
            "Original": rgb_t,
            "ABMHE RGB (per-channel)": abmhe_rgb_perchannel(rgb_t, block_size=FIXED_BLOCK_SIZE),
            "ABMHE RGB (YCbCr-Y)": abmhe_rgb_ycbcr(rgb_t, block_size=FIXED_BLOCK_SIZE),
        }

        render_image_grid(rgb_results, columns=3, tensor_to_pil=to_pil_rgb)
        rgb_metrics_rows = metrics_table_from_results(
            rgb_results,
            tensor_to_pil=to_pil_rgb,
            device=device,
            include_original=True,
        )
        st.dataframe(rgb_metrics_rows, use_container_width=True)