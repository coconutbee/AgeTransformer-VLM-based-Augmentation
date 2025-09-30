#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit demo app for AgeTransformer inference.

Run:
    streamlit run app.py

Notes:
- Expects a training-compatible checkpoint (.pt) that contains 'g_ema' (preferred) or 'g'.
- Generator forward signature assumed: out, _, _ = G(x, cond)
    where x:    [B, 3, H, W] in [-1, 1]
          cond: [B, num_classes] one-hot
- Preprocessing matches training: center-crop -> resize -> ToTensor -> Normalize((0.5,)*3, (0.5,)*3)
"""

import io
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import torch
from torch import nn
from torchvision import transforms, utils
from PIL import Image

# ===== import your project modules =====
# Ensure model.py (with class Generator) is importable from repo root.
from model import Generator


# ---------------------- UI Helpers ---------------------- #

st.set_page_config(
    page_title="AgeTransformer Demo",
    page_icon="🧓",
    layout="wide",
)

@st.cache_resource(show_spinner=False)
def load_generator(
    ckpt_path: str,
    img_size: int,
    latent: int,
    n_mlp: int,
    channel_multiplier: int,
    device: torch.device,
) -> nn.Module:
    """Load Generator and weights (prefer g_ema)."""
    G = Generator(img_size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
    G.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = None
    if "g_ema" in ckpt:
        state_dict = ckpt["g_ema"]
        st.info("Loaded 'g_ema' from checkpoint.")
    elif "g" in ckpt:
        state_dict = ckpt["g"]
        st.warning("Fallback to 'g' (no 'g_ema' found).")
    else:
        raise RuntimeError("No 'g_ema' or 'g' found in checkpoint.")

    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"Missing keys (first 5): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        st.warning(f"Unexpected keys (first 5): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    return G


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Lambda(center_crop_square),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),  # [-1,1]
    ])


def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1.0) / 2.0


def one_hot(indices: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    return torch.eye(num_classes, device=indices.device, dtype=torch.float32).index_select(0, indices)


def save_image_to_bytes(tensor: torch.Tensor) -> bytes:
    """tensor: [3, H, W] in [0,1]"""
    buf = io.BytesIO()
    pil = transforms.ToPILImage()(tensor)
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------- Sidebar ---------------------- #

st.sidebar.header("⚙️ Settings")

# Device & dtype
use_cuda = torch.cuda.is_available()
device_str = st.sidebar.selectbox("Device", ["cuda"] if use_cuda else ["cpu"])
device = torch.device(device_str)

mixed_precision = st.sidebar.selectbox("Mixed Precision (if on GPU)", ["none", "fp16", "bf16"])

# Model hyper-params (must match training)
img_size = st.sidebar.number_input("Input Size", min_value=64, max_value=1024, value=128, step=8)
latent = st.sidebar.number_input("Latent Dim", min_value=128, max_value=2048, value=512, step=64)
n_mlp = st.sidebar.number_input("n_mlp", min_value=1, max_value=16, value=8, step=1)
channel_multiplier = st.sidebar.number_input("Channel Multiplier", min_value=1, max_value=4, value=2, step=1)
num_classes = st.sidebar.number_input("Num Age Classes", min_value=2, max_value=32, value=10, step=1)

# Checkpoint
default_ckpt = "checkpoint/600000.pt"
ckpt_path = st.sidebar.text_input("Checkpoint (.pt)", value=default_ckpt)

# Targets
st.sidebar.markdown("**Target Age IDs** (0 ~ num_classes-1)")
default_targets = list(range(10))
targets_raw = st.sidebar.text_input("Comma-separated IDs", value=",".join(map(str, default_targets[:num_classes])))
try:
    targets = [int(t.strip()) for t in targets_raw.split(",") if t.strip() != ""]
except Exception:
    targets = list(range(num_classes))
targets = [t for t in targets if 0 <= t < num_classes]
if len(targets) == 0:
    st.sidebar.warning("No valid targets specified; fallback to all classes.")
    targets = list(range(num_classes))

grid_nrow = st.sidebar.number_input("Grid columns (nrow)", min_value=1, max_value=16, value=min(5, len(targets)), step=1)

# Batch size for UI processing (per click)
batch_size = st.sidebar.number_input("Internal Batch Size", min_value=1, max_value=16, value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips**")
st.sidebar.caption(
    "• Provide a checkpoint that contains 'g_ema' (preferred). "
    "• Inputs will be center-cropped and resized; try frontal face photos."
)


# ---------------------- Main UI ---------------------- #

st.title("🧓 AgeTransformer – Inference Demo")
st.write("Upload face images, select target age classes, and generate transformed results.")

# Uploader (multi-images)
files = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
)

# Load model button
G = None
load_ok = False
if ckpt_path and Path(ckpt_path).exists():
    try:
        with st.spinner("Loading generator..."):
            G = load_generator(
                ckpt_path=ckpt_path,
                img_size=img_size,
                latent=latent,
                n_mlp=n_mlp,
                channel_multiplier=channel_multiplier,
                device=device,
            )
            load_ok = True
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
else:
    st.warning(f"Checkpoint not found: {ckpt_path}")

tfm = build_transform(img_size)

# Inference controls
col_run, col_save = st.columns([1, 1])
run_clicked = col_run.button("🚀 Run Inference", disabled=not (files and load_ok))
clear_clicked = col_save.button("🧹 Clear Output")

if clear_clicked:
    st.experimental_rerun()

result_cont = st.container()

# ---------------------- Inference ---------------------- #

def get_dtype(device: torch.device, mp: str) -> torch.dtype:
    if device.type == "cuda":
        if mp == "fp16":
            return torch.float16
        if mp == "bf16":
            return torch.bfloat16
    return torch.float32

@torch.inference_mode()
def infer_one_image(pil_img: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        grid_img:  [3, H', W'] in [0,1]
        cat_imgs:  [T, 3, H, W] in [0,1] (each target)
    """
    dtype = get_dtype(device, mixed_precision)
    x = tfm(pil_img).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]
    outs = []
    for t in targets:
        cond = one_hot(torch.tensor([t], device=device), num_classes)  # [1, num_classes]
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=dtype):
                y, _, _ = G(x, cond)
        else:
            y, _, _ = G(x, cond)
        outs.append(y)
    Y = torch.cat(outs, dim=0)                 # [T,3,H,W] in [-1,1]
    Yv = denorm(Y).clamp(0, 1).cpu()           # [T,3,H,W] in [0,1]
    grid = utils.make_grid(Yv, nrow=grid_nrow) # [3,H',W']
    return grid, Yv


if run_clicked:
    if not files:
        st.warning("Please upload at least one image.")
    elif not load_ok:
        st.error("Model not loaded.")
    else:
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                st.error(f"Cannot open image: {f.name}")
                continue

            with st.spinner(f"Processing {f.name} ..."):
                grid_img, per_target = infer_one_image(img)

            # Show
            st.subheader(f.name)
            st.image(
                grid_img.permute(1, 2, 0).numpy(),
                caption=f"Targets: {targets} (nrow={grid_nrow})",
                use_column_width=True,
            )

            # Download grid
            grid_bytes = save_image_to_bytes(grid_img)
            st.download_button(
                label="Download Grid PNG",
                data=grid_bytes,
                file_name=f"{Path(f.name).stem}_ages_{'-'.join(map(str, targets))}.png",
                mime="image/png",
            )

            # Optional: per-target individual downloads (collapsed)
            with st.expander("Show & download individual results"):
                for t_idx, t in enumerate(targets):
                    st.markdown(f"**Target {t}**")
                    t_img = per_target[t_idx]
                    st.image(t_img.permute(1, 2, 0).numpy(), use_column_width=False)
                    t_bytes = save_image_to_bytes(t_img)
                    st.download_button(
                        label=f"Download target {t}",
                        data=t_bytes,
                        file_name=f"{Path(f.name).stem}_age_{t}.png",
                        mime="image/png",
                        key=f"dl_{f.name}_{t}",
                    )

