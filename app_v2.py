#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit demo app for AgeTransformer inference + automatic hair mask & graying (ages 6–9).

Run:
    streamlit run app.py
"""

import io
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import streamlit as st
import torch
from torch import nn
from torchvision import transforms, utils
from PIL import Image
import cv2

# ==== project modules ====
from model import Generator
from module.face_model import BiSeNet  # BiSeNet face parser (n_classes=19 by default)

# ---------------------- Streamlit base ---------------------- #

st.set_page_config(page_title="AgeTransformer Demo", page_icon="🧓", layout="wide")

# ---------------------- Model loading ---------------------- #

@st.cache_resource(show_spinner=False)
def load_generator(
    ckpt_path: str,
    img_size: int,
    latent: int,
    n_mlp: int,
    channel_multiplier: int,
    device: torch.device,
) -> nn.Module:
    G = Generator(img_size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
    G.eval()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("g_ema", ckpt.get("g", None))
    if state_dict is None:
        raise RuntimeError("No 'g_ema' or 'g' found in checkpoint.")
    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"Generator missing keys (first 5): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        st.warning(f"Generator unexpected keys (first 5): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    return G


@st.cache_resource(show_spinner=False)
def load_bisenet(weight_path: str, n_classes: int = 19, device: torch.device = torch.device("cpu")) -> nn.Module:
    net = BiSeNet(n_classes=n_classes)

    sd = torch.load(weight_path, map_location="cpu")
    # unwrap common wrappers
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # strip "module." prefix if present
    if isinstance(sd, dict):
        cleaned = {}
        for k, v in sd.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v
        sd = cleaned

    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        st.warning(f"BiSeNet missing keys (first 5): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        st.warning(f"BiSeNet unexpected keys (first 5): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    net.to(device).eval()
    return net

# ---------------------- Transforms ---------------------- #

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
    return (x.clamp(-1, 1) + 1.0) / 2.0

def one_hot(indices: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    return torch.eye(num_classes, device=indices.device, dtype=torch.float32).index_select(0, indices)

def save_image_to_bytes(tensor: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    pil = transforms.ToPILImage()(tensor)
    pil.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------- Hair mask & graying ---------------------- #

def _pil_to_bgr01(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _bgr01_to_pil(arr: np.ndarray) -> Image.Image:
    arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def _ellipse_fallback_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    center = (w // 2, int(h * 0.35))
    axes = (int(w * 0.35), int(h * 0.25))
    cv2.ellipse(mask, center, axes, angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)
    return mask.astype(np.float32)

@torch.inference_mode()
def generate_hair_mask_bisenet(
    pil_img: Image.Image,
    parser: nn.Module,
    img_size: int,
    device: torch.device,
    hair_ids: List[int] = [17],
) -> np.ndarray:
    """
    Returns binary mask (H,W) in {0,1}; uses fallback ellipse if empty.
    Supports multiple hair label IDs (e.g., [17, 13]) depending on the weights.
    """
    to_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(device)  # [1,3,H,W]

    out = parser(x)
    logits = out[0] if isinstance(out, (list, tuple)) else out  # [1,C,H,W]
    parsing = logits.squeeze(0).argmax(0).detach().cpu().numpy()

    hair = np.isin(parsing, np.array(hair_ids)).astype(np.float32)

    if hair.sum() < 50:  # too small or empty -> fallback
        hair = _ellipse_fallback_mask(img_size, img_size)

    return hair  # (H,W) in {0,1}

def hair_brightness_bgr01(bgr01: np.ndarray, mask01: np.ndarray) -> float:
    if bgr01.shape[:2] != mask01.shape:
        bgr01 = cv2.resize(bgr01, (mask01.shape[1], mask01.shape[0]), interpolation=cv2.INTER_CUBIC)
    yuv = cv2.cvtColor(bgr01, cv2.COLOR_BGR2YUV)
    Y = yuv[..., 0]
    m = mask01.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float((Y[m]).mean())

def apply_blend_to_white(pil_img: Image.Image, mask01: np.ndarray, alpha: float) -> Image.Image:
    if alpha <= 0:
        return pil_img
    bgr = _pil_to_bgr01(pil_img)
    if bgr.shape[:2] != mask01.shape:
        bgr = cv2.resize(bgr, (mask01.shape[1], mask01.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask3 = np.dstack([mask01]*3)
    out = bgr * (1.0 - alpha * mask3) + 1.0 * (alpha * mask3)
    return _bgr01_to_pil(out)

def ensure_brightness_floor(pil_img: Image.Image, mask01: np.ndarray, target_Y: float, max_extra: float = 0.25) -> Image.Image:
    bgr = _pil_to_bgr01(pil_img)
    curr = hair_brightness_bgr01(bgr, mask01)
    if curr >= target_Y:
        return pil_img
    extra = float(np.clip(target_Y - curr, 0.0, max_extra))
    mask3 = np.dstack([mask01]*3)
    out = bgr * (1.0 - extra * mask3) + 1.0 * (extra * mask3)
    return _bgr01_to_pil(out)

DEFAULT_ALPHA_BY_GROUP: Dict[int, float] = {6: 0.10, 7: 0.20, 8: 0.32, 9: 0.45}
DEFAULT_YFLOOR_BY_GROUP: Dict[int, float] = {6: 0.55, 7: 0.65, 8: 0.75, 9: 0.85}

# ---------------------- Sidebar ---------------------- #

st.sidebar.header("⚙️ Settings")

# devices
use_cuda = torch.cuda.is_available()
device_str = st.sidebar.selectbox("Device", ["cuda"] if use_cuda else ["cpu"])
device = torch.device(device_str)
mixed_precision = st.sidebar.selectbox("Mixed Precision (if on GPU)", ["none", "fp16", "bf16"])

# generator hparams
img_size = st.sidebar.number_input("Input Size", min_value=64, max_value=1024, value=128, step=8)
latent = st.sidebar.number_input("Latent Dim", min_value=128, max_value=2048, value=512, step=64)
n_mlp = st.sidebar.number_input("n_mlp", min_value=1, max_value=16, value=8, step=1)
channel_multiplier = st.sidebar.number_input("Channel Multiplier", min_value=1, max_value=4, value=2, step=1)
num_classes = st.sidebar.number_input("Num Age Classes", min_value=2, max_value=32, value=10, step=1)

# checkpoints
default_ckpt = "checkpoint/agetransformer.pt"
ckpt_path = st.sidebar.text_input("AgeTransformer checkpoint (.pt)", value=default_ckpt)

# BiSeNet weight path
default_bisenet = "./checkpoint/79999_iter.pth"
bisenet_path = st.sidebar.text_input("BiSeNet weights (.pth)", value=default_bisenet)

# hair label ids (comma-separated)
st.sidebar.markdown("**Hair label IDs (comma-separated)**")
hair_ids_raw = st.sidebar.text_input("IDs", value="17")
try:
    HAIR_IDS = [int(x.strip()) for x in hair_ids_raw.split(",") if x.strip() != ""]
except Exception:
    HAIR_IDS = [17]
if len(HAIR_IDS) == 0:
    HAIR_IDS = [17]

# targets
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

st.sidebar.markdown("---")
st.sidebar.markdown("**Hair graying (auto mask; ages 6–9)**")
apply_graying = st.sidebar.checkbox("Apply hair graying (monotonic for 6,7,8,9)", value=True)

alpha6 = st.sidebar.slider("Alpha for age 6", 0.0, 1.0, DEFAULT_ALPHA_BY_GROUP[6], 0.01)
alpha7 = st.sidebar.slider("Alpha for age 7", 0.0, 1.0, DEFAULT_ALPHA_BY_GROUP[7], 0.01)
alpha8 = st.sidebar.slider("Alpha for age 8", 0.0, 1.0, DEFAULT_ALPHA_BY_GROUP[8], 0.01)
alpha9 = st.sidebar.slider("Alpha for age 9", 0.0, 1.0, DEFAULT_ALPHA_BY_GROUP[9], 0.01)

y6 = st.sidebar.slider("Brightness floor Y for age 6", 0.0, 1.0, DEFAULT_YFLOOR_BY_GROUP[6], 0.01)
y7 = st.sidebar.slider("Brightness floor Y for age 7", 0.0, 1.0, DEFAULT_YFLOOR_BY_GROUP[7], 0.01)
y8 = st.sidebar.slider("Brightness floor Y for age 8", 0.0, 1.0, DEFAULT_YFLOOR_BY_GROUP[8], 0.01)
y9 = st.sidebar.slider("Brightness floor Y for age 9", 0.0, 1.0, DEFAULT_YFLOOR_BY_GROUP[9], 0.01)

ALPHAS = {6: alpha6, 7: alpha7, 8: alpha8, 9: alpha9}
YFLOORS = {6: y6, 7: y7, 8: y8, 9: y9}

if not (ALPHAS[6] < ALPHAS[7] < ALPHAS[8] < ALPHAS[9]):
    st.sidebar.warning("Alphas are not strictly increasing (6 < 7 < 8 < 9).")
if not (YFLOORS[6] < YFLOORS[7] < YFLOORS[8] < YFLOORS[9]):
    st.sidebar.warning("Brightness floors are not strictly increasing (6 < 7 < 8 < 9).")

st.sidebar.markdown("---")
st.sidebar.caption(
    "• Auto-generates hair masks with BiSeNet (IDs configurable; default 17). "
    "• If parsing fails or hair is tiny, a fallback ellipse mask is used. "
    "• Graying is applied only for targets 6/7/8/9."
)

# ---------------------- Main UI ---------------------- #

st.title("🧓 AgeTransformer – Inference Demo (auto hair mask & graying)")
st.write("Upload face images, select target age classes, and generate transformed results.")

files = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    accept_multiple_files=True,
)

# load models
G = None
Parser = None
load_ok = False
if ckpt_path and Path(ckpt_path).exists():
    try:
        with st.spinner("Loading generator..."):
            G = load_generator(ckpt_path, img_size, latent, n_mlp, channel_multiplier, device)
        with st.spinner("Loading BiSeNet hair parser..."):
            # run BiSeNet on GPU only if available; otherwise CPU
            parser_device = device if device.type == "cuda" else torch.device("cpu")
            Parser = load_bisenet(bisenet_path, n_classes=19, device=parser_device)
        load_ok = True
    except Exception as e:
        st.error(f"Failed to load model(s): {e}")
else:
    st.warning(f"Checkpoint not found: {ckpt_path}")

tfm = build_transform(img_size)

# ---------------------- Inference ---------------------- #

def _get_dtype(device: torch.device, mp: str) -> torch.dtype:
    if device.type == "cuda":
        if mp == "fp16": return torch.float16
        if mp == "bf16": return torch.bfloat16
    return torch.float32

@torch.inference_mode()
def infer_one_image(pil_img: Image.Image):
    """
    Returns:
        grid_img: [3, H', W'] in [0,1]
        out_imgs: [T, 3, H, W] in [0,1] after optional hair graying
    """
    dtype = _get_dtype(device, mixed_precision)
    x = tfm(pil_img).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]

    outs = []
    use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else torch.no_grad()
    with ctx:
        for t in targets:
            cond = one_hot(torch.tensor([t], device=device), num_classes)  # [1, num_classes]
            y, _, _ = G(x, cond)  # [-1,1]
            outs.append(y)

    Y = torch.cat(outs, dim=0)             # [T,3,H,W] in [-1,1]
    Yv = denorm(Y).clamp(0, 1).cpu()       # [T,3,H,W] in [0,1]
    T, _, H, W = Yv.shape

    # auto hair mask (always generated; guaranteed non-empty)
    # use parser's own device
    parser_dev = next(Parser.parameters()).device
    mask01 = generate_hair_mask_bisenet(
        pil_img, Parser, img_size=H, device=parser_dev, hair_ids=HAIR_IDS
    )

    # optional graying for 6/7/8/9
    if apply_graying:
        modified = []
        for t_idx, t in enumerate(targets):
            img_t = transforms.ToPILImage()(Yv[t_idx])  # [0,1] PIL
            if t in (6, 7, 8, 9):
                alpha = ALPHAS[t]
                y_floor = YFLOORS[t]
                img_t = apply_blend_to_white(img_t, mask01, alpha=alpha)
                img_t = ensure_brightness_floor(img_t, mask01, target_Y=y_floor, max_extra=0.25)
            modified.append(transforms.ToTensor()(img_t).clamp(0,1))
        Yv = torch.stack(modified, dim=0)

    grid = utils.make_grid(Yv, nrow=grid_nrow)  # [3,H',W']
    return grid, Yv

# ---------------------- Run ---------------------- #

col_run, col_save = st.columns([1, 1])
run_clicked = col_run.button("🚀 Run Inference", disabled=not (files and load_ok))
clear_clicked = col_save.button("🧹 Clear Output")
if clear_clicked:
    st.experimental_rerun()

if run_clicked:
    if not files:
        st.warning("Please upload at least one image.")
    elif not load_ok:
        st.error("Model(s) not loaded.")
    else:
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
            except Exception:
                st.error(f"Cannot open image: {f.name}")
                continue

            with st.spinner(f"Processing {f.name} ..."):
                grid_img, per_target = infer_one_image(img)

            st.subheader(f.name)
            st.image(
                grid_img.permute(1, 2, 0).numpy(),
                caption=f"Targets: {targets} (nrow={grid_nrow})",
                use_container_width=True,
            )

            grid_bytes = save_image_to_bytes(grid_img)
            st.download_button(
                label="Download Grid PNG",
                data=grid_bytes,
                file_name=f"{Path(f.name).stem}_ages_{'-'.join(map(str, targets))}.png",
                mime="image/png",
            )

            with st.expander("Show & download individual results"):
                for t_idx, t in enumerate(targets):
                    st.markdown(f"**Target {t}**")
                    t_img = per_target[t_idx]
                    st.image(t_img.permute(1, 2, 0).numpy(), use_container_width=False)
                    t_bytes = save_image_to_bytes(t_img)
                    st.download_button(
                        label=f"Download target {t}",
                        data=t_bytes,
                        file_name=f"{Path(f.name).stem}_age_{t}.png",
                        mime="image/png",
                        key=f"dl_{f.name}_{t}",
                    )
