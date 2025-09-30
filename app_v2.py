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

from diffusers import StableDiffusionInpaintPipeline
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
def load_sd15_inpaint(device: torch.device, dtype: torch.dtype, use_xformers: bool = False, cpu_offload: bool = True):
    """
    Load SD-1.5 inpaint with safe defaults:
    - keep safety checker ON (public-friendly)
    - DO NOT enable xFormers by default (avoid flash-attention kernel issues)
    - enable attention/vae slicing and (optional) CPU offload to save VRAM
    """
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        # keep safety checker enabled for public apps (do NOT pass safety_checker=None)
    )

    # move to device
    pipe = pipe.to(device if device.type == "cuda" else "cpu")

    # Memory-friendly features (safe on any GPU/CPU)
    pipe.enable_attention_slicing("max")   # split attention
    pipe.vae.enable_tiling()               # tile VAE decode
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    # xFormers is optional, and can crash on some combos. Try-enable then fallback.
    if use_xformers and device.type == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            # Fallback: disable xformers and keep slicing instead
            try:
                pipe.disable_xformers_memory_efficient_attention()
            except Exception:
                pass
            st.warning(f"xFormers not available/compatible, using standard attention instead. ({e})")

    # Optional CPU offload to reduce VRAM (slower but safer)
    if cpu_offload:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pass

    return pipe


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

def dilate_mask(mask01: np.ndarray, k: int = 5) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.uint8) * 255
    kernel = np.ones((k, k), np.uint8)
    m = cv2.dilate(m, kernel, iterations=1)
    return (m.astype(np.float32) / 255.0)

def feather_mask(mask01: np.ndarray, k: int = 17, sigma: float = 6.0) -> np.ndarray:
    m = np.clip(mask01.astype(np.float32), 0.0, 1.0)
    m = cv2.GaussianBlur(m, (k, k), sigma)
    return np.clip(m, 0.0, 1.0)

def sd15_inpaint_once(
    pipe,
    init_pil: Image.Image,
    mask_soft01: np.ndarray,  # 0~1 soft mask, 白=要重繪
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    strength: float,
    seed: int,
):
    # 目標尺寸 = 輸入大小
    W, H = init_pil.size
    if mask_soft01.shape != (H, W):
        mask_soft01 = cv2.resize(mask_soft01, (W, H), interpolation=cv2.INTER_CUBIC)
    mask_img = Image.fromarray((np.clip(mask_soft01, 0, 1) * 255).astype(np.uint8), mode="L")

    generator = torch.Generator(device=pipe.device if isinstance(pipe.device, torch.device) else "cpu").manual_seed(seed)

    # 回傳 dict 才拿得到 nsfw 標誌（不同版本可能無此欄，容錯處理）
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_pil,
        mask_image=mask_img,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
        return_dict=True,
    )
    img = out.images[0]
    # 強制與輸入相同大小（個別版本可能回 512×512）
    if img.size != (W, H):
        img = img.resize((W, H), Image.BICUBIC)

    nsfw = False
    if hasattr(out, "nsfw_content_detected") and out.nsfw_content_detected:
        # 部分版本提供這個欄位
        nsfw = bool(out.nsfw_content_detected[0])

    return img, nsfw


def is_nearly_black(pil_img: Image.Image, thr: float = 8.0) -> bool:
    arr = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    return float(arr.mean()) < thr


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
    min_area_ratio: float = 0.015,  # 至少佔 1.5% 影像才算有效
) -> Tuple[np.ndarray, bool]:
    """
    Returns:
        mask01: (H,W) in [0,1] float soft mask (尚未羽化)
        used_fallback: True 代表分割不可靠（將在推論處跳過白化）
    """
    to_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),  # 保留邊界
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = to_tensor(pil_img.convert("RGB")).unsqueeze(0).to(device)

    out = parser(x)
    logits = out[0] if isinstance(out, (list, tuple)) else out  # [1,C,H,W]
    parsing = logits.squeeze(0).argmax(0).detach().cpu().numpy()

    # 多個 hair 類別容錯
    mask = np.isin(parsing, np.array(hair_ids)).astype(np.float32)

    # 面積門檻檢查
    H, W = mask.shape
    area = mask.sum()
    used_fallback = False
    if area < (H * W * min_area_ratio):
        # 直接宣告 fallback，不做白化（避免你看到的白橢圓）
        used_fallback = True
        mask = np.zeros_like(mask, dtype=np.float32)

    return mask, used_fallback

def hair_brightness_bgr01(bgr01: np.ndarray, mask01: np.ndarray) -> float:
    if bgr01.shape[:2] != mask01.shape:
        bgr01 = cv2.resize(bgr01, (mask01.shape[1], mask01.shape[0]), interpolation=cv2.INTER_CUBIC)
    yuv = cv2.cvtColor(bgr01, cv2.COLOR_BGR2YUV)
    Y = yuv[..., 0]
    m = mask01.astype(bool)
    if m.sum() == 0:
        return 0.0
    return float((Y[m]).mean())


def feather_mask(mask01: np.ndarray, k: int = 9, sigma: float = 3.0) -> np.ndarray:
    """將二值遮罩羽化成 0~1 軟遮罩：先形態學開閉，再高斯模糊與正規化。"""
    m = mask01.astype(np.float32)
    if m.max() > 1.0:  # 容錯
        m = (m > 127.5).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    m = cv2.GaussianBlur(m, (k, k), sigma)
    m = np.clip(m, 0.0, 1.0)
    return m

def blend_to_white_soft(pil_img: Image.Image, soft_mask: np.ndarray, alpha: float) -> Image.Image:
    """使用軟遮罩做白化混合（避免硬邊）。"""
    if alpha <= 0:
        return pil_img
    bgr = _pil_to_bgr01(pil_img)
    if bgr.shape[:2] != soft_mask.shape:
        bgr = cv2.resize(bgr, (soft_mask.shape[1], soft_mask.shape[0]), interpolation=cv2.INTER_CUBIC)
    m = np.clip(soft_mask, 0.0, 1.0)
    m = (alpha * m)[..., None]  # (H,W,1)
    out = bgr * (1.0 - m) + 1.0 * m
    return _bgr01_to_pil(out)

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

st.sidebar.markdown("---")
st.sidebar.markdown("**Diffusion Inpaint (SD-1.5) for hair graying**")
use_sd15 = st.sidebar.checkbox("Use SD-1.5 inpaint for ages 6–9 (recommended)", value=True)
seed = st.sidebar.number_input("Inpaint Seed", min_value=0, max_value=2_147_483_647, value=1234, step=1)

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
SD15 = None
if use_sd15:
    with st.spinner("Loading SD-1.5 Inpaint ..."):
        sd_dtype = torch.float16 if (device.type == "cuda") else torch.float32
        SD15 = load_sd15_inpaint(device, sd_dtype)
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

def neutralize_hair_region(init_pil: Image.Image, soft_mask: np.ndarray, gray_level: float = 0.82) -> Image.Image:
    """
    先把髮區預處理成偏白灰，降低原圖顏色影響。
    gray_level: 0~1，越大越接近白；建議 0.78~0.88
    """
    W, H = init_pil.size
    if soft_mask.shape != (H, W):
        soft_mask = cv2.resize(soft_mask, (W, H), interpolation=cv2.INTER_CUBIC)
    base = np.array(init_pil.convert("RGB")).astype(np.float32) / 255.0
    m = np.clip(soft_mask, 0.0, 1.0)[..., None]  # (H,W,1)
    target = np.full_like(base, gray_level, dtype=np.float32)  # 灰白
    mixed = base * (1.0 - m) + target * m
    out = (np.clip(mixed, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")

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
    # 產生髮絲遮罩（若分割不可靠，used_fallback=True）
    parser_dev = next(Parser.parameters()).device
    mask01, used_fallback = generate_hair_mask_bisenet(
        pil_img,
        Parser,
        img_size=H,
        device=parser_dev,
        hair_ids=[17]  # 可改成 [17, 13] 視權重而定
    )

    # 羽化成軟遮罩，避免硬邊
    soft_mask = feather_mask(mask01, k=11, sigma=4.0)

    # optional graying for 6/7/8/9
    if apply_graying:
        # 產生軟遮罩：先略微膨脹避免漏邊，再羽化避免硬邊
        mask_grow  = dilate_mask(mask01, k=9)
        soft_mask  = feather_mask(mask_grow, k=21, sigma=7.0)

        # 每個年齡組遞增的 Inpaint 參數（你可在側欄再做成可調）
        grp = {
            6: dict(
                prompt="more gray hair, visible silver strands, natural, realistic, high detail, photorealistic hair texture",
                strength=0.58, guidance=9.5, steps=38
            ),
            7: dict(
                prompt="gray hair with white streaks, salt-and-pepper trending to white, natural, realistic, high detail, photorealistic hair texture",
                strength=0.72, guidance=13.0, steps=40
            ),
            8: dict(
                prompt="mostly white hair, silver-white, minimal remaining pigment, realistic, very high detail, finely detailed hair texture, photorealistic",
                strength=0.86, guidance=15, steps=44
            ),
            9: dict(
                prompt="pure white hair, snow-white, no visible pigment, realistic, ultra high detail, finely detailed hair texture, photorealistic",
                strength=0.92, guidance=15, steps=48
            ),
        }
        # 更嚴格的負向提示，避免回染與奇怪色偏
        neg = (
            "black hair, brown hair, blonde, red hair, colored hair, highlights, "
            "green, blue, purple, saturated colors, cartoon, painting, watercolor, "
            "lowres, artifacts, color bleed, unrealistic tint, overexposed skin"
        )


        modified = []
        warned_once = False
        for t_idx, t in enumerate(targets):
            img_t = transforms.ToPILImage()(Yv[t_idx])  # [0,1] PIL
            if t in (6, 7, 8, 9):
                if SD15 is None or not use_sd15:
                    # 備援：線性白化
                    alpha = ALPHAS[t]; y_floor = YFLOORS[t]
                    img_t = apply_blend_to_white(img_t, soft_mask, alpha=alpha)
                    img_t = ensure_brightness_floor(img_t, soft_mask, target_Y=y_floor, max_extra=0.20)
                else:
                    # ★ 先將髮區預去色 → 灰白，增加可控性
                    img_neutral = neutralize_hair_region(img_t, soft_mask, gray_level=0.94)

                    # 一次 inpaint（更強參數）
                    p = grp[t]
                    gen_img, nsfw = sd15_inpaint_once(
                        SD15, img_neutral, soft_mask,
                        prompt=p["prompt"], negative_prompt=neg,
                        steps=p["steps"], guidance=p["guidance"],
                        strength=p["strength"], seed=seed + t,
                    )

                    # NSFW/黑圖回退
                    if nsfw or is_nearly_black(gen_img):
                        if not warned_once:
                            st.warning("SD safety filter triggered on hair inpaint. Falling back to blended graying for this image.")
                            warned_once = True
                        alpha = ALPHAS[t]; y_floor = YFLOORS[t]
                        gen_img = apply_blend_to_white(img_t, soft_mask, alpha=alpha)
                        gen_img = ensure_brightness_floor(gen_img, soft_mask, target_Y=y_floor, max_extra=0.20)

                    # ★ 8/9 再做第二段短 inpaint，完全拉白
                    if t in (8, 9) and SD15 is not None and use_sd15:
                        p2_prompt = "pure white hair, snow-white hair, no pigment, very high detail, realistic, photorealistic hair"
                        gen_img2, nsfw2 = sd15_inpaint_once(
                            SD15, gen_img, soft_mask,
                            prompt=p2_prompt, negative_prompt=neg,
                            steps=28, guidance=p["guidance"] + 0.5,
                            strength=min(0.88 if t == 8 else 0.94, 0.96),
                            seed=seed + 10_000 + t,
                        )
                        if not (nsfw2 or is_nearly_black(gen_img2)):
                            gen_img = gen_img2  # 只在成功時採用二段式

                    img_t = gen_img

            # ★ 統一尺寸
            if img_t.size != (W, H):
                img_t = img_t.resize((W, H), Image.BICUBIC)
            modified.append(transforms.ToTensor()(img_t).clamp(0, 1))

        Yv = torch.stack(modified, dim=0)



    grid = utils.make_grid(Yv, nrow=grid_nrow)  # [3,H',W']
    return grid, Yv

# ---------------------- Run ---------------------- #

col_run, col_save = st.columns([1, 1])
run_clicked = col_run.button("🚀 Run Inference", disabled=not (files and load_ok))
clear_clicked = col_save.button("🧹 Clear Output")
if clear_clicked:
    st.rerun()

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
