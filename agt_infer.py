#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, utils
from PIL import Image
import numpy as np

# ===== import your project modules =====
from model import Generator

# ---------- Utils ----------
def is_img(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(path: Path):
    if path.is_file():
        return [path] if is_img(path) else []
    return sorted([p for p in path.rglob("*") if is_img(p)])

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),  # [-1, 1]
    ])

def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) / 2.0

def one_hot(indices: torch.LongTensor, num_classes: int) -> torch.FloatTensor:
    return torch.eye(num_classes, device=indices.device, dtype=torch.float32).index_select(0, indices)

def load_generator(ckpt_path: str, img_size: int, latent: int, n_mlp: int, channel_multiplier: int, device: torch.device):
    G = Generator(img_size, latent, n_mlp, channel_multiplier=channel_multiplier).to(device)
    G.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = None
    if "g_ema" in ckpt:
        state_dict = ckpt["g_ema"]
        print("[INFO] Loaded g_ema from checkpoint.")
    elif "g" in ckpt:
        state_dict = ckpt["g"]
        print("[INFO] Loaded g from checkpoint (g_ema not found).")
    else:
        raise RuntimeError("No 'g_ema' or 'g' found in checkpoint.")

    missing, unexpected = G.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    return G

# ---------- Core Inference ----------
@torch.inference_mode()
def run_inference(
    ckpt: str,
    input_path: str,
    out_dir: str,
    size: int = 128,
    latent: int = 512,
    n_mlp: int = 8,
    channel_multiplier: int = 2,
    num_classes: int = 10,
    targets: list = None,
    nrow: int = 5,
    fp16: bool = False,
    bf16: bool = False,
    device_str: str = None,
):
    device = torch.device(device_str) if device_str else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    G = load_generator(ckpt, size, latent, n_mlp, channel_multiplier, device)

    if fp16 and bf16:
        print("[WARN] Both fp16 and bf16 set. Using bf16.")
        fp16 = False

    dtype = torch.float32
    if fp16 and device.type == "cuda":
        dtype = torch.float16
    if bf16 and device.type != "cpu":
        dtype = torch.bfloat16

    tfm = build_transform(size)
    input_path = Path(input_path)
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Found {len(images)} image(s). Writing results to: {out_dir}")

    if targets is None or len(targets) == 0:
        targets = list(range(num_classes))
    targets = [int(t) for t in targets]
    assert all(0 <= t < num_classes for t in targets), "targets must be within [0, num_classes-1]"

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]

        outputs = []
        for t in targets:
            cond = one_hot(torch.tensor([t], device=device), num_classes)  # [1,10]
            out, _, _ = G(x, cond)
            outputs.append(out)

        outs = torch.cat(outputs, dim=0)                      # [T,3,H,W]
        grid = utils.make_grid(denorm(outs).cpu(), nrow=nrow) # [3,H',W']
        stem = img_path.stem
        save_to = Path(out_dir) / f"{stem}_ages_{'-'.join(map(str,targets))}.png"
        utils.save_image(grid, str(save_to))
        print(f"[OK] Saved: {save_to}")

    print("[DONE] Inference finished.")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="AgeTransformer Inference")
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt (should contain g_ema).")
    p.add_argument("--input", type=str, required=True, help="Path to an image file or a folder of images.")
    p.add_argument("--out", type=str, default="outputs", help="Output folder.")
    p.add_argument("--size", type=int, default=128, help="Input resolution (must match training).")
    p.add_argument("--latent", type=int, default=512)
    p.add_argument("--n_mlp", type=int, default=8)
    p.add_argument("--channel_multiplier", type=int, default=2)
    p.add_argument("--num_classes", type=int, default=10, help="Number of age classes used in training.")
    p.add_argument("--targets", type=int, nargs="*", default=None, help="Target age IDs, e.g. --targets 0 3 5 9")
    p.add_argument("--nrow", type=int, default=5, help="Grid columns when saving multiple targets.")
    p.add_argument("--fp16", action="store_true", help="Use float16 (CUDA only).")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (non-CPU).")
    p.add_argument("--device", type=str, default=None, help="Force device, e.g. 'cpu' or 'cuda:0'")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        ckpt=args.ckpt,
        input_path=args.input,
        out_dir=args.out,
        size=args.size,
        latent=args.latent,
        n_mlp=args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        num_classes=args.num_classes,
        targets=args.targets,
        nrow=args.nrow,
        fp16=args.fp16,
        bf16=args.bf16,
        device_str=args.device,
    )
