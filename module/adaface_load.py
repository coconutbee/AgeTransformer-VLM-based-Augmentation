import torch
import torch.nn.functional as F
import module.net as net
import os

def load_adaface_backbone(architecture: str = "ir_50",
                           ckpt_path: str = "./models/adaface_ir50_ms1mv2.ckpt",
                           device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build backbone (e.g., ir_50)
    backbone = net.build_model(architecture)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"AdaFace checkpoint not found: {ckpt_path}")
    # load with weights_only=False to accept Lightning checkpoint structure (trusting source)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # unwrap if wrapped
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # clean prefix and skip head.* (AdaFace margin head) since backbone only expects its own keys
    cleaned = {}
    for k, v in state_dict.items():
        key = k
        if k.startswith("model."):
            key = k[len("model."):]

        if key.startswith("head."):
            continue  # skip AdaFace head parameters
        cleaned[key] = v

    # load with strict=False to avoid failures from minor mismatches, log issues
    missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[AdaFace loader] missing keys in backbone: {missing}")
    if unexpected:
        print(f"[AdaFace loader] unexpected keys skipped: {unexpected}")

    backbone = backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False  # freeze

    return backbone, device

def preprocess_for_adaface(img: torch.Tensor):
    """
    img: [B,3,H,W] in RGB [0,1]
    Returns preprocessed tensor for AdaFace backbone: BGR, normalized to [-1,1], resized to 112x112.
    """
    # resize to expected 112x112 (AdaFace backbones usually trained on 112x112)
    img = F.interpolate(img, size=(112, 112), mode="bilinear", align_corners=False)
    # RGB -> BGR
    img = img[:, [2, 1, 0], :, :]
    # normalize as (x - 0.5)/0.5 => [-1,1]
    img = (img - 0.5) / 0.5
    return img

def extract_embedding_output(output):
    # Backbone may return tuple; take first element if so
    if isinstance(output, tuple) or isinstance(output, list):
        return output[0]
    return output

def get_normalized_embedding(raw_emb: torch.Tensor, eps: float = 1e-6):
    raw_emb = extract_embedding_output(raw_emb)
    norms = torch.norm(raw_emb, p=2, dim=1, keepdim=True)
    return raw_emb / norms.clamp(min=eps)  # L2-normalized

def encode_with_adaface(backbone, img: torch.Tensor):
    """
    img: [B,3,H,W] RGB [0,1], aligned face images.
    returns: normalized embedding [B, D]
    """
    proc = preprocess_for_adaface(img)
    raw = backbone(proc)  # maybe tuple
    normed = get_normalized_embedding(raw)
    return normed  # [B, D]
