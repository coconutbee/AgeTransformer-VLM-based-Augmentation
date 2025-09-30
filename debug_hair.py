import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

# 你的 BiSeNet 定義
from module.face_model import BiSeNet

# ==== 可調參 ====
WEIGHT_PATH = "./checkpoint/79999_iter.pth"
TEST_IMAGE = "/media/ee303/disk1/dataset/iu.jpg"          # 換成你的某一張輸入圖
SAVE_DIR   = "mask_debug_out"
N_CLASSES  = 19                  # 多數人臉分割權重是 19 類
HAIR_IDS   = [17]                # 若抓不到，可改成 [17, 13] 試試（不同標籤集）
IMG_SIZE   = 512

os.makedirs(SAVE_DIR, exist_ok=True)

def load_bisenet(weight_path: str, n_classes: int, device: torch.device):
    net = BiSeNet(n_classes=n_classes)
    sd = torch.load(weight_path, map_location="cpu")
    # 兼容各種包裝方式
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # 去掉可能的 "module." 前綴
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_sd[nk] = v
    missing, unexpected = net.load_state_dict(new_sd, strict=False)
    print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:   print("  missing (first 10):", missing[:10])
    if unexpected:print("  unexpected (first 10):", unexpected[:10])
    net.to(device).eval()
    return net

def build_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1,1]
    ])

def colorize_label(parsing: np.ndarray, n_classes: int) -> np.ndarray:
    # 簡單上色方便肉眼檢查
    np.random.seed(123)
    palette = np.random.randint(0, 255, size=(n_classes, 3), dtype=np.uint8)
    h, w = parsing.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color[parsing == c] = palette[c]
    return color

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    # 讀圖
    assert os.path.exists(TEST_IMAGE), f"TEST_IMAGE not found: {TEST_IMAGE}"
    pil = Image.open(TEST_IMAGE).convert("RGB")

    # 準備模型
    assert os.path.exists(WEIGHT_PATH), f"WEIGHT_PATH not found: {WEIGHT_PATH}"
    net = load_bisenet(WEIGHT_PATH, N_CLASSES, device)

    # 前處理
    tfm = build_transform(IMG_SIZE)
    x = tfm(pil).unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        out = net(x)  # 通常返回 tuple(list)
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        # [1,C,H,W]
        print("[shape]", tuple(logits.shape))
        parsing = logits.squeeze(0).argmax(0).detach().cpu().numpy().astype(np.uint8)

    # 檢查有哪些類別被預測到
    uq = np.unique(parsing)
    print("[unique labels]", uq.tolist())

    # 存彩色分割圖
    color = colorize_label(parsing, N_CLASSES)
    cv2.imwrite(os.path.join(SAVE_DIR, "parsing_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    # 做 hair mask（允許多 hair id）
    hair_mask = np.zeros_like(parsing, dtype=np.uint8)
    for hid in HAIR_IDS:
        hair_mask[parsing == hid] = 255

    # 若太小或全黑，給備援橢圓
    nonzero = int((hair_mask > 0).sum())
    print("[hair pixels]", nonzero)
    if nonzero < 50:
        print("[warn] hair region too small/empty -> use fallback ellipse")
        h, w = hair_mask.shape
        mask = np.zeros((h, w), np.uint8)
        center = (w // 2, int(h * 0.35))
        axes = (int(w * 0.35), int(h * 0.25))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        hair_mask = mask

    cv2.imwrite(os.path.join(SAVE_DIR, "hair_mask.png"), hair_mask)
    print("[ok] saved:", os.path.join(SAVE_DIR, "hair_mask.png"))
    print("[ok] saved:", os.path.join(SAVE_DIR, "parsing_color.png"))

if __name__ == "__main__":
    main()
