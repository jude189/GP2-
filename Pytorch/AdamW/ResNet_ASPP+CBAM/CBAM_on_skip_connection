import os
import math
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# =========================
# 0) KAGGLE PATHS + CONFIG
# =========================
BASE_PATH = "/kaggle/input/datasets/mahmudulhasantasin/isic-2017-original-dataset/isic 2017"

TRAIN_IMG_DIR  = os.path.join(BASE_PATH, "ISIC-2017_Training_Data")
TRAIN_MASK_DIR = os.path.join(BASE_PATH, "ISIC-2017_Training_Part1_GroundTruth")
VAL_IMG_DIR    = os.path.join(BASE_PATH, "ISIC-2017_Validation_Data")
VAL_MASK_DIR   = os.path.join(BASE_PATH, "ISIC-2017_Validation_Part1_GroundTruth")
TEST_IMG_DIR   = os.path.join(BASE_PATH, "ISIC-2017_Test_v2_Data")
TEST_MASK_DIR  = os.path.join(BASE_PATH, "ISIC-2017_Test_v2_Part1_GroundTruth")

OUTPUT_DIR = "/kaggle/working/outputs_resnet_attv2_256"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
EPOCHS      = 50
LR          = 1e-4
SEED        = 42
NUM_WORKERS = 2
PIN_MEMORY  = True

# ── ImageNet normalization constants ─────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def count_files(folder):
    return len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])

print("\n========= ISIC 2017 DATASET CHECK =========\n")
print("Training Images:", count_files(TRAIN_IMG_DIR))
print("Training Masks :", count_files(TRAIN_MASK_DIR))
print()
print("Validation Images:", count_files(VAL_IMG_DIR))
print("Validation Masks :", count_files(VAL_MASK_DIR))
print()
print("Test Images:", count_files(TEST_IMG_DIR))
print("Test Masks :", count_files(TEST_MASK_DIR))
print("\n============================================")

# =========================
# 1) SEED + DEVICE
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def get_device():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch:", torch.__version__)
    print("Device:", d)
    if d.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    return d

# =========================
# 2) FILE PAIRING
# =========================
def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def get_pairs(images_dir: str, masks_dir: str):
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Missing masks dir: {masks_dir}")

    img_paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob(os.path.join(images_dir, ext))
    )
    mask_paths = sorted(
        p for ext in ("*.png", "*.jpg", "*.jpeg")
        for p in glob(os.path.join(masks_dir, ext))
    )

    mask_map = {}
    for m in mask_paths:
        s = _stem(m)
        mask_map[s] = m
        if s.endswith("_segmentation"):
            mask_map[s.replace("_segmentation", "")] = m

    X, y, missing = [], [], 0
    for img in img_paths:
        key = _stem(img)
        if key in mask_map:
            X.append(img)
            y.append(mask_map[key])
        else:
            missing += 1
    return X, y, len(img_paths), len(mask_paths), missing

# =========================
# 3) ON-THE-FLY AUGMENTATION
# =========================
class AugPipeline:
    """
    Stochastic image + mask augmentation applied inside __getitem__.
    All operations work on float32 arrays in [0, 1] range.
    Note: augmentation is applied BEFORE ImageNet normalization.
    """
    def __init__(
        self,
        p_flip    = 0.5,
        p_rot90   = 0.5,
        p_bc      = 0.4,
        p_blur    = 0.3,
        p_erase   = 0.3,
        p_elastic = 0.3,
    ):
        self.p_flip    = p_flip
        self.p_rot90   = p_rot90
        self.p_bc      = p_bc
        self.p_blur    = p_blur
        self.p_erase   = p_erase
        self.p_elastic = p_elastic

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img  = img.copy()
        mask = mask.copy()

        # Flip
        if random.random() < self.p_flip:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if random.random() < self.p_flip:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()

        # Rotation (multiples of 90°)
        if random.random() < self.p_rot90:
            k    = random.choice([1, 2, 3])
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()

        # Brightness / Contrast
        if random.random() < self.p_bc:
            alpha = random.uniform(0.75, 1.25)
            beta  = random.uniform(-0.1, 0.1)
            img   = np.clip(alpha * img + beta, 0.0, 1.0)

        # Gaussian blur
        if random.random() < self.p_blur:
            ksize = random.choice([3, 5])
            img_u8 = (img * 255).astype(np.uint8)
            img_u8 = cv2.GaussianBlur(img_u8, (ksize, ksize), 0)
            img    = img_u8.astype(np.float32) / 255.0

        # Coarse dropout / random erasing
        if random.random() < self.p_erase:
            H, W = img.shape[:2]
            rh = random.randint(H // 16, H // 4)
            rw = random.randint(W // 16, W // 4)
            r0 = random.randint(0, H - rh)
            c0 = random.randint(0, W - rw)
            img[r0:r0+rh, c0:c0+rw] = random.random()

        # Grid distortion
        if random.random() < self.p_elastic:
            img, mask = self._grid_distort(img, mask)

        return img, mask

    @staticmethod
    def _grid_distort(img, mask, num_steps=5, distort_limit=0.15):
        H, W = img.shape[:2]
        xs = np.linspace(0, W, num_steps + 1)
        ys = np.linspace(0, H, num_steps + 1)
        jitter_x = np.random.uniform(
            -distort_limit * W / num_steps,
             distort_limit * W / num_steps,
            (num_steps + 1, num_steps + 1)).astype(np.float32)
        jitter_y = np.random.uniform(
            -distort_limit * H / num_steps,
             distort_limit * H / num_steps,
            (num_steps + 1, num_steps + 1)).astype(np.float32)

        map_x = np.zeros((H, W), dtype=np.float32)
        map_y = np.zeros((H, W), dtype=np.float32)
        for i in range(num_steps):
            for j in range(num_steps):
                x0, x1 = int(xs[j]), int(xs[j + 1])
                y0, y1 = int(ys[i]), int(ys[i + 1])
                if x0 >= x1 or y0 >= y1:
                    continue
                patch_y, patch_x = np.mgrid[y0:y1, x0:x1]
                bx = np.interp(patch_x, [x0, x1], [jitter_x[i, j], jitter_x[i, j + 1]])
                by = np.interp(patch_y, [y0, y1], [jitter_y[i, j], jitter_y[i + 1, j]])
                map_x[y0:y1, x0:x1] = patch_x.astype(np.float32) + bx.astype(np.float32)
                map_y[y0:y1, x0:x1] = patch_y.astype(np.float32) + by.astype(np.float32)

        img_u8   = (img  * 255).astype(np.uint8)
        mask_u8  = (mask * 255).astype(np.uint8)
        img_out  = cv2.remap(img_u8,  map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        mask_out = cv2.remap(mask_u8, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_REFLECT_101)
        return img_out.astype(np.float32) / 255.0, (mask_out > 127).astype(np.float32)

# =========================
# 4) DATASET
# =========================
_AUG = AugPipeline()

class ISICDataset(Dataset):
    def __init__(self, X, y, size=(256, 256), augment=False):
        self.X       = X
        self.y       = y
        self.size    = size
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.X[idx]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0          # [0, 1]

        mask = cv2.imread(self.y[idx], 0)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            img, mask = _AUG(img, mask)

        # ── ImageNet normalization ─────────────────────────────────────
        img = (img - IMAGENET_MEAN) / IMAGENET_STD    # per-channel z-score

        img  = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32)
        mask = torch.tensor(mask[None], dtype=torch.float32)
        return img, mask

# =========================
# 5) PREVIEW HELPERS
# =========================
def _denorm(img_tensor):
    """Reverse ImageNet normalization for visualization."""
    img = img_tensor.numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)

def save_batch_preview(loader, out_path, n=3):
    xb, yb = next(iter(loader))
    xb, yb = xb[:n], yb[:n].numpy()
    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img  = _denorm(xb[i])
        mask = yb[i, 0]
        plt.subplot(n, 3, 3*i+1); plt.imshow(img);               plt.title("Image");   plt.axis("off")
        plt.subplot(n, 3, 3*i+2); plt.imshow(mask, cmap="gray"); plt.title("Mask");    plt.axis("off")
        overlay = img.copy(); overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * mask, 0, 1)
        plt.subplot(n, 3, 3*i+3); plt.imshow(overlay);           plt.title("Overlay"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

def save_history_plot(history, out_path):
    plt.figure(figsize=(16, 5))
    for k, (tr, va, title) in enumerate([
        ("loss", "val_loss", "Loss"),
        ("dice", "val_dice", "Dice"),
        ("iou",  "val_iou",  "IoU"),
    ]):
        plt.subplot(1, 3, k+1)
        plt.plot(history[tr], label="train")
        plt.plot(history[va], label="val")
        plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

def save_prediction_preview(model, loader, device, out_path, n=4, thr=0.5):
    model.eval()
    xb, yb = next(iter(loader))
    xb = xb[:n].to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        preds = (probs > thr).astype(np.float32)
    xb_np, yb_np = xb.cpu(), yb[:n].numpy()
    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img = _denorm(xb_np[i])
        plt.subplot(n, 4, 4*i+1); plt.imshow(img);               plt.title("Image"); plt.axis("off")
        plt.subplot(n, 4, 4*i+2); plt.imshow(yb_np[i, 0], cmap="gray"); plt.title("GT");    plt.axis("off")
        plt.subplot(n, 4, 4*i+3); plt.imshow(probs[i, 0],  cmap="gray"); plt.title("Prob");  plt.axis("off")
        plt.subplot(n, 4, 4*i+4); plt.imshow(preds[i, 0],  cmap="gray"); plt.title("Bin");   plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

# =========================
# 6) METRICS + LOSS
# =========================
def dice_coeff(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter  = (y_true * y_pred).sum(dim=1)
    return ((2 * inter + eps) / (y_true.sum(dim=1) + y_pred.sum(dim=1) + eps)).mean()

def iou_score(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter  = (y_true * y_pred).sum(dim=1)
    union  = y_true.sum(dim=1) + y_pred.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, y_true):
        return self.bce(logits, y_true) + 1.0 - dice_coeff(y_true, torch.sigmoid(logits))

# =========================
# 7) BUILDING BLOCKS
# =========================

# ── Basic double conv ─────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ── ① ASPP  (rates adjusted to prevent overshoot at 256×256) ─────────────────
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling — bottleneck module.

    Dilation rates set to (2, 4, 8) instead of the original (6, 12, 18).
    At 256×256 input the bottleneck resolution is 8×8; rates 6/12/18 would
    produce receptive-field extents larger than the feature map, causing
    padding artifacts.  Rates (2, 4, 8) give meaningful multi-scale context
    while staying well within the 8×8 spatial extent.

    Dropout2d raised to 0.2 for stronger regularisation.
    """
    def __init__(self, in_ch: int, out_ch: int, rates=(2, 4, 8)):
        super().__init__()
        self.b0 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        n_branches = 1 + len(rates) + 1          # b0 + dilated × 3 + gap
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),                    # raised from 0.1 → 0.2
        )

    def forward(self, x):
        h, w  = x.shape[-2:]
        parts = [self.b0(x)]
        for br in self.branches:
            parts.append(br(x))
        gap = self.global_pool(x)
        gap = F.interpolate(gap, size=(h, w), mode="bilinear", align_corners=False)
        parts.append(gap)
        return self.project(torch.cat(parts, dim=1))


# ── ② CBAM  ───────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        mid = max(ch // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
        )

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx  = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx).view(x.size(0), x.size(1), 1, 1)


class SpatialAttentionCBAM(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.bn   = nn.BatchNorm2d(1)

    def forward(self, x):
        avg    = x.mean(dim=1, keepdim=True)
        mx, _  = x.max(dim=1, keepdim=True)
        att    = torch.sigmoid(self.bn(self.conv(torch.cat([avg, mx], dim=1))))
        return x * att


class CBAM(nn.Module):
    """Full CBAM: channel-then-spatial."""
    def __init__(self, ch: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(ch, reduction)
        self.sa = SpatialAttentionCBAM(spatial_kernel)

    def forward(self, x):
        return self.sa(self.ca(x))


# ── ③ Decoder block — CBAM applied to skip connection ────────────────────────
class UpBlock(nn.Module):
    """
    Decoder step.
    The skip-connection feature map is refined by CBAM (channel + spatial
    attention) before being concatenated with the upsampled decoder feature.
    This replaces the DualAttentionGate from v2 and the SpatialAttentionGate
    from v1 with a self-contained, gating-free attention module.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.cbam = CBAM(ch=skip_ch, reduction=16, spatial_kernel=7)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.up(x)
        if g.shape[-2:] != skip.shape[-2:]:
            g = F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attended_skip = self.cbam(skip)                # refine skip with CBAM
        return self.conv(torch.cat([g, attended_skip], dim=1))


# =========================
# 8) FULL MODEL
# =========================
class ResNetAttUNetV2(nn.Module):
    """
    ResNet-34 encoder
    + ASPP bottleneck  (rates=2/4/8, dropout=0.2)
    + CBAM post-bottleneck
    + CBAM on every skip connection (replaces DualAttentionGate)
    + ImageNet-normalized inputs expected
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # ── Encoder ───────────────────────────────────────────────────
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2   64ch
        self.pool = enc.maxpool                                   # /4   64ch
        self.e1   = enc.layer1    # /4   64ch
        self.e2   = enc.layer2    # /8  128ch
        self.e3   = enc.layer3    # /16 256ch
        self.e4   = enc.layer4    # /32 512ch

        # ── Bottleneck: ASPP → CBAM ───────────────────────────────────
        self.aspp = ASPP(in_ch=512, out_ch=512, rates=(2, 4, 8))
        self.cbam = CBAM(ch=512, reduction=16, spatial_kernel=7)

        # ── Decoder with CBAM skip-connection attention ───────────────
        self.d1 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.d2 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.d3 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)
        self.d4 = UpBlock(in_ch=64,  skip_ch=64,  out_ch=64)

        # ── Final head ────────────────────────────────────────────────
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final    = nn.Sequential(ConvBlock(32, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.stem(x)          # (B,  64, H/2,  W/2)
        p  = self.pool(s0)         # (B,  64, H/4,  W/4)
        s1 = self.e1(p)            # (B,  64, H/4,  W/4)
        s2 = self.e2(s1)           # (B, 128, H/8,  W/8)
        s3 = self.e3(s2)           # (B, 256, H/16, W/16)
        s4 = self.e4(s3)           # (B, 512, H/32, W/32)

        # Bottleneck
        b  = self.cbam(self.aspp(s4))   # (B, 512, H/32, W/32)

        # Decoder
        x  = self.d1(b,  s3)       # (B, 256, H/16, W/16)
        x  = self.d2(x, s2)        # (B, 128, H/8,  W/8)
        x  = self.d3(x, s1)        # (B,  64, H/4,  W/4)
        x  = self.d4(x, s0)        # (B,  64, H/2,  W/2)

        x  = self.final_up(x)      # (B,  32, H,    W)
        return self.final(x)       # (B,   1, H,    W)


# =========================
# 9) TRAIN / EVAL LOOPS
# =========================
def run_epoch(model, loader, device, criterion, optimizer=None, train=False):
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = 0.0

    for xb, yb in tqdm(loader, desc="train" if train else "val", leave=True):
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_dice += dice_coeff(yb, preds).item()
            total_iou  += iou_score(yb, preds).item()
        total_loss += loss.item()

    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="test", leave=True):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()
            total_loss += loss.item()
            total_dice += dice_coeff(yb, preds).item()
            total_iou  += iou_score(yb, preds).item()
    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n


# =========================
# 10) MAIN
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    train_X, train_y, n_ti, n_tm, miss_t = get_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_X,   val_y,   n_vi, n_vm, miss_v = get_pairs(VAL_IMG_DIR,   VAL_MASK_DIR)
    test_X,  test_y,  n_xi, n_xm, miss_x = get_pairs(TEST_IMG_DIR,  TEST_MASK_DIR)

    print(f"Train: images={n_ti} masks={n_tm} paired={len(train_X)} missing={miss_t}")
    print(f"Val:   images={n_vi} masks={n_vm} paired={len(val_X)}   missing={miss_v}")
    print(f"Test:  images={n_xi} masks={n_xm} paired={len(test_X)}  missing={miss_x}")

    if not (train_X and val_X and test_X):
        raise RuntimeError("One split has 0 paired samples. Check filenames.")

    train_ds = ISICDataset(train_X, train_y, size=IMG_SIZE, augment=True)
    val_ds   = ISICDataset(val_X,   val_y,   size=IMG_SIZE, augment=False)
    test_ds  = ISICDataset(test_X,  test_y,  size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,  BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    save_batch_preview(train_loader, os.path.join(OUTPUT_DIR, "train_batch_preview.png"))

    # ── Model ─────────────────────────────────────────────────────────
    model = ResNetAttUNetV2(pretrained=True).to(device)
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    best_val_dice  = -1.0
    early_patience = 12
    patience_count = 0
    history = {k: [] for k in ("loss", "val_loss", "dice", "val_dice", "iou", "val_iou")}
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        tr_loss, tr_dice, tr_iou = run_epoch(
            model, train_loader, device, criterion, optimizer, train=True)
        va_loss, va_dice, va_iou = run_epoch(
            model, val_loader,   device, criterion, train=False)

        for k, v in zip(
            ("loss", "dice", "iou", "val_loss", "val_dice", "val_iou"),
            (tr_loss, tr_dice, tr_iou, va_loss, va_dice, va_iou)
        ):
            history[k].append(v)

        print(f"train  loss={tr_loss:.4f}  dice={tr_dice:.4f}  iou={tr_iou:.4f}")
        print(f"val    loss={va_loss:.4f}  dice={va_dice:.4f}  iou={va_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_dice)
        if optimizer.param_groups[0]["lr"] != prev_lr:
            print(f"  → LR reduced to {optimizer.param_groups[0]['lr']:.2e}")

        if va_dice > best_val_dice:
            best_val_dice  = va_dice
            patience_count = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_dice": va_dice}, ckpt_path)
            print(f"  ✓ Saved best checkpoint  (val_dice={va_dice:.4f})")
        else:
            patience_count += 1
            if patience_count >= early_patience:
                print("Early stopping triggered.")
                break

    save_history_plot(history, os.path.join(OUTPUT_DIR, "training_curves.png"))

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nLoaded best checkpoint: epoch {ckpt['epoch']}  val_dice={ckpt['val_dice']:.4f}")

    te_loss, te_dice, te_iou = evaluate(model, test_loader, device, criterion)
    print(f"\nTEST  loss={te_loss:.4f}  dice={te_dice:.4f}  iou={te_iou:.4f}")

    save_prediction_preview(model, test_loader, device,
                            os.path.join(OUTPUT_DIR, "test_pred_preview.png"), n=4)

    final_path = os.path.join(OUTPUT_DIR, "final_model_state_dict.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nDONE. Saved final model → {final_path}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
