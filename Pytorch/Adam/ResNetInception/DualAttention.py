# Architecture:
#   ENCODER  : ResNet34 (pretrained) — skip connections at /2,/4,/8,/16,/32
#   DECODER  : Inception-style ConvBlocks (1x1 + 3x3 + 3x3→3x3 branches)
#   ATTENTION: Dual Attention Gate on every skip connection before concat
#              = Channel Attention (SE-style) ⊗ Spatial Attention (additive gate)
#


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# =========================
# 0) CONFIG
# =========================
BASE_DIR = r"C:\Users\20210061\gdrive\GP1\ISIC2017"

IMG_SIZE    = (256, 256)
BATCH_SIZE  = 8
EPOCHS      = 50
LR          = 1e-4
SEED        = 42
NUM_WORKERS = 0
PIN_MEMORY  = False

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_resnet_inc_dualattn_256")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_IMG_DIR  = os.path.join(BASE_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "masks")
VAL_IMG_DIR    = os.path.join(BASE_DIR, "val",   "images")
VAL_MASK_DIR   = os.path.join(BASE_DIR, "val",   "masks")
TEST_IMG_DIR   = os.path.join(BASE_DIR, "test",  "images")
TEST_MASK_DIR  = os.path.join(BASE_DIR, "test",  "masks")

# =========================
# 1) SEED + DEVICE
# =========================
def set_seed(seed: int):
    import random
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

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_paths.extend(glob(os.path.join(images_dir, ext)))
    img_paths = sorted(img_paths)

    mask_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        mask_paths.extend(glob(os.path.join(masks_dir, ext)))
    mask_paths = sorted(mask_paths)

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
            X.append(img); y.append(mask_map[key])
        else:
            missing += 1

    return X, y, len(img_paths), len(mask_paths), missing

# =========================
# 3) DATASET
# =========================
class ISICDataset(Dataset):
    def __init__(self, X, y, size=(256, 256), augment=False):
        self.X = X; self.y = y; self.size = size; self.augment = augment

    def __len__(self): return len(self.X)

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy(); mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            img = np.flipud(img).copy(); mask = np.flipud(mask).copy()
        return img, mask

    def __getitem__(self, idx):
        img = cv2.imread(self.X[idx])
        if img is None: raise RuntimeError(f"Failed to read image: {self.X[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.y[idx], 0)
        if mask is None: raise RuntimeError(f"Failed to read mask: {self.y[idx]}")
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.augment: img, mask = self._augment(img, mask)

        img  = np.transpose(img, (2, 0, 1))
        mask = mask[None, ...]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# =========================
# 4) PLOT HELPERS
# =========================
def save_batch_preview(loader, out_path, n=3):
    xb, yb = next(iter(loader))
    xb = xb[:n].numpy(); yb = yb[:n].numpy()
    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img = np.transpose(xb[i], (1, 2, 0)); mask = yb[i, 0]
        plt.subplot(n, 3, 3*i+1); plt.imshow(img);              plt.title("Image");   plt.axis("off")
        plt.subplot(n, 3, 3*i+2); plt.imshow(mask, cmap="gray"); plt.title("Mask");    plt.axis("off")
        overlay = img.copy(); overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * mask, 0, 1)
        plt.subplot(n, 3, 3*i+3); plt.imshow(overlay);           plt.title("Overlay"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

def save_history_plot(history, out_path):
    plt.figure(figsize=(16, 5))
    plt.subplot(1,3,1); plt.plot(history["loss"],label="train"); plt.plot(history["val_loss"],label="val"); plt.title("Loss"); plt.legend(); plt.grid(True)
    plt.subplot(1,3,2); plt.plot(history["dice"],label="train"); plt.plot(history["val_dice"],label="val"); plt.title("Dice"); plt.legend(); plt.grid(True)
    plt.subplot(1,3,3); plt.plot(history["iou"], label="train"); plt.plot(history["val_iou"], label="val"); plt.title("IoU");  plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

def save_prediction_preview(model, loader, device, out_path, n=4, thr=0.5):
    model.eval()
    xb, yb = next(iter(loader)); xb = xb[:n].to(device); yb_np = yb[:n].numpy()
    with torch.no_grad():
        logits = model(xb); probs = torch.sigmoid(logits).cpu().numpy(); preds = (probs > thr).astype(np.float32)
    xb_np = xb.cpu().numpy()
    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img = np.transpose(xb_np[i], (1, 2, 0)); gt = yb_np[i,0]; pr = probs[i,0]; pb = preds[i,0]
        plt.subplot(n,4,4*i+1); plt.imshow(img);             plt.title("Image");    plt.axis("off")
        plt.subplot(n,4,4*i+2); plt.imshow(gt, cmap="gray"); plt.title("GT Mask");  plt.axis("off")
        plt.subplot(n,4,4*i+3); plt.imshow(pr, cmap="gray"); plt.title("Pred Prob");plt.axis("off")
        plt.subplot(n,4,4*i+4); plt.imshow(pb, cmap="gray"); plt.title("Pred Bin"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()

# =========================
# 5) METRICS + LOSS
# =========================
def dice_coeff(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter  = (y_true * y_pred).sum(dim=1)
    denom  = y_true.sum(dim=1) + y_pred.sum(dim=1)
    return ((2 * inter + eps) / (denom + eps)).mean()

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
        bce   = self.bce(logits, y_true)
        probs = torch.sigmoid(logits)
        return bce + (1.0 - dice_coeff(y_true, probs))

# =========================
# 6) MODEL BUILDING BLOCKS
# =========================

# ------------------------------------------------------------------
# 6a) Inception ConvBlock  (DECODER + bottleneck + head)
#
#  3 parallel branches → concat → 1×1 projection → out_ch
#    b1 : 1×1                         point-wise
#    b2 : 1×1 → 3×3                   local context
#    b3 : 1×1 → 3×3 → 3×3            ≈5×5 receptive field (cheaper)
# ------------------------------------------------------------------
class InceptionConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        b = max(out_ch // 3, 8)

        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        self.fuse = nn.Sequential(
            nn.Conv2d(b * 3, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1))

# ------------------------------------------------------------------
# 6b) Channel Attention  (Squeeze-and-Excitation style)
#
#  Applied to the SKIP feature map:
#    1. Global average pool  → [B, C, 1, 1]
#    2. Global max pool      → [B, C, 1, 1]   (extra signal vs pure SE)
#    3. Shared MLP (C → C//r → C)
#    4. Add avg + max branches → sigmoid → channel scale vector
#    5. Multiply back onto skip feature
#
#  Using both avg and max pools (CBAM-style) gives richer
#  channel descriptors than average alone.
# ------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out)   # [B, C, 1, 1]
        return x * scale

# ------------------------------------------------------------------
# 6c) Spatial Attention  (additive gating, Oktay et al. 2018)
#
#  Uses the DECODER feature as a gating signal to suppress
#  irrelevant spatial positions in the skip:
#    W_x(skip) + W_g(gate_upsampled) → ReLU → ψ → sigmoid → α
#    out = α * skip
# ------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, x_ch: int, g_ch: int, inter_ch: int = None):
        super().__init__()
        inter_ch = inter_ch or max(x_ch // 2, 8)
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g_up = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        att  = self.relu(self.W_x(x) + self.W_g(g_up))
        att  = self.psi(att)          # [B, 1, H, W]
        return x * att

# ------------------------------------------------------------------
# 6d) Dual Attention Gate  ← REPLACES single spatial attention
#
#  Sequential composition applied to every skip connection:
#    Step 1 — Channel Attention  : WHAT features matter
#               (recalibrates channel weights globally)
#    Step 2 — Spatial Attention  : WHERE features matter
#               (gated by decoder signal, suppresses background)
#
#  The two stages are complementary:
#    channel att  → reweights "which lesion-relevant channels to amplify"
#    spatial att  → reweights "which pixels are lesion vs background"
# ------------------------------------------------------------------
class DualAttentionGate(nn.Module):
    def __init__(self, x_ch: int, g_ch: int, reduction: int = 8):
        super().__init__()
        self.channel_att = ChannelAttention(x_ch, reduction=reduction)
        self.spatial_att = SpatialAttention(x_ch, g_ch)

    def forward(self, x, g):
        """
        x : skip feature  [B, x_ch, H,  W ]  (from encoder)
        g : gate signal   [B, g_ch, H', W']  (from decoder, coarser)
        """
        x = self.channel_att(x)     # step 1 — WHAT  (no g needed)
        x = self.spatial_att(x, g)  # step 2 — WHERE (guided by g)
        return x

# ------------------------------------------------------------------
# 6e) Decoder UpBlock
#
#  1. Transpose conv  → upsample decoder tensor
#  2. Dual Attention Gate on the skip connection
#  3. Concat(upsampled decoder, attended skip)
#  4. Inception ConvBlock to refine
# ------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att  = DualAttentionGate(x_ch=skip_ch, g_ch=out_ch)
        self.conv = InceptionConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x    = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(skip, x)            # dual attention on skip
        x    = torch.cat([x, skip], dim=1)
        return self.conv(x)

# =========================
# 7) FULL MODEL
# =========================
class ResNetInceptionDualAttnUNet(nn.Module):
    """
    Encoder  : ResNet34 (pretrained ImageNet)
    Decoder  : Inception ConvBlocks (multi-scale)
    Attention: Dual Attention Gate (Channel + Spatial) on all skips
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        # ---- ENCODER ----
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # 64ch,  /2
        self.pool = enc.maxpool                                   # 64ch,  /4
        self.e1   = enc.layer1                                    # 64ch,  /4
        self.e2   = enc.layer2                                    # 128ch, /8
        self.e3   = enc.layer3                                    # 256ch, /16
        self.e4   = enc.layer4                                    # 512ch, /32

        # ---- BOTTLENECK (Inception) ----
        self.bottleneck = InceptionConvBlock(512, 512)

        # ---- DECODER (Inception + Dual Attention) ----
        self.d1 = UpBlock(512, 256, 256)   # gate s3
        self.d2 = UpBlock(256, 128, 128)   # gate s2
        self.d3 = UpBlock(128,  64,  64)   # gate s1
        self.d4 = UpBlock( 64,  64,  64)   # gate s0 (stem)

        # ---- HEAD ----
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Sequential(
            InceptionConvBlock(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x):
        s0 = self.stem(x)       # /2,  64ch
        p  = self.pool(s0)      # /4,  64ch
        s1 = self.e1(p)         # /4,  64ch
        s2 = self.e2(s1)        # /8,  128ch
        s3 = self.e3(s2)        # /16, 256ch
        s4 = self.e4(s3)        # /32, 512ch

        b  = self.bottleneck(s4)

        x  = self.d1(b,  s3)
        x  = self.d2(x,  s2)
        x  = self.d3(x,  s1)
        x  = self.d4(x,  s0)

        x  = self.final_up(x)
        x  = self.head(x)
        return x

# =========================
# 8) TRAIN / EVAL LOOPS
# =========================
def run_epoch(model, loader, device, criterion, optimizer=None, train=False):
    model.train() if train else model.eval()
    total_loss = total_dice = total_iou = 0.0

    for xb, yb in tqdm(loader, desc=("train" if train else "val"), leave=True):
        xb = xb.to(device); yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss   = criterion(logits, yb)
            if train:
                loss.backward(); optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits); preds = (probs > 0.5).float()
            total_dice += dice_coeff(yb, preds).item()
            total_iou  += iou_score(yb,  preds).item()
        total_loss += loss.item()

    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="test", leave=True):
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb); loss = criterion(logits, yb)
            probs  = torch.sigmoid(logits); preds = (probs > 0.5).float()
            total_loss += loss.item()
            total_dice += dice_coeff(yb, preds).item()
            total_iou  += iou_score(yb,  preds).item()
    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n

# =========================
# 9) MAIN
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    train_X, train_y, n_ti, n_tm, miss_t = get_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_X,   val_y,   n_vi, n_vm, miss_v = get_pairs(VAL_IMG_DIR,   VAL_MASK_DIR)
    test_X,  test_y,  n_ei, n_em, miss_e = get_pairs(TEST_IMG_DIR,  TEST_MASK_DIR)

    print(f"Train: images={n_ti} masks={n_tm} paired={len(train_X)} missing={miss_t}")
    print(f"Val:   images={n_vi} masks={n_vm} paired={len(val_X)}   missing={miss_v}")
    print(f"Test:  images={n_ei} masks={n_em} paired={len(test_X)}  missing={miss_e}")

    if not (train_X and val_X and test_X):
        raise RuntimeError("One split has 0 paired samples — check filenames.")

    train_ds = ISICDataset(train_X, train_y, size=IMG_SIZE, augment=True)
    val_ds   = ISICDataset(val_X,   val_y,   size=IMG_SIZE, augment=False)
    test_ds  = ISICDataset(test_X,  test_y,  size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    preview_path = os.path.join(OUTPUT_DIR, "train_batch_preview.png")
    print("Saving train preview →", preview_path)
    save_batch_preview(train_loader, preview_path, n=3)

    model = ResNetInceptionDualAttnUNet(pretrained=True).to(device)
    print("Model:", model.__class__.__name__)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6)

    best_val_dice = -1.0
    early_patience = 12
    patience_count = 0
    history = {"loss": [], "val_loss": [], "dice": [], "val_dice": [], "iou": [], "val_iou": []}
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        tr_loss, tr_dice, tr_iou = run_epoch(model, train_loader, device, criterion, optimizer, train=True)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader,   device, criterion, train=False)

        history["loss"].append(tr_loss);  history["val_loss"].append(va_loss)
        history["dice"].append(tr_dice);  history["val_dice"].append(va_dice)
        history["iou"].append(tr_iou);    history["val_iou"].append(va_iou)

        print(f"  train  loss={tr_loss:.4f}  dice={tr_dice:.4f}  iou={tr_iou:.4f}")
        print(f"  val    loss={va_loss:.4f}  dice={va_dice:.4f}  iou={va_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_dice)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"  LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

        if va_dice > best_val_dice:
            best_val_dice = va_dice
            patience_count = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_dice": va_dice}, ckpt_path)
            print(f"  ✓ Saved best checkpoint (epoch {epoch})")
        else:
            patience_count += 1
            if patience_count >= early_patience:
                print("Early stopping.")
                break

    save_history_plot(history, os.path.join(OUTPUT_DIR, "training_curves.png"))
    print("Training curves saved.")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint: epoch {ckpt['epoch']}  val_dice={ckpt['val_dice']:.4f}")

    te_loss, te_dice, te_iou = evaluate(model, test_loader, device, criterion)
    print(f"\nTEST  loss={te_loss:.4f}  dice={te_dice:.4f}  iou={te_iou:.4f}")

    pred_path = os.path.join(OUTPUT_DIR, "test_pred_preview.png")
    save_prediction_preview(model, test_loader, device, pred_path, n=4)
    print("Test prediction preview saved →", pred_path)

    final_path = os.path.join(OUTPUT_DIR, "final_model_state_dict.pt")
    torch.save(model.state_dict(), final_path)
    print("DONE. Final weights saved →", final_path)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()



