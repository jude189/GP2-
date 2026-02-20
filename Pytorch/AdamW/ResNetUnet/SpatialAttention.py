# CHANGES vs original:
#   - Spatial Attention Gates added to ALL four skip connections (d1..d4)
#   - AttentionGate module: uses gating signal (decoder) + skip (encoder)
#     to produce a spatial soft-attention map, then applies it to the skip
#   - Everything else (training loop, metrics, I/O) is unchanged

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

IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
SEED = 42

# Windows stable
NUM_WORKERS = 0
PIN_MEMORY = False

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_resnet_attunet_256")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "masks")
VAL_IMG_DIR   = os.path.join(BASE_DIR, "val",   "images")
VAL_MASK_DIR  = os.path.join(BASE_DIR, "val",   "masks")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "test",  "images")
TEST_MASK_DIR = os.path.join(BASE_DIR, "test",  "masks")

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
# 2) FILE PAIRING (robust)
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

    # Build mask index by possible keys:
    # - exact stem
    # - stem with common suffix removed (e.g., _segmentation)
    mask_map = {}
    for m in mask_paths:
        s = _stem(m)
        mask_map[s] = m
        if s.endswith("_segmentation"):
            mask_map[s.replace("_segmentation", "")] = m

    X, y = [], []
    missing = 0
    for img in img_paths:
        key = _stem(img)
        if key in mask_map:
            X.append(img)
            y.append(mask_map[key])
        else:
            missing += 1

    return X, y, len(img_paths), len(mask_paths), missing

# =========================
# 3) DATASET
# =========================
class ISICDataset(Dataset):
    def __init__(self, X, y, size=(256, 256), augment=False):
        self.X = X
        self.y = y
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, img, mask):
        if np.random.rand() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        if np.random.rand() < 0.5:
            img = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
        return img, mask

    def __getitem__(self, idx):
        img_path = self.X[idx]
        mask_path = self.y[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = np.transpose(img, (2, 0, 1))  # CHW
        mask = mask[None, ...]              # 1xHxW

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# =========================
# 4) SAVE PREVIEWS (never blocks)
# =========================
def save_batch_preview(loader, out_path, n=3):
    xb, yb = next(iter(loader))
    xb = xb[:n].numpy()
    yb = yb[:n].numpy()

    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img = np.transpose(xb[i], (1, 2, 0))
        mask = yb[i, 0]

        plt.subplot(n, 3, 3 * i + 1)
        plt.imshow(img); plt.title("Image"); plt.axis("off")

        plt.subplot(n, 3, 3 * i + 2)
        plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")

        overlay = img.copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * mask, 0, 1)
        plt.subplot(n, 3, 3 * i + 3)
        plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_history_plot(history, out_path):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history["dice"], label="train")
    plt.plot(history["val_dice"], label="val")
    plt.title("Dice")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history["iou"], label="train")
    plt.plot(history["val_iou"], label="val")
    plt.title("IoU")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_prediction_preview(model, loader, device, out_path, n=4, thr=0.5):
    model.eval()
    xb, yb = next(iter(loader))
    xb = xb[:n].to(device)
    yb_np = yb[:n].numpy()

    with torch.no_grad():
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > thr).astype(np.float32)

    xb_np = xb.cpu().numpy()

    plt.figure(figsize=(12, 4 * n))
    for i in range(n):
        img = np.transpose(xb_np[i], (1, 2, 0))
        gt = yb_np[i, 0]
        pr = probs[i, 0]
        pb = preds[i, 0]

        plt.subplot(n, 4, 4 * i + 1); plt.imshow(img); plt.title("Image"); plt.axis("off")
        plt.subplot(n, 4, 4 * i + 2); plt.imshow(gt, cmap="gray"); plt.title("GT Mask"); plt.axis("off")
        plt.subplot(n, 4, 4 * i + 3); plt.imshow(pr, cmap="gray"); plt.title("Pred Prob"); plt.axis("off")
        plt.subplot(n, 4, 4 * i + 4); plt.imshow(pb, cmap="gray"); plt.title("Pred Bin"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# =========================
# 5) METRICS + LOSS
# =========================
def dice_coeff(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter = (y_true * y_pred).sum(dim=1)
    denom = y_true.sum(dim=1) + y_pred.sum(dim=1)
    return ((2 * inter + eps) / (denom + eps)).mean()

def iou_score(y_true, y_pred, eps=1e-6):
    y_true = y_true.contiguous().view(y_true.size(0), -1)
    y_pred = y_pred.contiguous().view(y_pred.size(0), -1)
    inter = (y_true * y_pred).sum(dim=1)
    union = y_true.sum(dim=1) + y_pred.sum(dim=1) - inter
    return ((inter + eps) / (union + eps)).mean()

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, y_true):
        bce = self.bce(logits, y_true)
        probs = torch.sigmoid(logits)
        dice = 1.0 - dice_coeff(y_true, probs)
        return bce + dice

# =========================
# 6) MODEL
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------
# Spatial Attention Gate (Oktay et al., Attention U-Net)
# -------------------------------------------------------
# Given:
#   g  : gating signal  (from decoder, coarser or same resolution) — channels = g_ch
#   x  : skip feature   (from encoder)                             — channels = x_ch
#
# Steps:
#   1. 1×1 conv both g and x down to inter_ch features
#   2. Add them (align spatial dims via interpolation if needed)
#   3. ReLU → 1×1 conv → sigmoid  →  attention map α  (spatial, 1 channel)
#   4. Return x * α  (attended skip)
# -------------------------------------------------------
class SpatialAttentionGate(nn.Module):
    """
    Spatial soft-attention gate for a single skip connection.

    Parameters
    ----------
    g_ch    : number of channels in the gating signal (decoder feature)
    x_ch    : number of channels in the skip connection (encoder feature)
    inter_ch: number of intermediate channels (typically min(g_ch, x_ch) // 2)
    """
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        # Project gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # Project skip features
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # Produce scalar attention coefficient per spatial location
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g : (B, g_ch, H_g, W_g)  — gating signal (may be smaller than x)
        x : (B, x_ch, H_x, W_x)  — skip connection feature map

        Returns attended skip: (B, x_ch, H_x, W_x)
        """
        # Project both to inter_ch
        g1 = self.W_g(g)  # (B, inter_ch, H_g, W_g)
        x1 = self.W_x(x)  # (B, inter_ch, H_x, W_x)

        # Upsample g1 to match x1's spatial size (if different)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)

        # Element-wise add → ReLU → 1×1 → sigmoid
        alpha = self.psi(self.relu(g1 + x1))  # (B, 1, H_x, W_x)

        # Apply spatial attention to skip
        return x * alpha                       # broadcast over channels

class UpBlock(nn.Module):
    """
    Decoder block with integrated Spatial Attention Gate on the skip connection.

    Parameters
    ----------
    in_ch   : channels coming from the previous decoder stage (or bottleneck)
    skip_ch : channels of the encoder skip connection
    out_ch  : output channels after the double ConvBlock
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Transposed conv to upsample the gating signal / decoder feature
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # Attention gate: gate = upsampled decoder (out_ch channels),
        #                 skip = encoder feature   (skip_ch channels)
        inter_ch = max(out_ch // 2, 1)
        self.att = SpatialAttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=inter_ch)

        # Standard double conv after concatenation
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # 1. Upsample decoder feature → becomes the gating signal
        g = self.up(x)  # (B, out_ch, H, W)

        # Align spatial size to skip (just in case of odd dimensions)
        if g.shape[-2:] != skip.shape[-2:]:
            g = F.interpolate(g, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        # 2. Apply spatial attention gate to the skip connection
        attended_skip = self.att(g, skip)  # (B, skip_ch, H, W)

        # 3. Concatenate gating signal with attended skip, then conv
        x = torch.cat([g, attended_skip], dim=1)  # (B, out_ch + skip_ch, H, W)
        return self.conv(x)

class ResNetAttUNet(nn.Module):
    """
    ResNet-34 encoder + U-Net decoder with Spatial Attention Gates
    on all four skip connections (s3, s2, s1, s0).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        enc = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        # ---- Encoder (frozen backbone optional) ----
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)  # /2  → 64ch
        self.pool = enc.maxpool                                  # /4  → 64ch

        self.e1 = enc.layer1   # /4  →  64ch
        self.e2 = enc.layer2   # /8  → 128ch
        self.e3 = enc.layer3   # /16 → 256ch
        self.e4 = enc.layer4   # /32 → 512ch

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(512, 512)

        # ---- Decoder with Attention Gates ----
        # d1: gate from bottleneck (512→256), skip=s3 (256ch)
        self.d1 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        # d2: gate from d1 (256→128), skip=s2 (128ch)
        self.d2 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        # d3: gate from d2 (128→64), skip=s1 (64ch)
        self.d3 = UpBlock(in_ch=128, skip_ch=64,  out_ch=64)
        # d4: gate from d3 (64→64), skip=s0 (64ch)  [stem output]
        self.d4 = UpBlock(in_ch=64,  skip_ch=64,  out_ch=64)

        # ---- Final upsample + head ----
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final = nn.Sequential(
            ConvBlock(32, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        s0 = self.stem(x)      # (B,  64, H/2,  W/2)
        x  = self.pool(s0)     # (B,  64, H/4,  W/4)
        s1 = self.e1(x)        # (B,  64, H/4,  W/4)
        s2 = self.e2(s1)       # (B, 128, H/8,  W/8)
        s3 = self.e3(s2)       # (B, 256, H/16, W/16)
        s4 = self.e4(s3)       # (B, 512, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(s4)  # (B, 512, H/32, W/32)

        # Decoder — each UpBlock applies attention gate internally
        x = self.d1(b, s3)       # (B, 256, H/16, W/16)
        x = self.d2(x, s2)       # (B, 128, H/8,  W/8)
        x = self.d3(x, s1)       # (B,  64, H/4,  W/4)
        x = self.d4(x, s0)       # (B,  64, H/2,  W/2)

        # Final upsample to input resolution
        x = self.final_up(x)     # (B,  32, H,    W)
        x = self.final(x)        # (B,   1, H,    W)
        return x

# =========================
# 7) TRAIN / EVAL
# =========================
def run_epoch(model, loader, device, criterion, optimizer=None, train=False):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    for xb, yb in tqdm(loader, desc=("train" if train else "val"), leave=True):
        xb = xb.to(device)
        yb = yb.to(device)

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
            d = dice_coeff(yb, preds).item()
            j = iou_score(yb, preds).item()

        total_loss += loss.item()
        total_dice += d
        total_iou  += j

    n = max(1, len(loader))
    return total_loss / n, total_dice / n, total_iou / n

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="test", leave=True):
            xb = xb.to(device)
            yb = yb.to(device)
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
# 8) MAIN
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    train_X, train_y, n_train_imgs, n_train_masks, miss_train = get_pairs(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_X,   val_y,   n_val_imgs,   n_val_masks,   miss_val   = get_pairs(VAL_IMG_DIR,   VAL_MASK_DIR)
    test_X,  test_y,  n_test_imgs,  n_test_masks,  miss_test  = get_pairs(TEST_IMG_DIR,  TEST_MASK_DIR)

    print(f"Train: images={n_train_imgs} masks={n_train_masks} paired={len(train_X)} missing_pairs={miss_train}")
    print(f"Val:   images={n_val_imgs}   masks={n_val_masks}   paired={len(val_X)}   missing_pairs={miss_val}")
    print(f"Test:  images={n_test_imgs}  masks={n_test_masks}  paired={len(test_X)}  missing_pairs={miss_test}")

    if len(train_X) == 0 or len(val_X) == 0 or len(test_X) == 0:
        raise RuntimeError("One of the splits has 0 paired samples. Check filenames in images/masks.")

    train_ds = ISICDataset(train_X, train_y, size=IMG_SIZE, augment=True)
    val_ds   = ISICDataset(val_X,   val_y,   size=IMG_SIZE, augment=False)
    test_ds  = ISICDataset(test_X,  test_y,  size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    preview_path = os.path.join(OUTPUT_DIR, "train_batch_preview.png")
    print("Saving train batch preview to:", preview_path)
    save_batch_preview(train_loader, preview_path, n=3)

    # ----- Attention U-Net -----
    model = ResNetAttUNet(pretrained=True).to(device)
    print("Model:", model.__class__.__name__)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    best_val_dice  = -1.0
    early_patience = 12
    patience_count = 0
    history = {"loss": [], "val_loss": [], "dice": [], "val_dice": [], "iou": [], "val_iou": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        tr_loss, tr_dice, tr_iou = run_epoch(model, train_loader, device, criterion, optimizer, train=True)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader,   device, criterion, optimizer=None, train=False)

        history["loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["dice"].append(tr_dice)
        history["val_dice"].append(va_dice)
        history["iou"].append(tr_iou)
        history["val_iou"].append(va_iou)

        print(f"train loss={tr_loss:.4f} dice={tr_dice:.4f} iou={tr_iou:.4f}")
        print(f"val   loss={va_loss:.4f} dice={va_dice:.4f} iou={va_iou:.4f}")
        print("Current LR:", optimizer.param_groups[0]["lr"])

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_dice)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != prev_lr:
            print(f"LR reduced: {prev_lr} -> {new_lr}")

        ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        if va_dice > best_val_dice:
            best_val_dice  = va_dice
            patience_count = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_dice": va_dice}, ckpt_path)
            print("Saved best:", ckpt_path)
        else:
            patience_count += 1
            if patience_count >= early_patience:
                print("Early stopping.")
                break

    curves_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    print("Saving training curves to:", curves_path)
    save_history_plot(history, curves_path)

    ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint: epoch {ckpt['epoch']} val_dice={ckpt['val_dice']:.4f}")

    te_loss, te_dice, te_iou = evaluate(model, test_loader, device, criterion)
    print(f"TEST loss={te_loss:.4f} dice={te_dice:.4f} iou={te_iou:.4f}")

    pred_path = os.path.join(OUTPUT_DIR, "test_pred_preview.png")
    print("Saving test prediction preview to:", pred_path)
    save_prediction_preview(model, test_loader, device, pred_path, n=4)

    final_path = os.path.join(OUTPUT_DIR, "final_model_state_dict.pt")
    torch.save(model.state_dict(), final_path)
    print("DONE. Saved:", final_path)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
