# =============================================================================
# ISIC 2017 - FULL DATA (uses train/val/test folders as-is)
# Model: InceptionV3 Encoder + U-Net Decoder
#        WITH SKIP CONNECTIONS + DUAL ATTENTION (Channel + Spatial)
# Optimizer: AdamW
# Image size: 256x256
# TF 2.19 SAFE
# =============================================================================

# =============================================================================
# STEP 1: Install + Imports
# =============================================================================
!pip install -q tensorflow opencv-python scikit-learn matplotlib

import os, cv2, numpy as np
import matplotlib.pyplot as plt
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from google.colab import drive

print("TF:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.backend.clear_session()

# =============================================================================
# STEP 2: Mount Drive + Paths
# =============================================================================
drive.mount("/content/drive")

DRIVE_BASE = "/content/drive/MyDrive/GP1/ISIC2017"

TRAIN_IMG_DIR  = os.path.join(DRIVE_BASE, "train/images")
TRAIN_MASK_DIR = os.path.join(DRIVE_BASE, "train/masks")

VAL_IMG_DIR    = os.path.join(DRIVE_BASE, "val/images")
VAL_MASK_DIR   = os.path.join(DRIVE_BASE, "val/masks")

TEST_IMG_DIR   = os.path.join(DRIVE_BASE, "test/images")
TEST_MASK_DIR  = os.path.join(DRIVE_BASE, "test/masks")

OUTPUT_DIR     = os.path.join(DRIVE_BASE, "outputs_inception_unet_adamw_dualattn_256_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 3: Dataset Manager
# =============================================================================
class DatasetManager:
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def get_file_paths(self):
        imgs = sorted(glob(os.path.join(self.img_dir, "*.jpg")))
        masks = sorted(glob(os.path.join(self.mask_dir, "*.png")))

        mask_map = {}
        for m in masks:
            key = os.path.basename(m).replace("_segmentation.png", "")
            mask_map[key] = m

        X, y = [], []
        for img in imgs:
            key = os.path.splitext(os.path.basename(img))[0]
            if key in mask_map:
                X.append(img)
                y.append(mask_map[key])

        return X, y

train_dm = DatasetManager(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
val_dm   = DatasetManager(VAL_IMG_DIR,   VAL_MASK_DIR)
test_dm  = DatasetManager(TEST_IMG_DIR,  TEST_MASK_DIR)

train_X, train_y = train_dm.get_file_paths()
val_X,   val_y   = val_dm.get_file_paths()
test_X,  test_y  = test_dm.get_file_paths()

print("Counts:", len(train_X), len(val_X), len(test_X))

assert len(train_X) > 0, "No TRAIN images found. Check TRAIN paths."
assert len(val_X) > 0,   "No VAL images found. Check VAL paths."
assert len(test_X) > 0,  "No TEST images found. Check TEST paths."

# =============================================================================
# STEP 4: Data Generator
# =============================================================================
class ISICGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch=8, size=(256,256), shuffle=True):
        self.X, self.y = X, y
        self.batch, self.size = batch, size
        self.shuffle = shuffle
        self.idx = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __getitem__(self, i):
        ids = self.idx[i*self.batch:(i+1)*self.batch]
        Xb, yb = [], []

        for j in ids:
            img = cv2.imread(self.X[j])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size).astype(np.float32) / 255.0

            mask = cv2.imread(self.y[j], 0)
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)

            Xb.append(img)
            yb.append(mask[..., None])

        return np.array(Xb), np.array(yb)

train_gen = ISICGenerator(train_X, train_y, batch=8, size=(256,256), shuffle=True)
val_gen   = ISICGenerator(val_X,   val_y,   batch=8, size=(256,256), shuffle=False)
test_gen  = ISICGenerator(test_X,  test_y,  batch=8, size=(256,256), shuffle=False)

# =============================================================================
# STEP 4.5: Quick dataset visualization
# =============================================================================
def show_batch(gen, n=3):
    xb, yb = gen[0]
    n = min(n, len(xb))
    plt.figure(figsize=(12, 4*n))
    for i in range(n):
        plt.subplot(n, 3, 3*i+1); plt.imshow(xb[i]); plt.title("Image"); plt.axis("off")
        plt.subplot(n, 3, 3*i+2); plt.imshow(yb[i].squeeze(), cmap="gray"); plt.title("Mask"); plt.axis("off")
        overlay = xb[i].copy()
        overlay[..., 0] = np.clip(overlay[..., 0] + 0.5*yb[i].squeeze(), 0, 1)
        plt.subplot(n, 3, 3*i+3); plt.imshow(overlay); plt.title("Overlay"); plt.axis("off")
    plt.tight_layout(); plt.show()

show_batch(train_gen, n=3)

# =============================================================================
# STEP 5: Loss + Metrics
# =============================================================================
def dice(y_t, y_p, smooth=1e-6):
    y_t = tf.cast(y_t, tf.float32)
    y_p = tf.cast(y_p, tf.float32)
    y_p = tf.clip_by_value(y_p, 0.0, 1.0)
    y_t = tf.reshape(y_t, [-1])
    y_p = tf.reshape(y_p, [-1])
    inter = tf.reduce_sum(y_t * y_p)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_t) + tf.reduce_sum(y_p) + smooth)

def dice_loss(y_t, y_p):
    return 1.0 - dice(y_t, y_p)

def iou(y_t, y_p, smooth=1e-6):
    y_t = tf.cast(y_t, tf.float32)
    y_p = tf.cast(y_p, tf.float32)
    y_p = tf.clip_by_value(y_p, 0.0, 1.0)
    y_t = tf.reshape(y_t, [-1])
    y_p = tf.reshape(y_p, [-1])
    inter = tf.reduce_sum(y_t * y_p)
    union = tf.reduce_sum(y_t) + tf.reduce_sum(y_p) - inter
    return (inter + smooth) / (union + smooth)

def combined_loss(y_t, y_p):
    bce = tf.keras.losses.binary_crossentropy(y_t, y_p)
    return bce + dice_loss(y_t, y_p)

# =============================================================================
# STEP 6: Dual Attention Module — Keras-native (no tf.reduce_* on KerasTensors)
#
#  The fix: replace tf.reduce_mean / tf.reduce_max with proper Keras layers:
#    - keras.layers.Lambda  wrapping keras.ops  (graph-safe)
#  Everything else is identical in logic.
# =============================================================================

class ChannelAttention(layers.Layer):
    """
    Squeeze-and-Excitation channel attention.
    Uses GlobalAvgPool + GlobalMaxPool → shared MLP → sigmoid gate.
    Implemented as a proper Keras Layer so it works inside Functional models.
    """
    def __init__(self, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.d1  = layers.Dense(channels // self.ratio, activation="relu", use_bias=False)
        self.d2  = layers.Dense(channels, use_bias=False)
        self.reshape = layers.Reshape((1, 1, channels))
        super().build(input_shape)

    def call(self, x):
        avg = self.d2(self.d1(self.gap(x)))
        mx  = self.d2(self.d1(self.gmp(x)))
        gate = tf.sigmoid(avg + mx)
        gate = self.reshape(gate)
        return x * gate

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class SpatialAttention(layers.Layer):
    """
    CBAM spatial attention.
    Avg + Max along channel dim → 7x7 conv → sigmoid gate.
    Implemented as a proper Keras Layer to avoid tf.reduce_* on KerasTensors.
    """
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1, self.kernel_size, padding="same",
            activation="sigmoid", use_bias=False
        )
        super().build(input_shape)

    def call(self, x):
        # keras.ops is graph-safe for KerasTensors
        avg_out = keras.ops.mean(x, axis=-1, keepdims=True)   # (B,H,W,1)
        max_out = keras.ops.max(x,  axis=-1, keepdims=True)   # (B,H,W,1)
        concat  = keras.ops.concatenate([avg_out, max_out], axis=-1)  # (B,H,W,2)
        gate    = self.conv(concat)                            # (B,H,W,1)
        return x * gate

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def dual_attention(x, ratio=8, kernel_size=7):
    """
    Apply ChannelAttention then SpatialAttention sequentially (CBAM order).
    Both are proper Keras Layers — fully compatible with Functional API.
    """
    x = ChannelAttention(ratio=ratio)(x)
    x = SpatialAttention(kernel_size=kernel_size)(x)
    return x

# =============================================================================
# STEP 7: InceptionV3 U-Net with Dual Attention on Skip Connections
# =============================================================================
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def up_block(x, skip, f, attn_ratio=8, attn_kernel=7):
    x = layers.Conv2DTranspose(f, 2, strides=2, padding="same")(x)

    # Dual Attention on skip connection
    skip = dual_attention(skip, ratio=attn_ratio, kernel_size=attn_kernel)

    # Resize if spatial dims mismatch (InceptionV3 quirk)
    if (x.shape[1] != skip.shape[1]) or (x.shape[2] != skip.shape[2]):
        x = layers.Resizing(skip.shape[1], skip.shape[2], interpolation="bilinear")(x)

    x = layers.Concatenate()([x, skip])
    x = conv_block(x, f)
    return x


def build_inception_unet_dual_attn(input_shape=(256,256,3), base_trainable=False,
                                   attn_ratio=8, attn_kernel=7):
    keras.backend.clear_session()

    inp = layers.Input(shape=input_shape)

    base = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base.trainable = base_trainable

    s1 = base.get_layer("mixed0").output
    s2 = base.get_layer("mixed3").output
    s3 = base.get_layer("mixed6").output
    s4 = base.get_layer("mixed9").output
    b  = base.get_layer("mixed10").output

    feat = models.Model(
        inputs=base.input,
        outputs=[s1, s2, s3, s4, b],
        name="InceptionV3_Features"
    )
    s1, s2, s3, s4, b = feat(inp)

    d1 = up_block(b,  s4, 512, attn_ratio=attn_ratio, attn_kernel=attn_kernel)
    d2 = up_block(d1, s3, 256, attn_ratio=attn_ratio, attn_kernel=attn_kernel)
    d3 = up_block(d2, s2, 128, attn_ratio=attn_ratio, attn_kernel=attn_kernel)
    d4 = up_block(d3, s1,  64, attn_ratio=attn_ratio, attn_kernel=attn_kernel)

    x = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(d4)
    if (x.shape[1] != input_shape[0]) or (x.shape[2] != input_shape[1]):
        x = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(x)

    x = conv_block(x, 32)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)

    return models.Model(inp, out, name="Inception_UNet_DualAttn_AdamW")


model = build_inception_unet_dual_attn(
    (256, 256, 3),
    base_trainable=False,
    attn_ratio=8,
    attn_kernel=7
)
model.summary()

# =============================================================================
# STEP 8: Compile with AdamW
# =============================================================================
adamw = keras.optimizers.AdamW(
    learning_rate=1e-4,
    weight_decay=1e-4,
)

model.compile(
    optimizer=adamw,
    loss=combined_loss,
    metrics=[dice, iou, "accuracy"]
)

# =============================================================================
# STEP 9: Train
# =============================================================================
callbacks_list = [
    callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.keras"),
        monitor="val_dice",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_dice",
        patience=12,
        restore_best_weights=True,
        mode="max",
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_dice",
        factor=0.5,
        patience=4,
        mode="max",
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks_list
)

# =============================================================================
# STEP 9.5: Training curves visualization
# =============================================================================
def plot_history(hist):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title("Loss"); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(hist.history["dice"], label="train")
    plt.plot(hist.history["val_dice"], label="val")
    plt.title("Dice"); plt.legend(); plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(hist.history["iou"], label="train")
    plt.plot(hist.history["val_iou"], label="val")
    plt.title("IoU"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_history(history)

# =============================================================================
# STEP 10: Evaluate on TEST set
# =============================================================================
res = model.evaluate(test_gen, verbose=1)
print(dict(zip(model.metrics_names, res)))

# =============================================================================
# STEP 10.5: Prediction visualization
# =============================================================================
def visualize_predictions(model, gen, n=3, thr=0.5):
    xb, yb = gen[0]
    preds = model.predict(xb, verbose=0)
    preds_bin = (preds > thr).astype(np.float32)

    n = min(n, len(xb))
    plt.figure(figsize=(12, 4*n))
    for i in range(n):
        plt.subplot(n, 4, 4*i+1); plt.imshow(xb[i]); plt.title("Image"); plt.axis("off")
        plt.subplot(n, 4, 4*i+2); plt.imshow(yb[i].squeeze(), cmap="gray"); plt.title("GT Mask"); plt.axis("off")
        plt.subplot(n, 4, 4*i+3); plt.imshow(preds[i].squeeze(), cmap="gray"); plt.title("Pred Prob"); plt.axis("off")
        plt.subplot(n, 4, 4*i+4); plt.imshow(preds_bin[i].squeeze(), cmap="gray"); plt.title("Pred Bin"); plt.axis("off")
    plt.tight_layout(); plt.show()

visualize_predictions(model, test_gen, n=4)

# =============================================================================
# STEP 11: Save final model
# =============================================================================
model.save(os.path.join(OUTPUT_DIR, "final_model.keras"))
print("✅ DONE — InceptionV3 U-Net + Dual Attention + AdamW trained on full train/val/test folders")
