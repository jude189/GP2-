# GP2-


This repository contains segmentation models implemented in PyTorch and TensorFlow with different optimizers and architectural variants.

---

## 🔥 PyTorch

### Adam → ResNetUNet
- Scheduled Learning Rate (Stepwise)  
  → `pytorch/Adam/ResNetUNet/Scheduled_Learning_Rate_Stepwise/`

- Scheduled Learning Rate (Exponential Decay)  
  → `pytorch/Adam/ResNetUNet/Scheduled_Learning_Rate_Exponential_Decay/`

### AdamW → ResNetUNet
- Baseline  
  → `pytorch/AdamW/ResNetUNet/`

---

## 🧠 TensorFlow

### InceptionV3

- With Skip  
  → `tensorflow/InceptionV3/With_Skip/`

- With Skip + Attention  
  → `tensorflow/InceptionV3/With_Skip/With_Attention/`

- With Skip + Attention + Spatial on Skip  
  → `tensorflow/InceptionV3/With_Skip/With_Attention/Spatial_on_Skip/`

- With Skip + Attention + Dual Attention on Skip  
  → `tensorflow/InceptionV3/With_Skip/With_Attention/Dual_on_Skip/`

---

## 📌 Project Structure

framework / optimizer-or-backbone / model / variant
