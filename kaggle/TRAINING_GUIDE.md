# 🎓 Model Training Guide

## 📋 Overview

Bu kılavuz, ECG image-to-signal regression modelini eğitmek için adım adım talimatlar içerir.

---

## 🎯 Problem Tanımı

**Input**: Taranmış ECG kağıdı görseli (1700×2200×3 PNG)
**Output**: 12-lead dijital sinyal (12 leads × 5000 timesteps)

**Yaklaşım**: End-to-end regression (supervised learning)

---

## 📊 Training Data

- **977 training samples**
- Her sample için:
  - 9 görsel (farklı açılar/crops)
  - 1 CSV (ground truth signal)
  - Metadata (fs, sig_len)

```
train/
  └─ 735384893/
      ├─ 735384893-0001.png
      ├─ 735384893-0002.png
      ├─ ...
      └─ 735384893.csv  (12 leads × 5000 timesteps)
```

---

## 🏗️ Model Mimarisi

```
Input Image (512×512×3)
    ↓
Encoder: EfficientNet-B2 (pre-trained)
    ↓
Global Average Pooling
    ↓
FC Layer (2048)
    ↓
FC Layer (4096)
    ↓
FC Layer (12 × 5000)
    ↓
Output Signal (12, 5000)
```

**Özellikler:**
- Transfer learning: ImageNet pre-trained weights
- Encoder: EfficientNet-B2 (~9M parameters)
- Toplam: ~25M parameters

---

## 🔧 Hyperparameters

```python
IMAGE_SIZE = (512, 512)        # Resized from 1700×2200
BATCH_SIZE = 8                 # GPU memory'ye göre ayarla
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
PATIENCE = 10                  # Early stopping

ENCODER = 'efficientnet_b2'
TRAIN_SPLIT = 0.85             # 831 train, 146 val
```

---

## 📉 Loss Function

**Combined Loss** = 0.5 × MSE + 0.5 × SNR Loss

1. **MSE Loss**: Standard L2 loss
2. **SNR Loss**: Yarışma metriği
   ```
   SNR = 10 × log10(signal_power / noise_power)
   Loss = -SNR
   ```

---

## 🚀 Kaggle'da Training

### 1️⃣ Script'i İndir

```python
!wget -O kaggle_training.py https://raw.githubusercontent.com/EmreUludasdemir/PhysioNet-ECG-Image-Digitization-Challenge-2024/claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq/kaggle/kaggle_training.py
```

### 2️⃣ Gerekli Kütüphaneleri Yükle

```python
!pip install -q timm albumentations opencv-python-headless
```

### 3️⃣ Training'i Başlat

```python
!python kaggle_training.py
```

**Süre**:
- ~5-6 saat (GPU T4)
- ~10-12 saat (CPU - önerilmez)

### 4️⃣ Model'i İndir

Training tamamlandığında:

```python
from IPython.display import FileLink
FileLink('/kaggle/working/best_model.pth')
```

---

## 📊 Beklenen Sonuçlar

### Training Metrics:

```
Epoch 1/50
Train Loss: 0.0045
Val Loss: 0.0038 | Val SNR: 15.2 dB

Epoch 10/50
Train Loss: 0.0012
Val Loss: 0.0010 | Val SNR: 22.5 dB

...

Best Val SNR: 24.8 dB
```

**Gerçekçi SNR Hedefleri:**
- ❌ Random model: ~5-10 dB
- ✅ Baseline: ~15-20 dB
- 🎯 İyi model: ~20-25 dB
- 🏆 Yarışma kazanan: ~25-30 dB

---

## 🔄 Eğitilmiş Modeli Kullanma

### Inference Script'inde Kullan

1. Model'i Kaggle dataset olarak yükle:
   - Dataset oluştur: "ecg-trained-model"
   - `best_model.pth` yükle

2. Inference notebook'a ekle:
   ```
   Add Data > Dataset > ecg-trained-model
   ```

3. Model path'ini güncelle:
   - `/kaggle/input/ecg-trained-model/best_model.pth`

Inference script otomatik olarak bu modeli kullanacak!

---

## ⚙️ Hyperparameter Tuning

### Batch Size
```python
# GPU Memory'ye göre:
T4 GPU (15GB):  BATCH_SIZE = 8
P100 (16GB):    BATCH_SIZE = 8-12
A100 (40GB):    BATCH_SIZE = 16-24
CPU:            BATCH_SIZE = 2-4 (çok yavaş!)
```

### Encoder Seçimi
```python
# Hız vs Accuracy trade-off:
'efficientnet_b0':  Hızlı, hafif (~5M params)
'efficientnet_b2':  Dengeli (~9M params) ✅ Önerilen
'efficientnet_b4':  Yavaş, güçlü (~19M params)
'resnet50':         Alternatif (~26M params)
```

### Learning Rate
```python
# Eğer overfit:
LEARNING_RATE = 5e-5  # Daha küçük

# Eğer underfit:
LEARNING_RATE = 2e-4  # Daha büyük
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```python
# Batch size'ı küçült
BATCH_SIZE = 4  # veya 2
```

### "Training çok yavaş"
```python
# Daha küçük encoder kullan
ENCODER_NAME = 'efficientnet_b0'

# Veya image size küçült
IMG_SIZE = (384, 384)
```

### "Validation loss artıyor"
```python
# Overfitting - regularization ekle
WEIGHT_DECAY = 1e-4  # Artır

# Veya early stopping patience azalt
PATIENCE = 5
```

### "SNR çok düşük"
```python
# Loss weights ayarla
mse_weight = 0.3
snr_weight = 0.7  # SNR'a daha fazla ağırlık
```

---

## 📈 Gelişmiş Teknikler

### Data Augmentation (TODO)
```python
# Eklenebilir:
- RandomRotation (küçük açılar)
- RandomBrightness
- RandomContrast
- GaussianNoise
```

### Multi-Image Learning (TODO)
```python
# Her record için 9 görsel var
# Hepsini kullan ve average al
```

### Ensemble (TODO)
```python
# Birden fazla model eğit
# Tahminleri average al
```

---

## 💡 İpuçları

1. **İlk 5 epoch'a dikkat et**: Hızlıca improvement görmeli
2. **Val SNR'ı takip et**: Bu yarışma metriği
3. **Checkpoint'leri kaydet**: Her epoch'ta kaydet
4. **GPU kullan**: CPU ile çok yavaş
5. **Patience ayarla**: Erken durmayı önle

---

## 📚 Kaynaklar

- [TimM Documentation](https://github.com/huggingface/pytorch-image-models)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [PhysioNet Challenge](https://physionet.org/)

---

## ✅ Checklist

Training öncesi:
- [ ] Kaggle GPU enabled
- [ ] Competition data added
- [ ] Libraries installed
- [ ] Script downloaded

Training sırasında:
- [ ] Training loss düşüyor
- [ ] Val SNR artıyor
- [ ] No CUDA errors
- [ ] Checkpoints saving

Training sonrası:
- [ ] best_model.pth indirildi
- [ ] SNR > 15 dB achieved
- [ ] Model Kaggle'a dataset olarak yüklendi

---

**Başarılar!** 🚀

Sorular için: GitHub Issues
