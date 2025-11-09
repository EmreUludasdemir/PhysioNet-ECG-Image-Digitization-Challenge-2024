# 🚀 Kaggle Hızlı Başlangıç Kılavuzu

## ⚡ EN HIZLI YOL - Tek Komutla Başlat

Kaggle notebook'unuzda **tek bir cell**'de şunu çalıştırın:

```python
!wget https://raw.githubusercontent.com/EmreUludasdemir/PhysioNet-ECG-Image-Digitization-Challenge-2024/claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq/kaggle/kaggle_inference_notebook.py
!python kaggle_inference_notebook.py
```

**İşte bu kadar!** Script otomatik olarak:
- ✅ Projeyi GitHub'dan klonlar
- ✅ Gerekli paketleri yükler
- ✅ Modeli yükler (veya dummy model oluşturur)
- ✅ Test görsellerini işler
- ✅ Submission dosyası oluşturur
- ✅ Görselleştirme yapar

---

## 📋 Alternatif: Adım Adım Manuel Kurulum

### 1. Kaggle Notebook Ayarları

Yeni notebook oluşturduktan sonra:

- **Accelerator:** GPU (T4 veya P100)
- **Internet:** ON
- **Persistence:** Files only

### 2. Projeyi Klonla

```python
!git clone https://github.com/EmreUludasdemir/PhysioNet-ECG-Image-Digitization-Challenge-2024.git
%cd PhysioNet-ECG-Image-Digitization-Challenge-2024
!git checkout claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq
```

### 3. Paketleri Yükle

```python
!pip install -q segmentation-models-pytorch timm albumentations
```

### 4. Inference Scriptini Çalıştır

```python
!python kaggle/kaggle_inference_notebook.py
```

### 5. Submission'ı İndir

```python
from IPython.display import FileLink
FileLink('/kaggle/working/submission.csv')
```

---

## 📦 Gerekli Kaggle Datasets

Gerçek sonuçlar için şu dataset'leri ekleyin:

### 1. Test Images
```
Add Data > Search: "physionet ecg images"
```

### 2. Eğitilmiş Model (opsiyonel)
Eğer modelinizi eğittiyseniz:
```
Add Data > Upload > your_model.pth
```

---

## 🎯 Beklenen Çıktılar

Script çalıştığında şu dosyalar oluşturulur:

```
/kaggle/working/
├── submission.csv          ← BUNU SUBMIT EDİN
├── ecg_visualization.png   ← Sonuç görseli
├── test_prediction.png     ← Test prediction
└── sample_ecg_image.png    ← Örnek ECG
```

---

## ⚠️ Önemli Notlar

### Dummy Model Modu
Eğer eğitilmiş model yoksa script **DUMMY MODE**'da çalışır:
- ✅ Pipeline test edilebilir
- ✅ Submission formatı doğrulanır
- ❌ Sonuçlar rastgele (gerçek değil)

**Gerçek sonuçlar için:** Önce modeli eğitin!

### Model Eğitimi İçin
```bash
# Lokal makinenizde:
python scripts/train.py --data_dir data/raw --epochs 100

# Model'i Kaggle'a yükleyin
```

---

## 🆘 Sorun Giderme

### "Module not found" hatası
```python
!pip install --upgrade segmentation-models-pytorch
```

### "CUDA out of memory" hatası
```python
# Batch size'ı küçültün config'de
# veya CPU modunda çalıştırın
```

### "No test images found" uyarısı
- Test dataset'i Kaggle'a ekleyin
- Veya demo modu için devam edin (dummy data kullanır)

---

## 📊 Sonuçlar

Script tamamlandığında:

```
✅ SUBMISSION HAZIR! SUBMIT EDEBİLİRSİNİZ!
```

mesajını görmelisiniz.

**Submission.csv** dosyasını indirip Kaggle Competition'a submit edin!

---

## 🎓 Daha Fazla Bilgi

- 📖 Ana README: `/README.md`
- 🔧 Konfigürasyon: `/src/config.py`
- 🧪 Test scriptleri: `/scripts/`
- 📓 Detaylı dokümantasyon: `/notebooks/README.md`

---

## 💡 İpuçları

1. **İlk çalıştırma:** Dummy mode ile test edin
2. **Model eğitimi:** Lokal makinede veya Kaggle'da eğitin
3. **Ensemble:** Birden fazla model kullanın
4. **TTA:** Test-time augmentation ile accuracy artırın

**Başarılar!** 🚀
