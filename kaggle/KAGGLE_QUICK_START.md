# 🚀 Kaggle Hızlı Başlangıç Kılavuzu

## ⚠️ ÖNEMLİ: Yarışma Verileri Gerekli

Script çalışması için **PhysioNet ECG yarışmasının input datasını notebook'a eklemelisiniz**:
1. Kaggle notebook'unuzda **"Add Data"** butonuna tıklayın
2. PhysioNet ECG Image Digitization Competition'ı arayın ve ekleyin
3. Gerekli dosyalar: `test_images/` dizini ve `sample_submission.csv`

## ⚡ EN HIZLI YOL - Tek Komutla Başlat

Kaggle notebook'unuzda **tek bir cell**'de şunu çalıştırın:

```python
!wget https://raw.githubusercontent.com/EmreUludasdemir/PhysioNet-ECG-Image-Digitization-Challenge-2024/claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq/kaggle/kaggle_inference_notebook.py
!python kaggle_inference_notebook.py
```

**İşte bu kadar!** Script otomatik olarak:
- ✅ Projeyi GitHub'dan klonlar
- ✅ Gerekli paketleri yükler
- ✅ NumPy 2.x uyumluluk sorununu çözer (otomatik downgrade)
- ✅ Test verilerini ve sample_submission.csv'yi bulur
- ✅ Modeli yükler (veya dummy model oluşturur)
- ✅ TÜM test görsellerini işler
- ✅ Submission dosyası oluşturur
- ✅ Görselleştirme yapar

**Not:** Script, matplotlib uyumluluğu için NumPy'ı otomatik olarak 1.x versiyonuna downgrade eder.

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
!pip install 'numpy<2.0' --force-reinstall -q  # matplotlib uyumluluğu için
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

### 1. Test Images (ZORUNLU)
Script çalışması için mutlaka gerekli:
```
Add Data > Competition > PhysioNet ECG Image Digitization
```
Bu dataset şunları içermelidir:
- `test_images/` veya `test/` dizini (tüm test görselleri)
- `sample_submission.csv` veya `sample_submission.parquet` (record_id listesi)
- Alternatif: `test.csv` (record_id'ler için fallback)

### 2. Eğitilmiş Model (opsiyonel)
Eğer modelinizi eğittiyseniz:
```
Add Data > Upload > your_model.pth
```
Model yoksa script dummy model ile çalışır (rastgele tahminler)

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

### Test Verileri (ZORUNLU)
Script artık **gerçek test verileri** ile çalışır:
- ✅ sample_submission dosyasından record_id'leri okur (CSV veya Parquet)
- ✅ Her record_id için tahmin yapar
- ⚠️ Görseli olmayan kayıtlar için sıfır değerli signal kullanır

### Desteklenen Dosya Formatları
- ✅ CSV: `sample_submission.csv`, `test.csv`
- ✅ Parquet: `sample_submission.parquet`
- ✅ Otomatik format tespiti

### Submission Format
Script otomatik olarak yarışmanın formatını kullanır:
- Input: `id` kolonu (format: `{record_id}_{time}_{lead}`)
- Output: `id` ve `value` kolonları
- Örnek: `1053922973_0_I` → record_id `1053922973`, time `0`, lead `I`

### Dummy Model Modu
Eğer eğitilmiş model yoksa script **DUMMY MODE**'da çalışır:
- ✅ Pipeline test edilebilir
- ✅ Submission formatı doğrulanır
- ✅ Gerçek record_id'ler kullanılır
- ❌ Sonuçlar rastgele (gerçek tahmin değil)

**Gerçek sonuçlar için:** Önce modeli eğitin!

### Model Eğitimi İçin
```bash
# Lokal makinenizde:
python scripts/train.py --data_dir data/raw --epochs 100

# Model'i Kaggle'a yükleyin
```

---

## 🆘 Sorun Giderme

### "AttributeError: _ARRAY_API not found" veya NumPy hatası
Script artık bunu otomatik çözüyor. Manuel çözüm:
```python
!pip install 'numpy<2.0' --force-reinstall -q
# Kernel'i restart edin
```

### "Module not found" hatası
```python
!pip install --upgrade segmentation-models-pytorch
```

### "CUDA out of memory" hatası
```python
# Batch size'ı küçültün config'de
# veya CPU modunda çalıştırın
```

### "sample_submission dosyası bulunamadı" hatası
- Yarışmanın input datasını notebook'a ekleyin
- Add Data > Competition > PhysioNet ECG Image Digitization
- Şu dosyalardan biri olmalı:
  - `sample_submission.csv`
  - `sample_submission.parquet`
  - `test.csv` (fallback olarak)

### "Test görselleri bulunamadı" hatası
- test_images/ dizininin input'ta olduğundan emin olun
- Dizin yapısı: `/kaggle/input/[competition-name]/test_images/*.png`

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
