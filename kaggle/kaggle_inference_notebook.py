"""
PhysioNet ECG Image Digitization - Kaggle Inference Script
===========================================================

Bu script Kaggle notebook'unda çalıştırılmak üzere hazırlanmıştır.
Tüm adımları otomatik olarak gerçekleştirir.

Kullanım:
1. Yeni Kaggle notebook oluşturun
2. GPU'yu aktif edin (Settings > Accelerator > GPU)
3. Internet'i açın (Settings > Internet > ON)
4. Bu scripti çalıştırın

Author: PhysioNet Challenge Team
"""

import os
import sys
from pathlib import Path
import subprocess

print("=" * 80)
print("PhysioNet ECG Image Digitization - Kaggle Inference Pipeline")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Setup ve Kurulum
# ============================================================================
print("📦 STEP 1: Projeyi GitHub'dan klonlama ve kurulum")
print("-" * 80)

# GitHub'dan klonla
if not os.path.exists('/kaggle/working/PhysioNet-ECG-Image-Digitization-Challenge-2024'):
    print("Klonlanıyor...")
    subprocess.run([
        'git', 'clone',
        'https://github.com/EmreUludasdemir/PhysioNet-ECG-Image-Digitization-Challenge-2024.git'
    ], cwd='/kaggle/working')
else:
    print("✓ Proje zaten mevcut")

# Proje dizinine geç
os.chdir('/kaggle/working/PhysioNet-ECG-Image-Digitization-Challenge-2024')
print(f"✓ Çalışma dizini: {os.getcwd()}")

# Branch'i checkout et
print("Branch kontrol ediliyor...")
subprocess.run(['git', 'checkout', 'claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq'])
print("✓ Branch: claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq")

# Gerekli paketleri yükle
print("\nGerekli paketler yükleniyor...")
packages = [
    'segmentation-models-pytorch',
    'timm',
    'albumentations',
    'opencv-python',
    'scikit-image',
    'scipy',
    'pandas',
    'tqdm'
]

for package in packages:
    subprocess.run(['pip', 'install', '-q', package])
    print(f"✓ {package}")

print("\n✅ Kurulum tamamlandı!")
print()


# ============================================================================
# STEP 2: Import'lar ve Konfigürasyon
# ============================================================================
print("📚 STEP 2: Modülleri yükleme")
print("-" * 80)

# Path ekle
sys.path.insert(0, '/kaggle/working/PhysioNet-ECG-Image-Digitization-Challenge-2024')

# Import'lar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2
import warnings
warnings.filterwarnings('ignore')

# Proje modülleri
try:
    from src.config import get_config
    from src.inference import ECGInferencePipeline
    from src.data_preprocessing import ECGImagePreprocessor
    from src.evaluation import ECGEvaluator
    from src.segmentation_model import create_model
    from src.vectorization import ECGVectorizer
    print("✅ Tüm modüller başarıyla yüklendi!")
except ImportError as e:
    print(f"❌ Modül yükleme hatası: {e}")
    sys.exit(1)

# Device kontrolü
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🔧 Device: {device.upper()}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Config yükle
config = get_config()
print("✓ Konfigürasyon yüklendi")
print()


# ============================================================================
# STEP 3: Test Verisi Kontrolü
# ============================================================================
print("📂 STEP 3: Test verisi kontrol")
print("-" * 80)

# Olası test veri lokasyonları
possible_paths = [
    '/kaggle/input/physionet-ecg-digitization-challenge-2024/test_images',
    '/kaggle/input/physionet-challenge-2024/test',
    '/kaggle/input/ecg-test-images',
    '/kaggle/input/test-images',
]

test_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        test_data_path = path
        print(f"✓ Test verisi bulundu: {path}")
        break

if test_data_path is None:
    print("⚠️ Test verisi bulunamadı!")
    print("Aşağıdaki lokasyonlarda arandı:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nDemo modunda devam ediliyor (dummy data ile)...")
    USE_DUMMY_DATA = True
    test_images = []
else:
    # Test görsellerini bul
    test_images = (
        list(Path(test_data_path).glob('*.png')) +
        list(Path(test_data_path).glob('*.jpg')) +
        list(Path(test_data_path).glob('*.jpeg')) +
        list(Path(test_data_path).glob('*.PNG')) +
        list(Path(test_data_path).glob('*.JPG'))
    )
    print(f"✓ {len(test_images)} test görseli bulundu")
    USE_DUMMY_DATA = False

# İlk görseli görselleştir
if len(test_images) > 0:
    print("\n📊 Örnek görsel görselleştiriliyor...")
    sample_img = cv2.imread(str(test_images[0]))
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))
    plt.imshow(sample_img)
    plt.title(f"Örnek ECG Görüntüsü: {test_images[0].name}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/kaggle/working/sample_ecg_image.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Görsel kaydedildi: sample_ecg_image.png")

print()


# ============================================================================
# STEP 4: Model Yükleme veya Dummy Model Oluşturma
# ============================================================================
print("🤖 STEP 4: Model yükleme")
print("-" * 80)

# Model path'leri
possible_model_paths = [
    '/kaggle/input/ecg-model-weights/fold_0_best.pth',
    '/kaggle/input/ecg-model/best_model.pth',
    '/kaggle/input/physionet-model/fold_0_best.pth',
]

model_path = None
for path in possible_model_paths:
    if os.path.exists(path):
        model_path = path
        print(f"✓ Model bulundu: {path}")
        break

if model_path is None:
    print("⚠️ Eğitilmiş model bulunamadı!")
    print("Aşağıdaki lokasyonlarda arandı:")
    for path in possible_model_paths:
        print(f"  - {path}")
    print("\n⚠️ DUMMY MODEL MODU")
    print("Not: Bu mod sadece test içindir. Gerçek sonuçlar için eğitilmiş model gerekir!")

    USE_REAL_MODEL = False

    # Dummy prediction fonksiyonu
    preprocessor = ECGImagePreprocessor()
    vectorizer = ECGVectorizer()

    def predict_image(image_path):
        """Dummy prediction - eğitilmiş model olmadan"""
        # Preprocessing yap
        preprocessed = preprocessor.preprocess(image_path, apply_normalization=False)

        # Random signal üret (gerçek değil!)
        num_leads = 12
        signal_length = 5000

        # Biraz daha gerçekçi görünmesi için sinüzoidal bileşenler ekle
        t = np.linspace(0, 10, signal_length)
        dummy_signals = np.zeros((num_leads, signal_length))

        for i in range(num_leads):
            # Her lead için farklı frekanslar
            freq1 = 1.0 + i * 0.1  # Ana kalp atışı
            freq2 = 10.0 + i * 0.5  # Yüksek frekans bileşeni

            dummy_signals[i] = (
                0.8 * np.sin(2 * np.pi * freq1 * t) +
                0.2 * np.sin(2 * np.pi * freq2 * t) +
                0.1 * np.random.randn(signal_length)
            )

        return dummy_signals

else:
    print("✓ Gerçek model kullanılıyor")
    USE_REAL_MODEL = True

    try:
        # Pipeline oluştur
        pipeline = ECGInferencePipeline(
            model_path=model_path,
            config=config,
            device=device
        )
        print("✅ Model başarıyla yüklendi!")

        def predict_image(image_path):
            """Gerçek model ile prediction"""
            return pipeline.predict(
                image_path,
                correct_rotation=True,
                threshold=0.5,
                return_dict=False
            )

    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        print("Dummy mode'a geçiliyor...")
        USE_REAL_MODEL = False

        preprocessor = ECGImagePreprocessor()

        def predict_image(image_path):
            preprocessed = preprocessor.preprocess(image_path, apply_normalization=False)
            return np.random.randn(12, 5000) * 0.5

print()


# ============================================================================
# STEP 5: Test Prediction
# ============================================================================
print("🧪 STEP 5: Test prediction")
print("-" * 80)

if len(test_images) > 0:
    print("İlk görsel üzerinde test prediction yapılıyor...")

    try:
        test_signal = predict_image(test_images[0])
        print(f"✓ Prediction tamamlandı")
        print(f"  Shape: {test_signal.shape}")
        print(f"  Range: [{test_signal.min():.3f}, {test_signal.max():.3f}] mV")
        print(f"  Mean: {test_signal.mean():.3f} mV")
        print(f"  Std: {test_signal.std():.3f} mV")

        # Görselleştir
        print("\n📊 Sinyal görselleştirme...")
        fig, axes = plt.subplots(4, 3, figsize=(20, 15))
        axes = axes.flatten()

        lead_names = config.data.lead_names
        time = np.arange(1000) / 500  # İlk 1000 sample, 500 Hz

        for i, lead_name in enumerate(lead_names):
            ax = axes[i]
            ax.plot(time, test_signal[i, :1000], 'b-', linewidth=0.8)
            ax.set_title(f'Lead {lead_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (mV)')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Test Prediction - İlk 2 saniye', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/kaggle/working/test_prediction.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Görsel kaydedildi: test_prediction.png")

    except Exception as e:
        print(f"❌ Test prediction hatası: {e}")

else:
    print("⚠️ Test görseli bulunamadı, test prediction atlanıyor")

print()


# ============================================================================
# STEP 6: Batch Prediction - Tüm Test Setini İşle
# ============================================================================
print("🚀 STEP 6: Batch prediction")
print("-" * 80)

predictions = {}

if len(test_images) > 0:
    print(f"Toplam {len(test_images)} görsel işleniyor...")

    success_count = 0
    error_count = 0

    for image_path in tqdm(test_images, desc="Processing"):
        try:
            signals = predict_image(image_path)
            record_id = image_path.stem
            predictions[record_id] = signals
            success_count += 1

        except Exception as e:
            error_count += 1
            if error_count <= 5:  # İlk 5 hatayı göster
                tqdm.write(f"❌ Hata ({image_path.name}): {e}")

    print(f"\n✅ Başarılı: {success_count}/{len(test_images)}")
    if error_count > 0:
        print(f"❌ Hatalı: {error_count}/{len(test_images)}")

    # İstatistikler
    if len(predictions) > 0:
        all_signals = np.stack(list(predictions.values()))
        print(f"\n📊 Prediction İstatistikleri:")
        print(f"   Shape: {all_signals.shape}")
        print(f"   Min: {all_signals.min():.3f} mV")
        print(f"   Max: {all_signals.max():.3f} mV")
        print(f"   Mean: {all_signals.mean():.3f} mV")
        print(f"   Std: {all_signals.std():.3f} mV")

else:
    print("⚠️ Test görseli yok, dummy prediction oluşturuluyor...")
    # Demo için 5 dummy prediction oluştur
    for i in range(5):
        record_id = f"dummy_record_{i:03d}"
        dummy_signal = np.random.randn(12, 5000) * 0.5
        predictions[record_id] = dummy_signal

    print(f"✓ {len(predictions)} dummy prediction oluşturuldu")

print()


# ============================================================================
# STEP 7: Submission File Oluştur
# ============================================================================
print("📝 STEP 7: Kaggle submission dosyası oluşturma")
print("-" * 80)

lead_names = config.data.lead_names

print("Submission formatı hazırlanıyor...")
rows = []

for record_id, signals in tqdm(predictions.items(), desc="Creating submission"):
    for lead_idx, lead_name in enumerate(lead_names):
        for time_idx in range(signals.shape[1]):
            rows.append({
                'record_id': record_id,
                'lead': lead_name,
                'time': time_idx,
                'value': float(signals[lead_idx, time_idx])
            })

submission_df = pd.DataFrame(rows)

# Kaydet
submission_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\n✅ Submission dosyası oluşturuldu!")
print(f"   Path: {submission_path}")
print(f"   Toplam satır: {len(submission_df):,}")
print(f"   Toplam record: {len(predictions)}")
print(f"   Dosya boyutu: {os.path.getsize(submission_path) / (1024*1024):.2f} MB")

# Önizleme
print(f"\n📋 Submission Önizlemesi (ilk 20 satır):")
print(submission_df.head(20).to_string())

print()


# ============================================================================
# STEP 8: Submission Validation
# ============================================================================
print("✅ STEP 8: Submission validation")
print("-" * 80)

print("Submission dosyası kontrol ediliyor...\n")

# Boyut kontrolü
print(f"✓ Toplam satır: {len(submission_df):,}")
print(f"✓ Kolonlar: {list(submission_df.columns)}")

# Eksik değer kontrolü
missing = submission_df.isnull().sum().sum()
if missing > 0:
    print(f"⚠️ {missing} eksik değer bulundu!")
else:
    print(f"✓ Eksik değer yok")

# Record sayısı
unique_records = submission_df['record_id'].nunique()
print(f"✓ Unique record sayısı: {unique_records}")

# Lead kontrolü
unique_leads = submission_df['lead'].nunique()
expected_leads = len(lead_names)
if unique_leads == expected_leads:
    print(f"✓ Lead sayısı doğru: {unique_leads}/{expected_leads}")
else:
    print(f"⚠️ Lead sayısı hatalı: {unique_leads}/{expected_leads}")

# Değer aralığı kontrolü
val_min = submission_df['value'].min()
val_max = submission_df['value'].max()
val_mean = submission_df['value'].mean()
print(f"✓ Değer aralığı: [{val_min:.3f}, {val_max:.3f}] mV")
print(f"✓ Ortalama değer: {val_mean:.3f} mV")

# Dosya boyutu
file_size_mb = os.path.getsize(submission_path) / (1024 * 1024)
print(f"✓ Dosya boyutu: {file_size_mb:.2f} MB")

# NaN/Inf kontrolü
if np.isinf(submission_df['value']).any():
    print("⚠️ Infinity değerleri tespit edildi!")
else:
    print("✓ Infinity değeri yok")

print("\n" + "=" * 80)

if missing == 0 and unique_leads == expected_leads and not np.isinf(submission_df['value']).any():
    print("✅✅✅ SUBMISSION HAZIR! SUBMIT EDEBİLİRSİNİZ! ✅✅✅")
else:
    print("⚠️ Submission'da bazı sorunlar var, lütfen kontrol edin")

print("=" * 80)
print()


# ============================================================================
# STEP 9: Görselleştirme ve Raporlama
# ============================================================================
print("📊 STEP 9: Sonuç görselleştirme")
print("-" * 80)

if len(predictions) > 0:
    # Rastgele bir record seç
    import random
    random_record = random.choice(list(predictions.keys()))
    signals = predictions[random_record]

    print(f"Görselleştirilen record: {random_record}\n")

    # Tüm 12 lead'i görselleştir
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()

    for i, lead_name in enumerate(lead_names):
        ax = axes[i]

        # Tüm sinyali çiz (10 saniye)
        time = np.arange(signals.shape[1]) / 500  # 500 Hz sampling rate
        ax.plot(time, signals[i], 'b-', linewidth=0.5)

        ax.set_title(f'Lead {lead_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude (mV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10])  # 10 saniye

        # İstatistikler ekle
        stats_text = f'Min: {signals[i].min():.2f}\nMax: {signals[i].max():.2f}\nMean: {signals[i].mean():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'12-Lead ECG Signal - Record: {random_record}',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/kaggle/working/ecg_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Görsel kaydedildi: ecg_visualization.png")

    # Per-lead istatistikler
    print("\n📊 Lead İstatistikleri:")
    print("-" * 80)
    print(f"{'Lead':<6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 80)
    for i, lead_name in enumerate(lead_names):
        print(f"{lead_name:<6} {signals[i].min():>10.3f} {signals[i].max():>10.3f} "
              f"{signals[i].mean():>10.3f} {signals[i].std():>10.3f}")

print()


# ============================================================================
# ÖZET VE SONUÇ
# ============================================================================
print("=" * 80)
print("🎉 PIPELINE TAMAMLANDI!")
print("=" * 80)

print(f"\n📊 ÖZET:")
print(f"   • İşlenen görsel sayısı: {len(predictions)}")
print(f"   • Submission satır sayısı: {len(submission_df):,}")
print(f"   • Model tipi: {'GERÇEK MODEL' if USE_REAL_MODEL else 'DUMMY MODEL (Test)'}")
print(f"   • Submission dosyası: {submission_path}")
print(f"   • Dosya boyutu: {file_size_mb:.2f} MB")

print(f"\n📁 OLUŞTURULAN DOSYALAR:")
output_files = [
    '/kaggle/working/submission.csv',
    '/kaggle/working/ecg_visualization.png',
    '/kaggle/working/test_prediction.png',
    '/kaggle/working/sample_ecg_image.png',
]

for file_path in output_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024  # KB
        print(f"   ✓ {file_path} ({size:.1f} KB)")

print(f"\n🚀 SONRAKI ADIMLAR:")
if not USE_REAL_MODEL:
    print("   1. ⚠️ DUMMY MODEL KULLANILDI! Gerçek sonuçlar için:")
    print("      - Model eğitin: scripts/train.py")
    print("      - Eğitilmiş modeli Kaggle'a dataset olarak yükleyin")
    print("      - Bu scripti tekrar çalıştırın")
    print()

print("   2. submission.csv dosyasını indirin")
print("   3. Kaggle Competition sayfasına gidin")
print("   4. 'Submit Predictions' butonuna tıklayın")
print("   5. submission.csv dosyasını yükleyin")
print("   6. Sonuçları bekleyin!")

print("\n" + "=" * 80)
print("✅ Script başarıyla tamamlandı!")
print("=" * 80)
