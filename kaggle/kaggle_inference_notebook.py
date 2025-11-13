"""
PhysioNet ECG Image Digitization - Kaggle Inference Script (Timeout Optimized)
===============================================================================

Kaggle timeout sorununu çözmek için optimize edilmiş versiyon.
Her adımda çıktı vererek session'ın kapanmasını engeller.

Author: PhysioNet Challenge Team
Version: 2.0 (Timeout-Safe)
"""

import os
import sys
from pathlib import Path
import subprocess
import time as time_module
from datetime import datetime

# Başlangıç zamanı
START_TIME = time_module.time()

def log(message, level="INFO"):
    """Zaman damgalı log mesajı (timeout'u engellemek için)"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}", flush=True)

def heartbeat(message="Still working..."):
    """Heartbeat mesajı - Kaggle timeout'unu engeller"""
    print(f"💓 {message}", flush=True)

print("=" * 80)
print("PhysioNet ECG Image Digitization - Kaggle Inference Pipeline v2.0")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Setup ve Kurulum
# ============================================================================
log("STEP 1: Projeyi GitHub'dan klonlama ve kurulum", "START")
print("-" * 80)

# Eski/bozuk dizinleri temizle (cache problemi için)
project_dir = '/kaggle/working/PhysioNet-ECG-Image-Digitization-Challenge-2024'
if os.path.exists(project_dir):
    log("Eski proje dizini tespit edildi, temizleniyor...")
    import shutil
    try:
        shutil.rmtree(project_dir)
        log("✓ Eski dizin temizlendi")
    except Exception as e:
        log(f"⚠️ Temizleme uyarısı: {e}", "WARNING")

# GitHub'dan fresh clone
log("GitHub'dan klonlanıyor...")
result = subprocess.run([
    'git', 'clone',
    'https://github.com/EmreUludasdemir/PhysioNet-ECG-Image-Digitization-Challenge-2024.git'
], cwd='/kaggle/working', capture_output=True, text=True)

if result.returncode != 0:
    log(f"❌ Clone hatası: {result.stderr}", "ERROR")
    sys.exit(1)

log("✓ Klonlama tamamlandı")

# Proje dizinine geç
os.chdir(project_dir)
log(f"✓ Çalışma dizini: {os.getcwd()}")

# Branch'i checkout et
log("Branch kontrol ediliyor...")
subprocess.run(['git', 'checkout', 'claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq'],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
log("✓ Branch: claude/physionet-ecg-digitization-011CUq26jaEWm593owfiQqvq")

# Gerekli paketleri yükle
log("Gerekli paketler yükleniyor...")
packages = [
    'segmentation-models-pytorch',
    'timm',
    'albumentations',
    'opencv-python-headless',
    'scikit-image',
    'scipy',
    'pandas',
    'tqdm'
]

for i, package in enumerate(packages, 1):
    heartbeat(f"Yükleniyor ({i}/{len(packages)}): {package}")
    subprocess.run(['pip', 'install', '-q', package],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"✓ {package}")

log("✅ Kurulum tamamlandı!", "SUCCESS")
print()

# NumPy 2.x uyumluluk sorunu için downgrade
log("NumPy versiyonu kontrol ediliyor...")
heartbeat("NumPy 1.x'e downgrade yapılıyor (matplotlib uyumluluğu için)...")
subprocess.run(
    ['pip', 'install', 'numpy<2.0', '--force-reinstall', '-q'],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
log("✓ NumPy downgrade tamamlandı")
print()


# ============================================================================
# STEP 2: Import'lar ve Konfigürasyon
# ============================================================================
log("STEP 2: Modülleri yükleme", "START")
print("-" * 80)

# Path ekle
sys.path.insert(0, '/kaggle/working/PhysioNet-ECG-Image-Digitization-Challenge-2024')

# Import'lar
heartbeat("Numpy ve temel kütüphaneler yükleniyor...")
import numpy as np
import pandas as pd
heartbeat("Matplotlib yükleniyor...")
import matplotlib
matplotlib.use('Agg')  # GUI olmadan çalış
import matplotlib.pyplot as plt
heartbeat("CV2 ve görüntü işleme kütüphaneleri yükleniyor...")
import cv2
import warnings
warnings.filterwarnings('ignore')
heartbeat("PyTorch yükleniyor...")
import torch
heartbeat("Tqdm yükleniyor...")
from tqdm import tqdm

log("✓ Temel kütüphaneler yüklendi")

# Proje modülleri
try:
    heartbeat("Proje modülleri yükleniyor...")
    from src.config import get_config
    from src.inference import ECGInferencePipeline
    from src.data_preprocessing import ECGImagePreprocessor
    from src.evaluation import ECGEvaluator
    from src.segmentation_model import create_model
    from src.vectorization import ECGVectorizer
    log("✅ Tüm modüller başarıyla yüklendi!", "SUCCESS")
except ImportError as e:
    log(f"❌ Modül yükleme hatası: {e}", "ERROR")
    sys.exit(1)

# Device kontrolü
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"🔧 Device: {device.upper()}")
if device == 'cuda':
    log(f"   GPU: {torch.cuda.get_device_name(0)}")
    log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Config yükle
config = get_config()
log("✓ Konfigürasyon yüklendi")
print()


# ============================================================================
# STEP 3: Test Verisi ve Sample Submission Kontrolü
# ============================================================================
log("STEP 3: Test verisi ve sample_submission.csv kontrol", "START")
print("-" * 80)

# Olası input lokasyonları (yarışmanın farklı isimleri için)
possible_input_dirs = [
    '/kaggle/input/physionet-ecg-image-digitization',
    '/kaggle/input/physionet-ecg-digitization-challenge-2024',
    '/kaggle/input/physionet-challenge-2024',
    '/kaggle/input',
]

# Sample submission dosyasını bul
sample_submission_path = None
test_data_path = None
competition_input_dir = None

for input_dir in possible_input_dirs:
    heartbeat(f"Kontrol ediliyor: {input_dir}")
    if os.path.exists(input_dir):
        # sample_submission dosyasını ara (csv veya parquet)
        possible_submission_files = [
            os.path.join(input_dir, 'sample_submission.csv'),
            os.path.join(input_dir, 'sample_submission.parquet'),
            os.path.join(input_dir, 'sampleSubmission.csv'),
            os.path.join(input_dir, 'sampleSubmission.parquet'),
            os.path.join(input_dir, 'SampleSubmission.csv'),
        ]

        for sub_file in possible_submission_files:
            if os.path.exists(sub_file):
                sample_submission_path = sub_file
                competition_input_dir = input_dir
                log(f"✓ Submission dosyası bulundu: {sub_file}")
                break

        # test_images dizini ara
        possible_test_dirs = [
            os.path.join(input_dir, 'test_images'),
            os.path.join(input_dir, 'test'),
            os.path.join(input_dir, 'images'),
        ]

        for test_dir in possible_test_dirs:
            if os.path.exists(test_dir):
                test_data_path = test_dir
                log(f"✓ Test görselleri dizini bulundu: {test_dir}")
                break

        # test.csv dosyasını da kontrol et (fallback için)
        test_csv_path = os.path.join(input_dir, 'test.csv')
        if os.path.exists(test_csv_path):
            if not sample_submission_path:
                sample_submission_path = test_csv_path
                competition_input_dir = input_dir
                log(f"✓ test.csv bulundu (fallback olarak kullanılacak): {test_csv_path}")

        if sample_submission_path and test_data_path:
            break

# sample_submission dosyasını oku ve gerçek record_id'leri al
record_ids = []
if sample_submission_path:
    heartbeat("Submission dosyası okunuyor...")
    import pandas as pd

    # Dosya formatına göre oku
    try:
        if sample_submission_path.endswith('.parquet'):
            heartbeat("Parquet dosyası okunuyor...")
            sample_df = pd.read_parquet(sample_submission_path)
            log(f"✓ Parquet dosyası okundu: {len(sample_df)} satır")
        else:
            sample_df = pd.read_csv(sample_submission_path)
            log(f"✓ CSV dosyası okundu: {len(sample_df)} satır")

        # record_id'leri çıkar (unique)
        if 'record_id' in sample_df.columns:
            # Standart format: record_id kolonu var
            record_ids = sorted(sample_df['record_id'].unique().tolist())
            log(f"✓ {len(record_ids)} adet record_id bulundu")
            log(f"   İlk 5 record: {record_ids[:5]}")
        elif 'id' in sample_df.columns:
            # Alternatif format: id kolonu var (örn: "record_id_time_lead")
            # id'den record_id'yi parse et (ilk underscore'a kadar)
            heartbeat("id kolonundan record_id'ler parse ediliyor...")
            sample_df['parsed_record_id'] = sample_df['id'].str.split('_').str[0]
            record_ids = sorted(sample_df['parsed_record_id'].unique().tolist())
            log(f"✓ {len(record_ids)} adet record_id parse edildi")
            log(f"   İlk 5 record: {record_ids[:5]}")
            log(f"   Örnek id format: {sample_df['id'].iloc[0]}")
        else:
            log("❌ 'record_id' veya 'id' kolonu bulunamadı!", "ERROR")
            log(f"   Mevcut kolonlar: {sample_df.columns.tolist()}", "ERROR")
            raise ValueError("Submission dosyası formatı hatalı")

    except Exception as e:
        log(f"❌ Dosya okuma hatası: {e}", "ERROR")
        raise

else:
    log("❌ sample_submission dosyası bulunamadı!", "ERROR")
    log("Lütfen Kaggle yarışmasının input datasını notebook'a ekleyin:", "ERROR")
    log("  1. Notebook ayarlarından 'Add Data' seçin", "ERROR")
    log("  2. PhysioNet ECG yarışmasının datasını ekleyin", "ERROR")
    raise FileNotFoundError("sample_submission dosyası bulunamadı")

# Test görsellerini record_id'lere göre eşleştir
if test_data_path:
    heartbeat("Test görselleri eşleştiriliyor...")
    test_images_dict = {}

    # Tüm görselleri tara
    all_images = (
        list(Path(test_data_path).glob('*.png')) +
        list(Path(test_data_path).glob('*.jpg')) +
        list(Path(test_data_path).glob('*.jpeg')) +
        list(Path(test_data_path).glob('*.PNG')) +
        list(Path(test_data_path).glob('*.JPG'))
    )

    # record_id'lere göre eşleştir
    for img_path in all_images:
        record_id = img_path.stem  # dosya adından uzantıyı çıkar
        if record_id in record_ids:
            test_images_dict[record_id] = img_path

    log(f"✓ {len(test_images_dict)}/{len(record_ids)} görsel eşleştirildi")

    if len(test_images_dict) == 0:
        log("❌ Hiçbir test görseli bulunamadı!", "ERROR")
        raise FileNotFoundError("Test görselleri bulunamadı")

    USE_DUMMY_DATA = False
else:
    log("❌ Test görselleri dizini bulunamadı!", "ERROR")
    raise FileNotFoundError("Test görselleri dizini bulunamadı")

# İlk görseli görselleştir
if len(test_images_dict) > 0:
    heartbeat("Örnek görsel yükleniyor...")
    first_record_id = list(test_images_dict.keys())[0]
    first_image_path = test_images_dict[first_record_id]

    sample_img = cv2.imread(str(first_image_path))
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

    heartbeat("Görsel kaydediliyor...")
    plt.figure(figsize=(15, 10))
    plt.imshow(sample_img)
    plt.title(f"Örnek ECG Görüntüsü: {first_image_path.name} (Record: {first_record_id})", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/kaggle/working/sample_ecg_image.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("✓ Görsel kaydedildi: sample_ecg_image.png")

print()


# ============================================================================
# STEP 4: Model Yükleme veya Dummy Model Oluşturma
# ============================================================================
log("STEP 4: Model yükleme", "START")
print("-" * 80)

# Model path'leri
possible_model_paths = [
    '/kaggle/input/ecg-model-weights/fold_0_best.pth',
    '/kaggle/input/ecg-model/best_model.pth',
    '/kaggle/input/physionet-model/fold_0_best.pth',
]

model_path = None
for path in possible_model_paths:
    heartbeat(f"Model kontrol ediliyor: {path}")
    if os.path.exists(path):
        model_path = path
        log(f"✓ Model bulundu: {path}")
        break

if model_path is None:
    log("⚠️ Eğitilmiş model bulunamadı! DUMMY MODEL MODU", "WARNING")
    log("Not: Bu mod sadece test içindir. Gerçek sonuçlar için eğitilmiş model gerekir!")

    USE_REAL_MODEL = False

    # Dummy prediction fonksiyonu
    preprocessor = ECGImagePreprocessor()
    vectorizer = ECGVectorizer()

    def predict_image(image_path):
        """Dummy prediction - eğitilmiş model olmadan"""
        heartbeat(f"İşleniyor: {Path(image_path).name}")

        # Preprocessing yap
        preprocessed = preprocessor.preprocess(image_path, apply_normalization=False)

        # Random signal üret
        num_leads = 12
        signal_length = 5000

        # Biraz daha gerçekçi görünmesi için sinüzoidal bileşenler
        t = np.linspace(0, 10, signal_length)
        dummy_signals = np.zeros((num_leads, signal_length))

        for i in range(num_leads):
            freq1 = 1.0 + i * 0.1
            freq2 = 10.0 + i * 0.5
            dummy_signals[i] = (
                0.8 * np.sin(2 * np.pi * freq1 * t) +
                0.2 * np.sin(2 * np.pi * freq2 * t) +
                0.1 * np.random.randn(signal_length)
            )

        return dummy_signals

else:
    log("✓ Gerçek model kullanılıyor")
    USE_REAL_MODEL = True

    try:
        heartbeat("Model yükleniyor (bu biraz zaman alabilir)...")
        # Pipeline oluştur
        pipeline = ECGInferencePipeline(
            model_path=model_path,
            config=config,
            device=device
        )
        log("✅ Model başarıyla yüklendi!", "SUCCESS")

        def predict_image(image_path):
            """Gerçek model ile prediction"""
            heartbeat(f"Predicting: {Path(image_path).name}")
            return pipeline.predict(
                image_path,
                correct_rotation=True,
                threshold=0.5,
                return_dict=False
            )

    except Exception as e:
        log(f"❌ Model yükleme hatası: {e}", "ERROR")
        log("Dummy mode'a geçiliyor...", "WARNING")
        USE_REAL_MODEL = False

        preprocessor = ECGImagePreprocessor()

        def predict_image(image_path):
            heartbeat(f"Processing (dummy): {Path(image_path).name}")
            preprocessed = preprocessor.preprocess(image_path, apply_normalization=False)
            return np.random.randn(12, 5000) * 0.5

print()


# ============================================================================
# STEP 5: Test Prediction (Hızlı Kontrol)
# ============================================================================
log("STEP 5: Test prediction", "START")
print("-" * 80)

heartbeat("İlk görsel üzerinde test prediction yapılıyor...")

try:
    first_record_id = list(test_images_dict.keys())[0]
    first_image_path = test_images_dict[first_record_id]

    test_signal = predict_image(first_image_path)
    log(f"✓ Prediction tamamlandı (Record: {first_record_id})")
    log(f"  Shape: {test_signal.shape}")
    log(f"  Range: [{test_signal.min():.3f}, {test_signal.max():.3f}] mV")
    log(f"  Mean: {test_signal.mean():.3f} mV")

    # Görselleştir
    heartbeat("Sinyal görselleştiriliyor...")
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()

    lead_names = config.data.lead_names
    time = np.arange(1000) / 500

    for i, lead_name in enumerate(lead_names):
        ax = axes[i]
        ax.plot(time, test_signal[i, :1000], 'b-', linewidth=0.8)
        ax.set_title(f'Lead {lead_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Test Prediction - {first_record_id} (İlk 2 saniye)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/kaggle/working/test_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("✓ Görsel kaydedildi: test_prediction.png")

except Exception as e:
    log(f"❌ Test prediction hatası: {e}", "ERROR")

print()


# ============================================================================
# STEP 6: Batch Prediction - Timeout-Safe Version
# ============================================================================
log("STEP 6: Batch prediction (Timeout-Safe)", "START")
print("-" * 80)

predictions = {}

log(f"Toplam {len(record_ids)} record işlenecek")
log(f"  • Görseli bulunan: {len(test_images_dict)}")
log(f"  • Görseli bulunamayan: {len(record_ids) - len(test_images_dict)}")

# Batch boyutu (her N görselde bir checkpoint)
CHECKPOINT_INTERVAL = 10
HEARTBEAT_INTERVAL = 5

success_count = 0
error_count = 0
missing_count = 0

# Progress bar ile işle - TÜM record_id'ler için
pbar = tqdm(record_ids, desc="Processing records", unit="record")

for idx, record_id in enumerate(pbar, 1):
    try:
        # Heartbeat her N kayıtta
        if idx % HEARTBEAT_INTERVAL == 0:
            heartbeat(f"İşlenen: {idx}/{len(record_ids)} ({success_count} başarılı, {missing_count} eksik, {error_count} hatalı)")

        # Eğer görsel varsa, gerçek prediction yap
        if record_id in test_images_dict:
            image_path = test_images_dict[record_id]
            signals = predict_image(image_path)
            predictions[record_id] = signals
            success_count += 1
        else:
            # Görsel yoksa, sıfır değerli signal oluştur (uyarı ver)
            if missing_count == 0:
                log(f"⚠️ Görseli bulunamayan kayıtlar için sıfır değerli signal oluşturuluyor", "WARNING")
            signals = np.zeros((12, 5000))
            predictions[record_id] = signals
            missing_count += 1

        # Progress bar güncelle
        pbar.set_postfix({'success': success_count, 'missing': missing_count, 'errors': error_count})

        # Checkpoint kaydet
        if idx % CHECKPOINT_INTERVAL == 0:
            log(f"💾 Checkpoint: {idx}/{len(record_ids)} işlendi")
            # İsteğe bağlı: ara sonuçları kaydet
            checkpoint_file = f'/kaggle/working/checkpoint_{idx}.txt'
            with open(checkpoint_file, 'w') as f:
                f.write(f"Processed: {idx}\nSuccess: {success_count}\nMissing: {missing_count}\nErrors: {error_count}")

    except Exception as e:
        error_count += 1
        if error_count <= 5:
            log(f"❌ Hata (record: {record_id}): {e}", "ERROR")
        # Hata durumunda da sıfır değerli signal ekle
        predictions[record_id] = np.zeros((12, 5000))
        pbar.set_postfix({'success': success_count, 'missing': missing_count, 'errors': error_count})

pbar.close()

log(f"✅ Batch processing tamamlandı!", "SUCCESS")
log(f"   Başarılı: {success_count}/{len(record_ids)}")
if missing_count > 0:
    log(f"   Görseli yok: {missing_count}/{len(record_ids)}", "WARNING")
if error_count > 0:
    log(f"   Hatalı: {error_count}/{len(record_ids)}", "WARNING")

# İstatistikler
if len(predictions) > 0:
    heartbeat("İstatistikler hesaplanıyor...")
    all_signals = np.stack(list(predictions.values()))
    log(f"\n📊 Prediction İstatistikleri:")
    log(f"   Shape: {all_signals.shape}")
    log(f"   Min: {all_signals.min():.3f} mV")
    log(f"   Max: {all_signals.max():.3f} mV")
    log(f"   Mean: {all_signals.mean():.3f} mV")
    log(f"   Std: {all_signals.std():.3f} mV")

print()


# ============================================================================
# STEP 7: Submission File Oluştur
# ============================================================================
log("STEP 7: Kaggle submission dosyası oluşturma", "START")
print("-" * 80)

lead_names = config.data.lead_names

heartbeat("Submission formatı hazırlanıyor...")
rows = []

# Progress bar ile submission oluştur
# Format: id = "{record_id}_{time_idx}_{lead_name}", value = signal_value
total_rows = len(predictions) * len(lead_names) * 5000
log(f"Toplam {total_rows:,} satır oluşturulacak")
log(f"Format: id = {{record_id}}_{{time}}_{{lead}}, value = signal_value")

row_count = 0
for record_id, signals in predictions.items():
    heartbeat(f"Submission oluşturuluyor: {record_id}")

    for lead_idx, lead_name in enumerate(lead_names):
        for time_idx in range(signals.shape[1]):
            # ID formatı: {record_id}_{time_idx}_{lead_name}
            row_id = f"{record_id}_{time_idx}_{lead_name}"
            rows.append({
                'id': row_id,
                'value': float(signals[lead_idx, time_idx])
            })

            row_count += 1
            # Her 100k satırda heartbeat
            if row_count % 100000 == 0:
                heartbeat(f"Oluşturulan satır: {row_count:,}/{total_rows:,}")

heartbeat("DataFrame oluşturuluyor...")
submission_df = pd.DataFrame(rows)

# Kaydet
heartbeat("CSV dosyası kaydediliyor...")
submission_path = '/kaggle/working/submission.csv'
submission_df.to_csv(submission_path, index=False)

log(f"✅ Submission dosyası oluşturuldu!", "SUCCESS")
log(f"   Path: {submission_path}")
log(f"   Toplam satır: {len(submission_df):,}")
log(f"   Toplam record: {len(predictions)}")
log(f"   Dosya boyutu: {os.path.getsize(submission_path) / (1024*1024):.2f} MB")

# Önizleme
log(f"\n📋 Submission Önizlemesi (ilk 20 satır):")
print(submission_df.head(20).to_string())

print()


# ============================================================================
# STEP 8: Submission Validation
# ============================================================================
log("STEP 8: Submission validation", "START")
print("-" * 80)

heartbeat("Submission dosyası kontrol ediliyor...")

# Boyut kontrolü
log(f"✓ Toplam satır: {len(submission_df):,}")
log(f"✓ Kolonlar: {list(submission_df.columns)}")

# Kolon kontrolü
expected_columns = ['id', 'value']
if list(submission_df.columns) == expected_columns:
    log(f"✓ Kolonlar doğru: {expected_columns}")
else:
    log(f"⚠️ Kolon uyuşmazlığı! Beklenen: {expected_columns}, Mevcut: {list(submission_df.columns)}", "WARNING")

# Eksik değer kontrolü
missing = submission_df.isnull().sum().sum()
if missing > 0:
    log(f"⚠️ {missing} eksik değer bulundu!", "WARNING")
else:
    log(f"✓ Eksik değer yok")

# ID formatı kontrolü (örnek kontrol)
sample_id = submission_df['id'].iloc[0]
if '_' in sample_id:
    log(f"✓ ID formatı doğru (örnek: {sample_id})")
else:
    log(f"⚠️ ID formatı hatalı (örnek: {sample_id})", "WARNING")

# Record sayısı (id'lerden parse et)
unique_records = submission_df['id'].str.split('_').str[0].nunique()
log(f"✓ Unique record sayısı: {unique_records}")

# Değer aralığı kontrolü
val_min = submission_df['value'].min()
val_max = submission_df['value'].max()
val_mean = submission_df['value'].mean()
log(f"✓ Değer aralığı: [{val_min:.3f}, {val_max:.3f}] mV")
log(f"✓ Ortalama değer: {val_mean:.3f} mV")

# Dosya boyutu
file_size_mb = os.path.getsize(submission_path) / (1024 * 1024)
log(f"✓ Dosya boyutu: {file_size_mb:.2f} MB")

# NaN/Inf kontrolü
has_inf = np.isinf(submission_df['value']).any()
if has_inf:
    log("⚠️ Infinity değerleri tespit edildi!", "WARNING")
else:
    log("✓ Infinity değeri yok")

print("\n" + "=" * 80)

if missing == 0 and list(submission_df.columns) == expected_columns and not has_inf:
    log("✅✅✅ SUBMISSION HAZIR! SUBMIT EDEBİLİRSİNİZ! ✅✅✅", "SUCCESS")
else:
    log("⚠️ Submission'da bazı sorunlar var, lütfen kontrol edin", "WARNING")

print("=" * 80)
print()


# ============================================================================
# STEP 9: Görselleştirme
# ============================================================================
log("STEP 9: Sonuç görselleştirme", "START")
print("-" * 80)

if len(predictions) > 0:
    # Rastgele bir record seç
    import random
    random_record = random.choice(list(predictions.keys()))
    signals = predictions[random_record]

    heartbeat(f"Görselleştirilen record: {random_record}")

    # Tüm 12 lead'i görselleştir
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()

    for i, lead_name in enumerate(lead_names):
        heartbeat(f"Lead {lead_name} çiziliyor...")
        ax = axes[i]

        # Tüm sinyali çiz
        time = np.arange(signals.shape[1]) / 500
        ax.plot(time, signals[i], 'b-', linewidth=0.5)

        ax.set_title(f'Lead {lead_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Amplitude (mV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 10])

        # İstatistikler
        stats_text = f'Min: {signals[i].min():.2f}\nMax: {signals[i].max():.2f}\nMean: {signals[i].mean():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'12-Lead ECG Signal - Record: {random_record}',
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    heartbeat("Görsel kaydediliyor...")
    plt.savefig('/kaggle/working/ecg_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("✓ Görsel kaydedildi: ecg_visualization.png")

    # Per-lead istatistikler
    log("\n📊 Lead İstatistikleri:")
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
log("=" * 80)
log("🎉 PIPELINE TAMAMLANDI!", "SUCCESS")
log("=" * 80)

log(f"\n📊 ÖZET:")
log(f"   • Toplam record sayısı: {len(record_ids)}")
log(f"   • Gerçek görsellerden tahmin: {success_count}")
if missing_count > 0:
    log(f"   • ⚠️ Görseli olmayan (sıfır değer): {missing_count}")
if error_count > 0:
    log(f"   • ⚠️ Hatalı (sıfır değer): {error_count}")
log(f"   • Submission satır sayısı: {len(submission_df):,}")
log(f"   • Model tipi: {'GERÇEK MODEL' if USE_REAL_MODEL else 'DUMMY MODEL (Test)'}")
log(f"   • Submission dosyası: {submission_path}")
log(f"   • Dosya boyutu: {file_size_mb:.2f} MB")

log(f"\n📁 OLUŞTURULAN DOSYALAR:")
output_files = [
    '/kaggle/working/submission.csv',
    '/kaggle/working/ecg_visualization.png',
    '/kaggle/working/test_prediction.png',
    '/kaggle/working/sample_ecg_image.png',
]

for file_path in output_files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024
        log(f"   ✓ {file_path} ({size:.1f} KB)")

log(f"\n🚀 SONRAKI ADIMLAR:")
if not USE_REAL_MODEL:
    log("   1. ⚠️ DUMMY MODEL KULLANILDI! Gerçek sonuçlar için:")
    log("      - Model eğitin: scripts/train.py")
    log("      - Eğitilmiş modeli Kaggle'a dataset olarak yükleyin")
    log("      - Bu scripti tekrar çalıştırın")
    log("")

if missing_count > 0:
    log("   ⚠️ UYARI: Bazı kayıtların görselleri bulunamadı!")
    log("      - Bu kayıtlar için sıfır değerli signal kullanıldı")
    log("      - Gerçek yarışmada tüm görsellerin olduğundan emin olun")
    log("")

log("   • submission.csv dosyasını indirin")
log("   • Kaggle Competition sayfasına gidin")
log("   • 'Submit Predictions' butonuna tıklayın")
log("   • submission.csv dosyasını yükleyin")
log("   • Sonuçları bekleyin!")

log("\n" + "=" * 80)
log("✅ Script başarıyla tamamlandı!", "SUCCESS")
log(f"⏱️ Toplam süre: {time_module.time() - START_TIME:.2f} saniye")
log("=" * 80)
