# Kaggle Notebook Instructions

This directory contains files for running the ECG digitization pipeline on Kaggle.

## Usage on Kaggle

1. Create a new Kaggle notebook
2. Add this repository as a dataset or clone it:

```python
!git clone https://github.com/YOUR_USERNAME/PhysioNet-ECG-Image-Digitization-Challenge-2024.git
%cd PhysioNet-ECG-Image-Digitization-Challenge-2024
```

3. Install dependencies:

```python
!pip install -r requirements.txt
```

4. Run inference on test data:

```python
import sys
sys.path.append('/kaggle/working/PhysioNet-ECG-Image-Digitization-Challenge-2024')

from src.inference import ECGInferencePipeline

# Load model
pipeline = ECGInferencePipeline(
    model_path='/kaggle/input/your-model/best_model.pth',
    device='cuda'
)

# Run predictions
test_dir = '/kaggle/input/physionet-ecg-image-digitization/test'
# ... (add your prediction code here)
```

5. Generate submission:

```python
!python scripts/generate_submission.py \
    --predictions_dir predictions \
    --output_file submission.csv
```

## Model Weights

You will need to upload your trained model weights to Kaggle as a dataset.

## Expected Format

The submission file should follow the PhysioNet Challenge format with columns:
- `record_id`: Image identifier
- `lead`: Lead name (I, II, III, aVR, aVL, aVF, V1-V6)
- `time`: Time index
- `value`: Signal value in mV

Or in wide format with one row per record and columns for each lead and time point.
