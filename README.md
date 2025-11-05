# PhysioNet ECG Image Digitization Challenge 2024

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive solution for the PhysioNet ECG Image Digitization Challenge - converting scanned or photographed ECG images into time-series signals with high fidelity.

## Overview

This project implements a state-of-the-art pipeline for digitizing ECG images using a combination of classical image processing techniques and deep learning:

- **Hough Transform** for image rotation correction
- **U-Net segmentation** with pretrained encoders for lead detection
- **Advanced vectorization** with signal alignment
- **SNR optimization** for maximum accuracy

### Target Metric

- **Signal-to-Noise Ratio (SNR)**: Higher is better
- **Goal**: SNR > 10.0 (winning solutions achieved 12-17)

## Project Structure

```
physionet-ecg-digitization/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── setup.py                      # Installation file
├── .gitignore                   # Git ignore file
├── data/
│   ├── raw/                     # Raw data (not in git)
│   ├── processed/               # Processed data
│   └── synthetic/               # Synthetic training data
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration parameters
│   ├── data_preprocessing.py    # Data preprocessing
│   ├── augmentation.py          # Data augmentation
│   ├── rotation.py              # Hough transform rotation
│   ├── segmentation_model.py    # U-Net/nnU-Net segmentation
│   ├── vectorization.py         # Signal vectorization
│   ├── signal_alignment.py      # Signal alignment
│   ├── evaluation.py            # SNR calculation
│   └── inference.py             # Inference pipeline
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Preprocessing experiments
│   ├── 03_model_training.ipynb  # Model training
│   └── 04_evaluation.ipynb      # Model evaluation
├── models/
│   ├── pretrained/              # Pretrained weights
│   └── checkpoints/             # Training checkpoints
├── scripts/
│   ├── train.py                 # Training script
│   ├── predict.py               # Prediction script
│   └── generate_submission.py   # Kaggle submission generator
└── kaggle/
    ├── submission.csv           # Kaggle submission file
    └── kaggle_notebook.ipynb    # Kaggle notebook
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/physionet-ecg-digitization.git
cd physionet-ecg-digitization
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Technical Approach

### 1. Image Preprocessing

- Load and normalize images (2200x1700 pixels)
- Convert to grayscale
- Noise reduction (Gaussian blur, median filter)
- Contrast enhancement (CLAHE)

### 2. Rotation Correction

Uses **Hough Transform** to:
- Detect grid lines
- Calculate skew angle
- Correct image orientation

### 3. Segmentation Model

**U-Net architecture** with:
- Pretrained encoder (EfficientNet-B4/ResNet)
- Binary mask output for each lead
- Loss: Dice Loss + Binary Cross Entropy
- ImageNet pretrained weights

### 4. Vectorization

Extract signals from masks:
- Average y-position of non-zero pixels per column
- Scale using grid size and resolution
- Determine start time based on lead position
- Generate 12-lead ECG signal

### 5. Signal Alignment

Cross-correlation based alignment:
- ±0.5 second horizontal shift
- ±1 mV vertical shift
- Optimize for maximum cross-correlation

### 6. Evaluation

SNR (Signal-to-Noise Ratio) calculation:
```python
SNR = 10 * log10(sum(true_signal²) / sum(noise²))
```

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml --data_dir data/raw --output_dir models/checkpoints
```

### Prediction

```bash
python scripts/predict.py --model_path models/checkpoints/best_model.pth --input_dir data/test --output_dir predictions
```

### Generate Kaggle Submission

```bash
python scripts/generate_submission.py --predictions_dir predictions --output kaggle/submission.csv
```

## Data Augmentation

The training pipeline includes:
- Rotation (-5° to +5°)
- Gaussian noise
- Brightness/contrast variations
- Grid color variations
- Paper texture simulation

## Model Configuration

Default hyperparameters:
```python
CONFIG = {
    'image_size': (2200, 1700),
    'batch_size': 4,
    'learning_rate': 1e-4,
    'epochs': 100,
    'encoder': 'efficientnet-b4',
    'optimizer': 'AdamW',
    'weight_decay': 1e-5,
}
```

## Kaggle Integration

### Using on Kaggle Notebooks

1. Clone this repository in a Kaggle notebook:
```python
!git clone https://github.com/YOUR_USERNAME/physionet-ecg-digitization.git
%cd physionet-ecg-digitization
!pip install -r requirements.txt
```

2. Run inference:
```python
from src.inference import predict_test_set
predictions = predict_test_set('/kaggle/input/physionet-ecg-image-digitization/')
```

3. Generate submission:
```python
from scripts.generate_submission import create_submission
create_submission(predictions, 'submission.csv')
```

## Key Success Factors

1. **Data Diversity**: Use multiple sources (scans, photos, various qualities)
2. **Robust Preprocessing**: Hough transform for rotation correction is critical
3. **Segmentation Quality**: Use powerful segmentation models like nnU-Net
4. **Alignment**: Signal alignment is crucial for SNR
5. **Ensemble**: Ensemble models from different folds

## Results

| Model | CV SNR | Public LB | Private LB |
|-------|--------|-----------|------------|
| Baseline | TBD | TBD | TBD |
| + Rotation | TBD | TBD | TBD |
| + Segmentation | TBD | TBD | TBD |
| Final Ensemble | TBD | TBD | TBD |

## References

- [PhysioNet Challenge 2024](https://physionet.org/content/challenge-2024/)
- [Winning Solution Paper](https://arxiv.org/abs/2409.15975)
- [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit)
- [PTB-XL Dataset](https://physionet.org/content/ptb-xl/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PhysioNet Challenge organizers
- SignalSavants team for their winning approach
- PyTorch and segmentation-models-pytorch communities

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{physionet-ecg-digitization-2024,
  author = {Your Name},
  title = {PhysioNet ECG Image Digitization Challenge 2024 Solution},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/physionet-ecg-digitization}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

**Note**: This is a solution for the PhysioNet ECG Image Digitization Challenge 2024. The code is provided for educational and research purposes.
