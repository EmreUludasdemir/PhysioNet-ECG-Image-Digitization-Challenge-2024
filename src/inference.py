"""
Inference pipeline for ECG image digitization.

This module provides end-to-end inference from ECG images to time-series signals.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import cv2

from .config import get_config
from .data_preprocessing import ECGImagePreprocessor
from .rotation import ECGRotationCorrector
from .segmentation_model import ECGSegmentationModel, create_model
from .vectorization import ECGVectorizer
from .evaluation import ECGEvaluator


class ECGInferencePipeline:
    """Complete inference pipeline for ECG digitization."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config=None,
        device: str = "cuda"
    ):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model weights
            config: Configuration object
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.config = config or get_config()
        self.device = device if torch.cuda.is_available() else "cpu"

        # Initialize components
        self.preprocessor = ECGImagePreprocessor(self.config)
        self.rotation_corrector = ECGRotationCorrector(self.config)
        self.vectorizer = ECGVectorizer(self.config)
        self.evaluator = ECGEvaluator(self.config)

        # Initialize model
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load weights if provided
        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: Union[str, Path]):
        """
        Load model weights.

        Args:
            model_path: Path to model weights
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Loaded model from {model_path}")

    def preprocess_image(
        self,
        image: Union[str, Path, np.ndarray],
        correct_rotation: bool = True
    ) -> np.ndarray:
        """
        Preprocess ECG image.

        Args:
            image: Input image (path or array)
            correct_rotation: Whether to apply rotation correction

        Returns:
            Preprocessed image
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = self.preprocessor.load_image(image)

        # Correct rotation
        if correct_rotation:
            image = self.rotation_corrector.correct_rotation(image)

        # Apply preprocessing
        preprocessed = self.preprocessor.preprocess(
            image,
            apply_normalization=False  # Will normalize before model
        )

        return preprocessed

    def predict_masks(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict segmentation masks using the model.

        Args:
            image: Preprocessed image
            threshold: Threshold for binarization

        Returns:
            Binary masks (num_leads, H, W)
        """
        # Prepare image for model
        if len(image.shape) == 2:
            # Grayscale -> RGB
            image_rgb = np.stack([image] * 3, axis=-1)
        else:
            image_rgb = image

        # Normalize
        image_normalized = self.preprocessor.normalize_image(image_rgb)

        # Convert to tensor (C, H, W)
        if len(image_normalized.shape) == 2:
            image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0).float().to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()

        # Convert to numpy (num_leads, H, W)
        masks = masks.squeeze(0).cpu().numpy()

        return masks

    def extract_signals(
        self,
        masks: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract time-series signals from masks.

        Args:
            masks: Binary masks (num_leads, H, W)

        Returns:
            Dictionary mapping lead names to signals
        """
        signals = self.vectorizer.extract_lead_signals(masks)
        return signals

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        correct_rotation: bool = True,
        threshold: float = 0.5,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Complete inference pipeline: image -> signals.

        Args:
            image: Input image (path or array)
            correct_rotation: Whether to apply rotation correction
            threshold: Threshold for mask binarization
            return_dict: If True, return dict; else return array

        Returns:
            Extracted ECG signals
        """
        # Preprocess
        preprocessed = self.preprocess_image(image, correct_rotation)

        # Predict masks
        masks = self.predict_masks(preprocessed, threshold)

        # Extract signals
        signals = self.extract_signals(masks)

        if return_dict:
            return signals
        else:
            return self.vectorizer.signals_to_array(signals)

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        correct_rotation: bool = True,
        threshold: float = 0.5,
        batch_size: int = 4,
        return_dict: bool = False
    ) -> List[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """
        Predict on a batch of images.

        Args:
            images: List of images
            correct_rotation: Whether to apply rotation correction
            threshold: Threshold for mask binarization
            batch_size: Batch size for model inference
            return_dict: If True, return dicts; else return arrays

        Returns:
            List of extracted signals
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            for image in batch:
                result = self.predict(
                    image,
                    correct_rotation=correct_rotation,
                    threshold=threshold,
                    return_dict=return_dict
                )
                results.append(result)

        return results

    def predict_and_evaluate(
        self,
        image: Union[str, Path, np.ndarray],
        ground_truth: np.ndarray,
        correct_rotation: bool = True,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Predict and evaluate against ground truth.

        Args:
            image: Input image
            ground_truth: Ground truth signals (num_leads, signal_length)
            correct_rotation: Whether to apply rotation correction
            threshold: Threshold for mask binarization

        Returns:
            Predicted signals and evaluation results
        """
        # Predict
        predicted = self.predict(
            image,
            correct_rotation=correct_rotation,
            threshold=threshold,
            return_dict=False
        )

        # Evaluate
        results = self.evaluator.evaluate(predicted, ground_truth)

        return predicted, results


class EnsembleInferencePipeline:
    """Ensemble inference using multiple models."""

    def __init__(
        self,
        model_paths: List[Union[str, Path]],
        config=None,
        device: str = "cuda",
        ensemble_method: str = "average"
    ):
        """
        Initialize ensemble inference pipeline.

        Args:
            model_paths: List of paths to model weights
            config: Configuration object
            device: Device to run inference on
            ensemble_method: How to combine predictions ('average', 'voting')
        """
        self.config = config or get_config()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ensemble_method = ensemble_method

        # Initialize pipelines
        self.pipelines = []
        for model_path in model_paths:
            pipeline = ECGInferencePipeline(
                model_path=model_path,
                config=config,
                device=device
            )
            self.pipelines.append(pipeline)

        print(f"Initialized ensemble with {len(self.pipelines)} models")

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        correct_rotation: bool = True,
        threshold: float = 0.5,
        return_dict: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Ensemble prediction.

        Args:
            image: Input image
            correct_rotation: Whether to apply rotation correction
            threshold: Threshold for mask binarization
            return_dict: If True, return dict; else return array

        Returns:
            Ensemble predicted signals
        """
        # Get predictions from all models
        predictions = []
        for pipeline in self.pipelines:
            pred = pipeline.predict(
                image,
                correct_rotation=correct_rotation,
                threshold=threshold,
                return_dict=False
            )
            predictions.append(pred)

        # Stack predictions
        predictions = np.stack(predictions)  # (num_models, num_leads, signal_length)

        # Ensemble
        if self.ensemble_method == "average":
            ensemble_pred = np.mean(predictions, axis=0)
        elif self.ensemble_method == "median":
            ensemble_pred = np.median(predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

        if return_dict:
            # Convert to dictionary
            lead_names = self.config.data.lead_names
            signals = {}
            for i, lead_name in enumerate(lead_names):
                if i < ensemble_pred.shape[0]:
                    signals[lead_name] = ensemble_pred[i]
            return signals
        else:
            return ensemble_pred


def predict_from_image(
    image_path: Union[str, Path],
    model_path: Union[str, Path],
    config=None,
    device: str = "cuda"
) -> np.ndarray:
    """
    Convenience function for single image prediction.

    Args:
        image_path: Path to ECG image
        model_path: Path to trained model
        config: Configuration object
        device: Device to run on

    Returns:
        Predicted ECG signals (num_leads, signal_length)
    """
    pipeline = ECGInferencePipeline(model_path, config, device)
    return pipeline.predict(image_path, return_dict=False)


if __name__ == "__main__":
    # Test inference pipeline
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <model_path>")
        print("\nRunning in test mode with dummy data...")

        # Create dummy image
        dummy_image = np.ones((1024, 1024, 3), dtype=np.uint8) * 255

        # Create dummy model
        model = create_model()
        model_path = Path("test_model.pth")
        torch.save(model.state_dict(), model_path)

        # Test pipeline
        pipeline = ECGInferencePipeline(model_path=model_path, device="cpu")
        signals = pipeline.predict(dummy_image, return_dict=True)

        print(f"Predicted {len(signals)} lead signals:")
        for lead_name, signal in signals.items():
            print(f"  {lead_name}: shape={signal.shape}, "
                  f"range=[{signal.min():.3f}, {signal.max():.3f}]")

        # Clean up
        model_path.unlink()

    else:
        image_path = sys.argv[1]
        model_path = sys.argv[2]

        print(f"Running inference on {image_path}")
        pipeline = ECGInferencePipeline(model_path=model_path)
        signals = pipeline.predict(image_path, return_dict=True)

        print(f"\nPredicted {len(signals)} lead signals:")
        for lead_name, signal in signals.items():
            print(f"  {lead_name}: shape={signal.shape}, "
                  f"range=[{signal.min():.3f}, {signal.max():.3f}]")
