"""
Prediction script for ECG image digitization.

This script runs inference on ECG images and saves the extracted signals.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import ECGInferencePipeline, EnsembleInferencePipeline
from src.config import get_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict ECG signals from images")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing ECG images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model or comma-separated paths for ensemble"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for mask binarization"
    )
    parser.add_argument(
        "--correct_rotation",
        action="store_true",
        default=True,
        help="Apply rotation correction"
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Use ensemble of models (provide comma-separated paths)"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="csv",
        choices=["csv", "json", "npy"],
        help="Output format for predictions"
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default="png,jpg,jpeg",
        help="Comma-separated list of image extensions to process"
    )

    return parser.parse_args()


def get_image_files(input_dir, extensions):
    """
    Get all image files from directory.

    Args:
        input_dir: Input directory
        extensions: List of file extensions

    Returns:
        List of image file paths
    """
    input_dir = Path(input_dir)
    image_files = []

    for ext in extensions:
        image_files.extend(input_dir.glob(f"*.{ext}"))
        image_files.extend(input_dir.glob(f"*.{ext.upper()}"))

    return sorted(image_files)


def save_predictions(predictions, output_dir, output_format, config):
    """
    Save predictions to files.

    Args:
        predictions: Dictionary mapping file names to signals
        output_dir: Output directory
        output_format: Output format ('csv', 'json', 'npy')
        config: Configuration object
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lead_names = config.data.lead_names

    if output_format == "csv":
        # Save as CSV (one row per image, columns for each lead and time point)
        for filename, signals in predictions.items():
            # Flatten signals
            data = {}
            for i, lead_name in enumerate(lead_names):
                if i < signals.shape[0]:
                    for t in range(signals.shape[1]):
                        data[f"{lead_name}_t{t}"] = [signals[i, t]]

            df = pd.DataFrame(data)
            output_path = output_dir / f"{filename}.csv"
            df.to_csv(output_path, index=False)

    elif output_format == "json":
        # Save as JSON
        for filename, signals in predictions.items():
            data = {}
            for i, lead_name in enumerate(lead_names):
                if i < signals.shape[0]:
                    data[lead_name] = signals[i].tolist()

            output_path = output_dir / f"{filename}.json"
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

    elif output_format == "npy":
        # Save as numpy arrays
        for filename, signals in predictions.items():
            output_path = output_dir / f"{filename}.npy"
            np.save(output_path, signals)

    print(f"Saved predictions to {output_dir}")


def main(args):
    """
    Main prediction function.

    Args:
        args: Command line arguments
    """
    # Setup
    config = get_config()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("ECG Image Digitization - Prediction")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Output format: {args.output_format}")
    print("=" * 60)

    # Get image files
    extensions = args.image_extensions.split(',')
    image_files = get_image_files(input_dir, extensions)
    print(f"\nFound {len(image_files)} images to process")

    if len(image_files) == 0:
        print("No images found. Exiting.")
        return

    # Create inference pipeline
    print("\nLoading model(s)...")
    if args.ensemble:
        model_paths = [Path(p.strip()) for p in args.model_path.split(',')]
        pipeline = EnsembleInferencePipeline(
            model_paths=model_paths,
            config=config,
            device=args.device,
            ensemble_method="average"
        )
    else:
        pipeline = ECGInferencePipeline(
            model_path=args.model_path,
            config=config,
            device=args.device
        )

    # Run predictions
    print("\nRunning predictions...")
    predictions = {}

    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Predict
            signals = pipeline.predict(
                image_path,
                correct_rotation=args.correct_rotation,
                threshold=args.threshold,
                return_dict=False
            )

            # Store predictions
            predictions[image_path.stem] = signals

        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            continue

    print(f"\nSuccessfully processed {len(predictions)}/{len(image_files)} images")

    # Save predictions
    if len(predictions) > 0:
        print("\nSaving predictions...")
        save_predictions(predictions, output_dir, args.output_format, config)

        # Print summary
        print("\n" + "=" * 60)
        print("Prediction Summary")
        print("=" * 60)

        all_signals = np.stack(list(predictions.values()))
        print(f"Number of predictions: {len(predictions)}")
        print(f"Signal shape: {all_signals.shape}")
        print(f"Signal range: [{all_signals.min():.3f}, {all_signals.max():.3f}]")
        print(f"Signal mean: {all_signals.mean():.3f}")
        print(f"Signal std: {all_signals.std():.3f}")
        print("=" * 60)

    print("\nPrediction completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
