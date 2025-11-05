"""
Generate Kaggle submission file from predictions.

This script creates a submission file in the format required by
the PhysioNet ECG Image Digitization Challenge.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Kaggle submission from predictions"
    )

    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Path to directory containing prediction files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="kaggle/submission.csv",
        help="Path to output submission file"
    )
    parser.add_argument(
        "--prediction_format",
        type=str,
        default="npy",
        choices=["npy", "json", "csv"],
        help="Format of prediction files"
    )

    return parser.parse_args()


def load_predictions(predictions_dir, prediction_format):
    """
    Load all prediction files.

    Args:
        predictions_dir: Directory containing predictions
        prediction_format: Format of prediction files

    Returns:
        Dictionary mapping record IDs to signals
    """
    predictions_dir = Path(predictions_dir)
    predictions = {}

    if prediction_format == "npy":
        pred_files = list(predictions_dir.glob("*.npy"))
        for pred_file in pred_files:
            record_id = pred_file.stem
            signals = np.load(pred_file)
            predictions[record_id] = signals

    elif prediction_format == "json":
        import json
        pred_files = list(predictions_dir.glob("*.json"))
        for pred_file in pred_files:
            record_id = pred_file.stem
            with open(pred_file, 'r') as f:
                data = json.load(f)
            # Convert to array
            config = get_config()
            signals = np.zeros((12, config.data.signal_length))
            for i, lead_name in enumerate(config.data.lead_names):
                if lead_name in data:
                    signals[i] = np.array(data[lead_name])
            predictions[record_id] = signals

    elif prediction_format == "csv":
        pred_files = list(predictions_dir.glob("*.csv"))
        for pred_file in pred_files:
            record_id = pred_file.stem
            df = pd.read_csv(pred_file)
            # Parse CSV format (implementation depends on how it was saved)
            # This is a placeholder
            print(f"Warning: CSV format loading not fully implemented for {pred_file}")

    return predictions


def create_submission(predictions, output_file):
    """
    Create submission file from predictions.

    Args:
        predictions: Dictionary mapping record IDs to signals
        output_file: Path to output file
    """
    config = get_config()
    lead_names = config.data.lead_names

    # Prepare submission data
    rows = []

    for record_id, signals in tqdm(predictions.items(), desc="Creating submission"):
        # For each record, create rows for each lead and time point
        # Format: record_id, lead_name, time_index, value

        for lead_idx, lead_name in enumerate(lead_names):
            if lead_idx < signals.shape[0]:
                for time_idx in range(signals.shape[1]):
                    rows.append({
                        'record_id': record_id,
                        'lead': lead_name,
                        'time': time_idx,
                        'value': signals[lead_idx, time_idx]
                    })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nSubmission file created: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Number of records: {len(predictions)}")


def create_submission_alternative_format(predictions, output_file):
    """
    Create submission in alternative format (one row per record).

    Args:
        predictions: Dictionary mapping record IDs to signals
        output_file: Path to output file
    """
    config = get_config()
    lead_names = config.data.lead_names

    # Create columns for each lead and time point
    columns = ['record_id']
    for lead_name in lead_names:
        for t in range(config.data.signal_length):
            columns.append(f"{lead_name}_t{t}")

    rows = []

    for record_id, signals in tqdm(predictions.items(), desc="Creating submission"):
        row = {'record_id': record_id}

        for lead_idx, lead_name in enumerate(lead_names):
            if lead_idx < signals.shape[0]:
                for t in range(signals.shape[1]):
                    row[f"{lead_name}_t{t}"] = signals[lead_idx, t]

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save to CSV
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\nSubmission file created: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Number of records: {len(predictions)}")


def validate_submission(submission_file):
    """
    Validate submission file.

    Args:
        submission_file: Path to submission file

    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(submission_file)

        print("\nValidating submission file...")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")

        # Check for required columns
        if 'record_id' not in df.columns:
            print("Error: 'record_id' column not found")
            return False

        # Check for NaN values
        if df.isnull().any().any():
            print("Warning: Submission contains NaN values")

        print("Validation passed!")
        return True

    except Exception as e:
        print(f"Error validating submission: {e}")
        return False


def main(args):
    """
    Main function.

    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("Generating Kaggle Submission")
    print("=" * 60)
    print(f"Predictions directory: {args.predictions_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Prediction format: {args.prediction_format}")
    print("=" * 60)

    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(args.predictions_dir, args.prediction_format)
    print(f"Loaded {len(predictions)} predictions")

    if len(predictions) == 0:
        print("No predictions found. Exiting.")
        return

    # Create submission
    print("\nCreating submission file...")
    create_submission(predictions, args.output_file)

    # Validate
    validate_submission(args.output_file)

    print("\n" + "=" * 60)
    print("Submission generation completed!")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    main(args)
