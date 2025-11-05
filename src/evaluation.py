"""
Evaluation module for ECG digitization.

This module provides metrics for evaluating the quality of ECG signal extraction,
with SNR (Signal-to-Noise Ratio) as the primary metric for the competition.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import get_config
from .signal_alignment import SignalAligner


class ECGEvaluator:
    """Evaluator for ECG signals."""

    def __init__(self, config=None):
        """
        Initialize the evaluator.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.eval_config = self.config.evaluation
        self.data_config = self.config.data

        # Initialize aligner for SNR calculation
        self.aligner = SignalAligner(config)

    def calculate_snr(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        align_first: bool = True
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) in dB.

        Formula: SNR = 10 * log10(sum(target^2) / sum((aligned_pred - target)^2))

        Args:
            predicted: Predicted signal
            target: Ground truth signal
            align_first: Whether to align signals before calculating SNR

        Returns:
            SNR in dB
        """
        # Align signals if requested
        if align_first:
            aligned_pred, _ = self.aligner.align_signals(predicted, target)
        else:
            aligned_pred = predicted

        # Calculate signal power
        signal_power = np.sum(target ** 2)

        # Calculate noise power
        noise = aligned_pred - target
        noise_power = np.sum(noise ** 2)

        # Avoid division by zero
        if noise_power < 1e-10:
            return 100.0  # Very high SNR if noise is negligible

        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power)

        return snr_db

    def calculate_mse(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        align_first: bool = True
    ) -> float:
        """
        Calculate Mean Squared Error (MSE).

        Args:
            predicted: Predicted signal
            target: Ground truth signal
            align_first: Whether to align signals first

        Returns:
            MSE value
        """
        if align_first:
            aligned_pred, _ = self.aligner.align_signals(predicted, target)
        else:
            aligned_pred = predicted

        mse = np.mean((aligned_pred - target) ** 2)

        return mse

    def calculate_mae(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        align_first: bool = True
    ) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Args:
            predicted: Predicted signal
            target: Ground truth signal
            align_first: Whether to align signals first

        Returns:
            MAE value
        """
        if align_first:
            aligned_pred, _ = self.aligner.align_signals(predicted, target)
        else:
            aligned_pred = predicted

        mae = np.mean(np.abs(aligned_pred - target))

        return mae

    def calculate_correlation(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        align_first: bool = True
    ) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            predicted: Predicted signal
            target: Ground truth signal
            align_first: Whether to align signals first

        Returns:
            Correlation coefficient
        """
        if align_first:
            aligned_pred, _ = self.aligner.align_signals(predicted, target)
        else:
            aligned_pred = predicted

        correlation = np.corrcoef(aligned_pred, target)[0, 1]

        return correlation

    def evaluate_single_lead(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        metrics: Optional[List[str]] = None,
        align_first: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a single lead signal.

        Args:
            predicted: Predicted signal
            target: Ground truth signal
            metrics: List of metrics to calculate
            align_first: Whether to align signals first

        Returns:
            Dictionary of metric values
        """
        metrics = metrics or self.eval_config.metrics

        results = {}

        for metric in metrics:
            if metric == 'snr':
                results['snr'] = self.calculate_snr(predicted, target, align_first)
            elif metric == 'mse':
                results['mse'] = self.calculate_mse(predicted, target, align_first)
            elif metric == 'mae':
                results['mae'] = self.calculate_mae(predicted, target, align_first)
            elif metric == 'correlation':
                results['correlation'] = self.calculate_correlation(predicted, target, align_first)
            else:
                print(f"Warning: Unknown metric '{metric}'")

        return results

    def evaluate_multi_lead(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        lead_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        align_first: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple lead signals.

        Args:
            predicted: Predicted signals (num_leads, signal_length)
            target: Ground truth signals (num_leads, signal_length)
            lead_names: Names of leads
            metrics: List of metrics to calculate
            align_first: Whether to align signals first

        Returns:
            Dictionary mapping lead names to metric dictionaries
        """
        lead_names = lead_names or self.data_config.lead_names
        num_leads = predicted.shape[0]

        results = {}

        for i in range(num_leads):
            lead_name = lead_names[i] if i < len(lead_names) else f"Lead_{i}"

            lead_results = self.evaluate_single_lead(
                predicted[i],
                target[i],
                metrics=metrics,
                align_first=align_first
            )

            results[lead_name] = lead_results

        return results

    def calculate_average_metrics(
        self,
        lead_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate average metrics across all leads.

        Args:
            lead_results: Dictionary of per-lead results

        Returns:
            Dictionary of averaged metrics
        """
        # Collect all metrics
        all_metrics = {}
        for lead_name, metrics in lead_results.items():
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        # Calculate averages
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[f"avg_{metric_name}"] = np.mean(values)
            avg_metrics[f"std_{metric_name}"] = np.std(values)
            avg_metrics[f"min_{metric_name}"] = np.min(values)
            avg_metrics[f"max_{metric_name}"] = np.max(values)

        return avg_metrics

    def evaluate(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
        lead_names: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        align_first: bool = True,
        return_per_lead: bool = True
    ) -> Dict[str, any]:
        """
        Complete evaluation pipeline.

        Args:
            predicted: Predicted signals (num_leads, signal_length)
            target: Ground truth signals (num_leads, signal_length)
            lead_names: Names of leads
            metrics: List of metrics to calculate
            align_first: Whether to align signals first
            return_per_lead: Whether to return per-lead results

        Returns:
            Dictionary containing evaluation results
        """
        # Evaluate per lead
        lead_results = self.evaluate_multi_lead(
            predicted,
            target,
            lead_names=lead_names,
            metrics=metrics,
            align_first=align_first
        )

        # Calculate averages
        avg_results = self.calculate_average_metrics(lead_results)

        # Combine results
        results = {
            'average': avg_results
        }

        if return_per_lead:
            results['per_lead'] = lead_results

        return results

    def print_results(
        self,
        results: Dict[str, any],
        show_per_lead: bool = False
    ):
        """
        Print evaluation results in a formatted way.

        Args:
            results: Results dictionary from evaluate()
            show_per_lead: Whether to show per-lead results
        """
        print("=" * 60)
        print("ECG DIGITIZATION EVALUATION RESULTS")
        print("=" * 60)

        # Print average metrics
        if 'average' in results:
            print("\nAverage Metrics:")
            print("-" * 60)
            avg_metrics = results['average']

            # Print SNR prominently if available
            if 'avg_snr' in avg_metrics:
                print(f"  SNR (Signal-to-Noise Ratio): {avg_metrics['avg_snr']:.3f} dB")
                print(f"    (std: {avg_metrics.get('std_snr', 0):.3f}, "
                      f"min: {avg_metrics.get('min_snr', 0):.3f}, "
                      f"max: {avg_metrics.get('max_snr', 0):.3f})")

            # Print other metrics
            for key, value in sorted(avg_metrics.items()):
                if not key.startswith('avg_') or key == 'avg_snr':
                    continue
                metric_name = key.replace('avg_', '').upper()
                std_key = f"std_{key.replace('avg_', '')}"
                min_key = f"min_{key.replace('avg_', '')}"
                max_key = f"max_{key.replace('avg_', '')}"

                print(f"  {metric_name}: {value:.4f}")
                if std_key in avg_metrics:
                    print(f"    (std: {avg_metrics[std_key]:.4f}, "
                          f"min: {avg_metrics.get(min_key, 0):.4f}, "
                          f"max: {avg_metrics.get(max_key, 0):.4f})")

        # Print per-lead results if requested
        if show_per_lead and 'per_lead' in results:
            print("\nPer-Lead Metrics:")
            print("-" * 60)
            for lead_name, metrics in results['per_lead'].items():
                print(f"\n  {lead_name}:")
                for metric_name, value in sorted(metrics.items()):
                    print(f"    {metric_name.upper()}: {value:.4f}")

        print("=" * 60)


def calculate_snr(
    predicted: np.ndarray,
    target: np.ndarray,
    config=None,
    align_first: bool = True
) -> float:
    """
    Convenience function to calculate SNR.

    Args:
        predicted: Predicted signal
        target: Ground truth signal
        config: Configuration object
        align_first: Whether to align signals first

    Returns:
        SNR in dB
    """
    evaluator = ECGEvaluator(config)
    return evaluator.calculate_snr(predicted, target, align_first)


def evaluate_signals(
    predicted: np.ndarray,
    target: np.ndarray,
    config=None,
    **kwargs
) -> Dict[str, any]:
    """
    Convenience function to evaluate signals.

    Args:
        predicted: Predicted signals
        target: Ground truth signals
        config: Configuration object
        **kwargs: Additional arguments for evaluate()

    Returns:
        Evaluation results
    """
    evaluator = ECGEvaluator(config)
    return evaluator.evaluate(predicted, target, **kwargs)


if __name__ == "__main__":
    # Test evaluation
    print("Testing ECG Evaluation...")

    # Create synthetic signals
    num_leads = 12
    signal_length = 5000

    # Create target signals
    t = np.linspace(0, 10, signal_length)
    target = np.zeros((num_leads, signal_length))

    for i in range(num_leads):
        target[i] = (
            np.sin(2 * np.pi * t * (i + 1) / 12) +
            0.5 * np.sin(4 * np.pi * t * (i + 1) / 12)
        )

    # Create predicted signals with some noise and offset
    predicted = target + 0.1 * np.random.randn(num_leads, signal_length)
    predicted = np.roll(predicted, 50, axis=1)  # Add time shift
    predicted += 0.2  # Add voltage offset

    # Evaluate
    evaluator = ECGEvaluator()
    results = evaluator.evaluate(predicted, target)

    # Print results
    evaluator.print_results(results, show_per_lead=True)
