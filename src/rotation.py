"""
Image rotation correction module using Hough Transform.

This module detects ECG grid lines and corrects image rotation/skew
to ensure accurate signal extraction.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path

from .config import get_config


class ECGRotationCorrector:
    """Rotation corrector for ECG images using Hough Transform."""

    def __init__(self, config=None):
        """
        Initialize the rotation corrector.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.rotation_config = self.config.rotation

    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges in image using Canny edge detector.

        Args:
            image: Input grayscale image
            low_threshold: Lower threshold for Canny
            high_threshold: Upper threshold for Canny

        Returns:
            Binary edge map
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(image, low_threshold, high_threshold)

        return edges

    def detect_lines_hough(
        self,
        edges: np.ndarray,
        threshold: Optional[int] = None,
        min_line_length: Optional[int] = None,
        max_line_gap: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Detect lines using Probabilistic Hough Transform.

        Args:
            edges: Binary edge map
            threshold: Minimum number of votes
            min_line_length: Minimum line length
            max_line_gap: Maximum gap between line segments

        Returns:
            Array of detected lines [(x1, y1, x2, y2), ...]
        """
        threshold = threshold or self.rotation_config.hough_threshold
        min_line_length = min_line_length or self.rotation_config.hough_min_line_length
        max_line_gap = max_line_gap or self.rotation_config.hough_max_line_gap

        # Detect lines
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        return lines

    def calculate_line_angle(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate angle of a line in degrees.

        Args:
            x1, y1: Start point
            x2, y2: End point

        Returns:
            Angle in degrees (0 = horizontal, 90 = vertical)
        """
        # Calculate angle using arctangent
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalize to [0, 180)
        if angle < 0:
            angle += 180

        return angle

    def filter_horizontal_lines(
        self,
        lines: np.ndarray,
        angle_threshold: Optional[float] = None
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Filter horizontal lines from detected lines.

        Args:
            lines: Array of detected lines
            angle_threshold: Maximum deviation from horizontal (degrees)

        Returns:
            List of horizontal lines with their angles
        """
        angle_threshold = angle_threshold or self.rotation_config.horizontal_line_threshold

        horizontal_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = self.calculate_line_angle(x1, y1, x2, y2)

            # Check if line is close to horizontal (0° or 180°)
            if angle <= angle_threshold or angle >= (180 - angle_threshold):
                horizontal_lines.append((x1, y1, x2, y2, angle))

        return horizontal_lines

    def filter_vertical_lines(
        self,
        lines: np.ndarray,
        angle_threshold: Optional[float] = None
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        Filter vertical lines from detected lines.

        Args:
            lines: Array of detected lines
            angle_threshold: Maximum deviation from vertical (degrees)

        Returns:
            List of vertical lines with their angles
        """
        angle_threshold = angle_threshold or self.rotation_config.vertical_line_threshold

        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = self.calculate_line_angle(x1, y1, x2, y2)

            # Check if line is close to vertical (90°)
            if abs(angle - 90) <= angle_threshold:
                vertical_lines.append((x1, y1, x2, y2, angle))

        return vertical_lines

    def estimate_skew_angle(
        self,
        lines: np.ndarray,
        use_vertical: bool = False
    ) -> float:
        """
        Estimate skew angle from detected lines.

        Args:
            lines: Array of detected lines
            use_vertical: If True, use vertical lines; else use horizontal lines

        Returns:
            Estimated skew angle in degrees
        """
        if lines is None or len(lines) == 0:
            return 0.0

        if use_vertical:
            filtered_lines = self.filter_vertical_lines(lines)
            target_angle = 90.0
        else:
            filtered_lines = self.filter_horizontal_lines(lines)
            target_angle = 0.0

        if not filtered_lines:
            return 0.0

        # Calculate median angle
        angles = [line[4] for line in filtered_lines]

        # Adjust angles relative to target
        if use_vertical:
            adjusted_angles = [angle - 90 for angle in angles]
        else:
            # For horizontal lines, handle both 0 and 180 degrees
            adjusted_angles = []
            for angle in angles:
                if angle > 90:
                    adjusted_angles.append(angle - 180)
                else:
                    adjusted_angles.append(angle)

        # Return median angle
        skew_angle = np.median(adjusted_angles)

        return skew_angle

    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        interpolation: Optional[str] = None,
        border_mode: str = 'constant',
        border_value: int = 255
    ) -> np.ndarray:
        """
        Rotate image by given angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (positive = counter-clockwise)
            interpolation: Interpolation method
            border_mode: Border mode for rotation
            border_value: Border value (used for constant mode)

        Returns:
            Rotated image
        """
        interpolation = interpolation or self.rotation_config.rotation_interpolation

        # Map interpolation to OpenCV constant
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
        }
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)

        # Map border mode
        border_map = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT,
        }
        border = border_map.get(border_mode, cv2.BORDER_CONSTANT)

        # Get image center
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate image
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (w, h),
            flags=interp,
            borderMode=border,
            borderValue=border_value
        )

        return rotated

    def correct_rotation(
        self,
        image: np.ndarray,
        return_angle: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Detect and correct rotation in ECG image.

        Args:
            image: Input image
            return_angle: If True, also return detected angle

        Returns:
            Corrected image, and optionally the detected angle
        """
        if not self.rotation_config.use_hough_transform:
            if return_angle:
                return image, 0.0
            return image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Detect edges
        edges = self.detect_edges(gray)

        # Detect lines
        lines = self.detect_lines_hough(edges)

        if lines is None or len(lines) == 0:
            print("Warning: No lines detected for rotation correction")
            if return_angle:
                return image, 0.0
            return image

        # Estimate skew angle (prefer horizontal lines)
        skew_angle = self.estimate_skew_angle(lines, use_vertical=False)

        # If angle is too large, something might be wrong
        if abs(skew_angle) > self.rotation_config.max_rotation_angle:
            print(f"Warning: Detected angle {skew_angle:.2f}° exceeds max threshold")
            skew_angle = 0.0

        # Rotate image to correct skew
        if abs(skew_angle) > 0.1:  # Only rotate if angle is significant
            corrected = self.rotate_image(image, skew_angle)
        else:
            corrected = image

        if return_angle:
            return corrected, skew_angle

        return corrected

    def visualize_lines(
        self,
        image: np.ndarray,
        lines: np.ndarray,
        horizontal_color: Tuple[int, int, int] = (255, 0, 0),
        vertical_color: Tuple[int, int, int] = (0, 255, 0),
        other_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """
        Visualize detected lines on image.

        Args:
            image: Input image
            lines: Detected lines
            horizontal_color: Color for horizontal lines (RGB)
            vertical_color: Color for vertical lines (RGB)
            other_color: Color for other lines (RGB)

        Returns:
            Image with lines drawn
        """
        # Create color image if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            vis_image = image.copy()

        if lines is None:
            return vis_image

        # Get filtered lines
        horizontal_lines = self.filter_horizontal_lines(lines)
        vertical_lines = self.filter_vertical_lines(lines)

        horizontal_set = set((x1, y1, x2, y2) for x1, y1, x2, y2, _ in horizontal_lines)
        vertical_set = set((x1, y1, x2, y2) for x1, y1, x2, y2, _ in vertical_lines)

        # Draw lines
        for line in lines:
            x1, y1, x2, y2 = line[0]
            coords = (x1, y1, x2, y2)

            if coords in horizontal_set:
                color = horizontal_color
            elif coords in vertical_set:
                color = vertical_color
            else:
                color = other_color

            cv2.line(vis_image, (x1, y1), (x2, y2), color, 2)

        return vis_image


def correct_image_rotation(
    image: np.ndarray,
    config=None,
    return_angle: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Convenience function to correct rotation in an ECG image.

    Args:
        image: Input image
        config: Configuration object
        return_angle: If True, also return detected angle

    Returns:
        Corrected image, and optionally the detected angle
    """
    corrector = ECGRotationCorrector(config)
    return corrector.correct_rotation(image, return_angle)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python rotation.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)

    # Correct rotation
    corrector = ECGRotationCorrector()
    corrected, angle = corrector.correct_rotation(image, return_angle=True)

    print(f"Detected skew angle: {angle:.2f}°")

    # Save corrected image
    output_path = Path(image_path).parent / f"{Path(image_path).stem}_rotated.png"
    cv2.imwrite(str(output_path), corrected)
    print(f"Saved corrected image to: {output_path}")

    # Visualize lines
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = corrector.detect_edges(gray)
    lines = corrector.detect_lines_hough(edges)

    if lines is not None:
        vis_image = corrector.visualize_lines(image, lines)
        vis_path = Path(image_path).parent / f"{Path(image_path).stem}_lines.png"
        cv2.imwrite(str(vis_path), vis_image)
        print(f"Saved line visualization to: {vis_path}")
