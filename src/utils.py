"""
Utility functions for I/O, visualization, evaluation metrics, and bounding box operations.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrackingMetrics:
    """Tracking evaluation metrics."""
    precision: float
    success_rate: float
    avg_iou: float
    frames_tracked: int
    frames_lost: int
    avg_center_error: float


def load_image_sequence(directory: str, 
                        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
                        max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load image sequence from directory.
    
    Args:
        directory: Path to directory containing images
        extensions: Tuple of valid image extensions
        max_frames: Maximum number of frames to load (None for all)
    
    Returns:
        List of images (numpy arrays) in sorted order
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    # Sort by filename (handle numeric sorting)
    def sort_key(path):
        name = path.stem
        # Try to extract number from filename
        import re
        numbers = re.findall(r'\d+', name)
        if numbers:
            return (int(numbers[-1]), name)
        return (0, name)
    
    image_files = sorted(image_files, key=sort_key)
    
    if max_frames:
        image_files = image_files[:max_frames]
    
    # Load images
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)
    
    return images


def load_image_sequence_lazy(directory: str,
                            extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
                            ) -> Generator[np.ndarray, None, None]:
    """
    Lazily load image sequence from directory (memory efficient).
    
    Args:
        directory: Path to directory containing images
        extensions: Tuple of valid image extensions
    
    Yields:
        Images one at a time
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    import re
    def sort_key(path):
        name = path.stem
        numbers = re.findall(r'\d+', name)
        if numbers:
            return (int(numbers[-1]), name)
        return (0, name)
    
    image_files = sorted(image_files, key=sort_key)
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            yield img


def load_ground_truth(filepath: str) -> List[Tuple[int, int, int, int]]:
    """
    Load ground truth bounding boxes from file.
    Supports common formats: comma-separated or space-separated values.
    
    Args:
        filepath: Path to ground truth file
    
    Returns:
        List of bounding boxes
    """
    bboxes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try comma-separated
            if ',' in line:
                values = [float(x) for x in line.split(',')]
            else:
                values = [float(x) for x in line.split()]
            
            if len(values) >= 4:
                bbox = (int(values[0]), int(values[1]), int(values[2]), int(values[3]))
                bboxes.append(bbox)
    
    return bboxes


def draw_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], 
              color: Tuple[int, int, int] = (0, 255, 0), 
              thickness: int = 2, 
              label: Optional[str] = None,
              confidence: Optional[float] = None) -> np.ndarray:
    """
    Draw bounding box on image with optional label and confidence.
    
    Args:
        image: Input image
        bbox: Bounding box (x, y, width, height)
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label text
        confidence: Optional confidence value to display
    
    Returns:
        Image with bounding box drawn
    """
    img = image.copy()
    x, y, w, h = bbox
    
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Clip to image bounds
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(0, min(x2, image.shape[1] - 1))
    y2 = max(0, min(y2, image.shape[0] - 1))
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw corner accents for better visibility
    corner_length = min(20, w // 4, h // 4)
    corner_thickness = thickness + 1
    
    # Top-left
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
    
    # Prepare label text
    if label or confidence is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        text_parts = []
        if label:
            text_parts.append(label)
        if confidence is not None:
            text_parts.append(f"{confidence:.1%}")
        
        text = " | ".join(text_parts)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # Draw background for text
        cv2.rectangle(img, (x1, y1 - text_height - baseline - 6), 
                     (x1 + text_width + 4, y1), color, -1)
        
        # Draw text
        cv2.putText(img, text, (x1 + 2, y1 - baseline - 2), 
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return img


def draw_trajectory(image: np.ndarray, 
                   trajectory: List[Tuple[int, int]],
                   color: Tuple[int, int, int] = (255, 255, 0),
                   max_points: int = 50) -> np.ndarray:
    """
    Draw object trajectory on image.
    
    Args:
        image: Input image
        trajectory: List of center points
        color: BGR color for trajectory
        max_points: Maximum number of points to draw
    
    Returns:
        Image with trajectory drawn
    """
    img = image.copy()
    
    # Use only recent points
    points = trajectory[-max_points:]
    
    if len(points) < 2:
        return img
    
    # Draw fading trajectory
    for i in range(1, len(points)):
        # Fade based on age
        alpha = i / len(points)
        point_color = tuple(int(c * alpha) for c in color)
        
        pt1 = (int(points[i-1][0]), int(points[i-1][1]))
        pt2 = (int(points[i][0]), int(points[i][1]))
        
        thickness = max(1, int(2 * alpha))
        cv2.line(img, pt1, pt2, point_color, thickness)
    
    # Draw current position
    if points:
        cv2.circle(img, (int(points[-1][0]), int(points[-1][1])), 4, color, -1)
    
    return img


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Both boxes should be in (x, y, width, height) format.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
    
    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2
    
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_center_error(bbox1: Tuple[int, int, int, int], 
                          bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate center location error between two bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
    
    Returns:
        Euclidean distance between centers
    """
    cx1 = bbox1[0] + bbox1[2] / 2
    cy1 = bbox1[1] + bbox1[3] / 2
    cx2 = bbox2[0] + bbox2[2] / 2
    cy2 = bbox2[1] + bbox2[3] / 2
    
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def get_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Get center point of bounding box."""
    return (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)


def evaluate_tracking(predicted: List[Tuple[int, int, int, int]],
                     ground_truth: List[Tuple[int, int, int, int]],
                     iou_threshold: float = 0.5,
                     center_threshold: float = 20.0) -> TrackingMetrics:
    """
    Evaluate tracking performance against ground truth.
    
    Args:
        predicted: List of predicted bounding boxes
        ground_truth: List of ground truth bounding boxes
        iou_threshold: IoU threshold for success
        center_threshold: Center error threshold for precision
    
    Returns:
        TrackingMetrics with evaluation results
    """
    if len(predicted) == 0 or len(ground_truth) == 0:
        return TrackingMetrics(0, 0, 0, 0, len(ground_truth), float('inf'))
    
    n_frames = min(len(predicted), len(ground_truth))
    
    ious = []
    center_errors = []
    successes = 0
    precisions = 0
    frames_tracked = 0
    frames_lost = 0
    
    for i in range(n_frames):
        pred = predicted[i]
        gt = ground_truth[i]
        
        if pred is None:
            frames_lost += 1
            continue
        
        frames_tracked += 1
        
        iou = calculate_iou(pred, gt)
        ious.append(iou)
        
        center_error = calculate_center_error(pred, gt)
        center_errors.append(center_error)
        
        if iou >= iou_threshold:
            successes += 1
        
        if center_error <= center_threshold:
            precisions += 1
    
    precision = precisions / frames_tracked if frames_tracked > 0 else 0
    success_rate = successes / frames_tracked if frames_tracked > 0 else 0
    avg_iou = np.mean(ious) if ious else 0
    avg_center_error = np.mean(center_errors) if center_errors else float('inf')
    
    return TrackingMetrics(
        precision=precision,
        success_rate=success_rate,
        avg_iou=avg_iou,
        frames_tracked=frames_tracked,
        frames_lost=frames_lost,
        avg_center_error=avg_center_error
    )


def save_tracking_results(output_dir: str, 
                         frames: List[np.ndarray], 
                         bboxes: List[Tuple[int, int, int, int]],
                         prefix: str = "tracked_frame",
                         draw_trajectory_on_frames: bool = True) -> None:
    """
    Save tracking results as images.
    
    Args:
        output_dir: Output directory
        frames: List of frames
        bboxes: List of bounding boxes (one per frame)
        prefix: Filename prefix
        draw_trajectory_on_frames: Whether to draw trajectory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trajectory = []
    
    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        if bbox is not None:
            center = get_bbox_center(bbox)
            trajectory.append(center)
            
            frame_out = draw_bbox(frame, bbox, label=f"Frame {i+1}")
            
            if draw_trajectory_on_frames and len(trajectory) > 1:
                frame_out = draw_trajectory(frame_out, trajectory)
        else:
            frame_out = frame
        
        output_path = output_dir / f"{prefix}_{i+1:04d}.jpg"
        cv2.imwrite(str(output_path), frame_out)
    
    # Save bounding boxes to file
    bbox_file = output_dir / "tracking_results.txt"
    with open(bbox_file, 'w') as f:
        for bbox in bboxes:
            if bbox is not None:
                f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
            else:
                f.write("NaN,NaN,NaN,NaN\n")


def create_tracking_video(frames: List[np.ndarray],
                         bboxes: List[Tuple[int, int, int, int]],
                         output_path: str,
                         fps: int = 30,
                         show_trajectory: bool = True,
                         confidences: Optional[List[float]] = None) -> None:
    """
    Create video from tracking results.
    
    Args:
        frames: List of frames
        bboxes: List of bounding boxes
        output_path: Output video path
        fps: Frames per second
        show_trajectory: Whether to show trajectory
        confidences: Optional list of confidence values
    """
    if len(frames) == 0:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    trajectory = []
    
    for i, frame in enumerate(frames):
        frame_out = frame.copy()
        bbox = bboxes[i] if i < len(bboxes) else None
        confidence = confidences[i] if confidences and i < len(confidences) else None
        
        if bbox is not None:
            center = get_bbox_center(bbox)
            trajectory.append(center)
            
            # Color based on confidence
            if confidence is not None:
                # Green to red based on confidence
                r = int(255 * (1 - confidence))
                g = int(255 * confidence)
                color = (0, g, r)
            else:
                color = (0, 255, 0)
            
            frame_out = draw_bbox(frame_out, bbox, color=color, 
                                 label=f"Frame {i+1}", confidence=confidence)
            
            if show_trajectory and len(trajectory) > 1:
                frame_out = draw_trajectory(frame_out, trajectory)
        
        out.write(frame_out)
    
    out.release()


def display_frame(frame: np.ndarray, 
                 window_name: str = "Tracking", 
                 wait_key: int = 1) -> int:
    """Display frame in a window."""
    cv2.imshow(window_name, frame)
    return cv2.waitKey(wait_key) & 0xFF


def print_progress(current: int, total: int, prefix: str = "Progress", 
                  bar_length: int = 40, extra_info: str = "") -> None:
    """Print progress bar to console."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '=' * filled + '-' * (bar_length - filled)
    
    info = f" {extra_info}" if extra_info else ""
    print(f"\r{prefix}: [{bar}] {current}/{total} ({percent:.1%}){info}", end='', flush=True)
    
    if current == total:
        print()
