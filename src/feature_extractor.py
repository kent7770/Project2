"""
Feature extraction utilities for object tracking.
Extracts and preprocesses image patches from bounding boxes.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


class FeatureExtractor:
    """Extracts and preprocesses features from image patches."""
    
    def __init__(self, patch_size: Tuple[int, int] = (64, 64), use_hog: bool = False):
        """
        Initialize feature extractor.
        
        Args:
            patch_size: Target size (height, width) for normalized patches
            use_hog: Whether to use HOG features instead of raw pixels
        """
        self.patch_size = patch_size
        self.use_hog = use_hog
        
        # HOG parameters
        if use_hog:
            self.hog = cv2.HOGDescriptor(
                _winSize=(patch_size[1], patch_size[0]),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9
            )
    
    def extract_patch(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     pad_mode: str = 'edge') -> Optional[np.ndarray]:
        """
        Extract a patch from image given bounding box coordinates.
        
        Args:
            image: Input image (BGR or grayscale)
            bbox: Bounding box as (x, y, width, height)
            pad_mode: Padding mode for out-of-bounds regions ('edge', 'constant', 'reflect')
        
        Returns:
            Extracted and normalized patch, or None if bbox is invalid
        """
        x, y, w, h = self.parse_bbox(bbox, image.shape)
        
        if w <= 0 or h <= 0:
            return None
        
        # Handle out-of-bounds with padding
        img_h, img_w = image.shape[:2]
        
        # Calculate padding needed
        pad_left = max(0, -x)
        pad_top = max(0, -y)
        pad_right = max(0, (x + w) - img_w)
        pad_bottom = max(0, (y + h) - img_h)
        
        # Clip coordinates to image bounds
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(img_w, x + w)
        y_end = min(img_h, y + h)
        
        if x_end <= x_start or y_end <= y_start:
            return None
        
        # Extract valid region
        patch = image[y_start:y_end, x_start:x_end]
        
        # Apply padding if needed
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            if len(image.shape) == 3:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            else:
                pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
            patch = np.pad(patch, pad_width, mode=pad_mode)
        
        # Resize to target size
        patch = cv2.resize(patch, (self.patch_size[1], self.patch_size[0]), 
                          interpolation=cv2.INTER_LINEAR)
        
        return patch
    
    def extract_and_flatten(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                           normalize: bool = True) -> Optional[np.ndarray]:
        """
        Extract patch and flatten to feature vector.
        
        Args:
            image: Input image
            bbox: Bounding box coordinates
            normalize: Whether to normalize features
        
        Returns:
            Flattened feature vector, or None if extraction fails
        """
        patch = self.extract_patch(image, bbox)
        if patch is None:
            return None
        
        # Convert to grayscale if needed
        if len(patch.shape) == 3:
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray_patch = patch
        
        if self.use_hog:
            # Extract HOG features
            feature_vector = self.hog.compute(gray_patch).flatten()
        else:
            # Use raw pixel values
            feature_vector = gray_patch.astype(np.float32).flatten()
        
        if normalize:
            # Zero-mean unit-variance normalization
            mean = np.mean(feature_vector)
            std = np.std(feature_vector) + 1e-10
            feature_vector = (feature_vector - mean) / std
        
        return feature_vector
    
    def extract_multiscale(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                          scales: List[float] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Extract patches at multiple scales.
        
        Args:
            image: Input image
            bbox: Base bounding box coordinates
            scales: List of scale factors (default: [0.9, 1.0, 1.1])
        
        Returns:
            List of (patch, scale) tuples
        """
        if scales is None:
            scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        x, y, w, h = self.parse_bbox(bbox, image.shape)
        cx, cy = x + w // 2, y + h // 2
        
        results = []
        for scale in scales:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            scaled_x = cx - scaled_w // 2
            scaled_y = cy - scaled_h // 2
            
            scaled_bbox = (scaled_x, scaled_y, scaled_w, scaled_h)
            patch = self.extract_and_flatten(image, scaled_bbox)
            
            if patch is not None:
                results.append((patch, scale))
        
        return results
    
    def extract_color_histogram(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                                bins: int = 32, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Extract color histogram features from patch.
        
        Args:
            image: Input image (BGR)
            bbox: Bounding box coordinates
            bins: Number of bins per channel
            normalize: Whether to normalize histogram
        
        Returns:
            Concatenated histogram features, or None if extraction fails
        """
        patch = self.extract_patch(image, bbox)
        if patch is None:
            return None
        
        if len(patch.shape) == 2:
            # Grayscale image
            hist = cv2.calcHist([patch], [0], None, [bins], [0, 256])
        else:
            # Color image - use HSV for better color representation
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv_patch], [0], None, [bins], [0, 180])
            hist_s = cv2.calcHist([hsv_patch], [1], None, [bins], [0, 256])
            hist_v = cv2.calcHist([hsv_patch], [2], None, [bins], [0, 256])
            hist = np.concatenate([hist_h, hist_s, hist_v])
        
        hist = hist.flatten()
        
        if normalize:
            hist = hist / (np.sum(hist) + 1e-10)
        
        return hist
    
    def parse_bbox(self, bbox: Tuple[int, int, int, int], 
                   image_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """
        Parse bounding box to (x, y, width, height) format.
        
        Args:
            bbox: Bounding box (x, y, w, h) or (x1, y1, x2, y2)
            image_shape: Image shape (height, width, ...)
        
        Returns:
            (x, y, width, height)
        """
        if len(bbox) != 4:
            raise ValueError("Bounding box must have 4 elements")
        
        # Heuristic: if third and fourth values are larger than position values
        # and exceed image dimensions, assume (x1, y1, x2, y2) format
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            if bbox[2] > image_shape[1] or bbox[3] > image_shape[0]:
                # Format: (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox
                return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
        # Format: (x, y, width, height)
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    
    def extract_candidates(self, image: np.ndarray, 
                          center: Tuple[int, int],
                          base_size: Tuple[int, int],
                          search_radius: int,
                          step_size: int = 4,
                          scales: List[float] = None) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract candidate patches around a center point at multiple positions and scales.
        
        Args:
            image: Input image
            center: Center point (x, y)
            base_size: Base bounding box size (width, height)
            search_radius: Search radius in pixels
            step_size: Step size for position search
            scales: List of scale factors
        
        Returns:
            List of (patch, bbox) tuples
        """
        if scales is None:
            scales = [0.95, 1.0, 1.05]
        
        cx, cy = center
        base_w, base_h = base_size
        candidates = []
        
        for scale in scales:
            w = int(base_w * scale)
            h = int(base_h * scale)
            
            for dy in range(-search_radius, search_radius + 1, step_size):
                for dx in range(-search_radius, search_radius + 1, step_size):
                    x = cx + dx - w // 2
                    y = cy + dy - h // 2
                    bbox = (x, y, w, h)
                    
                    patch = self.extract_and_flatten(image, bbox)
                    if patch is not None:
                        candidates.append((patch, bbox))
        
        return candidates
