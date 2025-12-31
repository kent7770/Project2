"""
PCA-based object tracker.
Main tracking logic using PCA appearance model with motion prediction and occlusion handling.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass
from .appearance_model import AppearanceModel
from .feature_extractor import FeatureExtractor


@dataclass
class TrackingResult:
    """Result of tracking update."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    similarity: float
    is_occluded: bool
    scale: float


class PCATracker:
    """PCA-based object tracker with motion prediction and occlusion handling."""
    
    def __init__(self, 
                 patch_size: Tuple[int, int] = (64, 64),
                 n_components: Optional[int] = None,
                 variance_threshold: float = 0.95,
                 search_radius: int = 30,
                 similarity_method: str = 'combined',
                 update_rate: int = 3,
                 learning_rate: float = 0.05,
                 occlusion_threshold: float = 0.3,
                 use_motion_model: bool = True,
                 use_multiscale: bool = True,
                 use_hog: bool = False):
        """
        Initialize PCA tracker.
        
        Args:
            patch_size: Size of normalized patches (height, width)
            n_components: Number of PCA components (None for variance-based)
            variance_threshold: Minimum variance to retain
            search_radius: Search radius in pixels
            similarity_method: Similarity metric
            update_rate: Update appearance model every N frames
            learning_rate: Learning rate for incremental updates
            occlusion_threshold: Threshold below which object is considered occluded
            use_motion_model: Whether to use motion prediction
            use_multiscale: Whether to search at multiple scales
            use_hog: Whether to use HOG features
        """
        self.patch_size = patch_size
        self.search_radius = search_radius
        self.similarity_method = similarity_method
        self.update_rate = update_rate
        self.learning_rate = learning_rate
        self.occlusion_threshold = occlusion_threshold
        self.use_motion_model = use_motion_model
        self.use_multiscale = use_multiscale
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(patch_size=patch_size, use_hog=use_hog)
        self.appearance_model = AppearanceModel(
            n_components=n_components, 
            variance_threshold=variance_threshold
        )
        
        # Tracking state
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.prev_bbox: Optional[Tuple[int, int, int, int]] = None
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.scale: float = 1.0
        self.is_initialized = False
        self.frame_count = 0
        self.occlusion_count = 0
        self.max_occlusion_frames = 10
        
        # Confidence tracking
        self.confidence: float = 1.0
        self.confidence_history: List[float] = []
        
        # Training patches
        self.training_patches: List[np.ndarray] = []
    
    def initialize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with first frame and bounding box.
        
        Args:
            frame: First frame image
            bbox: Initial bounding box (x, y, width, height)
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.bbox = bbox
        self.prev_bbox = bbox
        self.velocity = (0.0, 0.0)
        self.scale = 1.0
        self.frame_count = 0
        self.occlusion_count = 0
        self.confidence = 1.0
        self.confidence_history = []
        self.training_patches = []
        
        x, y, w, h = self.feature_extractor.parse_bbox(bbox, frame.shape)
        
        # Extract initial patch
        patch = self.feature_extractor.extract_and_flatten(frame, (x, y, w, h))
        if patch is None:
            return False
        
        self.training_patches.append(patch)
        
        # Extract patches with variations for robust PCA
        variations = [
            # Position variations
            (-3, -3, 1.0), (3, 3, 1.0), (-3, 3, 1.0), (3, -3, 1.0),
            (0, -5, 1.0), (0, 5, 1.0), (-5, 0, 1.0), (5, 0, 1.0),
            # Scale variations
            (0, 0, 0.95), (0, 0, 1.05), (0, 0, 0.9), (0, 0, 1.1),
        ]
        
        cx, cy = x + w // 2, y + h // 2
        
        for dx, dy, s in variations:
            sw, sh = int(w * s), int(h * s)
            sx = cx + dx - sw // 2
            sy = cy + dy - sh // 2
            
            patch_var = self.feature_extractor.extract_and_flatten(frame, (sx, sy, sw, sh))
            if patch_var is not None:
                self.training_patches.append(patch_var)
        
        # Train PCA model
        success = self.appearance_model.train(self.training_patches)
        
        if success:
            self.is_initialized = True
            self.frame_count = 1
        
        return success
    
    def _predict_position(self) -> Tuple[int, int]:
        """Predict next position using motion model."""
        if self.bbox is None:
            return (0, 0)
        
        x, y, w, h = self.bbox
        cx, cy = x + w // 2, y + h // 2
        
        if self.use_motion_model and self.prev_bbox is not None:
            # Use velocity for prediction
            pred_cx = int(cx + self.velocity[0])
            pred_cy = int(cy + self.velocity[1])
            return (pred_cx, pred_cy)
        
        return (cx, cy)
    
    def _update_motion_model(self, new_bbox: Tuple[int, int, int, int]):
        """Update velocity estimate from new bounding box."""
        if self.bbox is None:
            return
        
        old_x, old_y, old_w, old_h = self.bbox
        new_x, new_y, new_w, new_h = new_bbox
        
        old_cx = old_x + old_w // 2
        old_cy = old_y + old_h // 2
        new_cx = new_x + new_w // 2
        new_cy = new_y + new_h // 2
        
        # Exponential moving average for velocity
        alpha = 0.3
        new_vx = new_cx - old_cx
        new_vy = new_cy - old_cy
        
        self.velocity = (
            (1 - alpha) * self.velocity[0] + alpha * new_vx,
            (1 - alpha) * self.velocity[1] + alpha * new_vy
        )
    
    def update(self, frame: np.ndarray) -> Optional[TrackingResult]:
        """
        Update tracker with new frame.
        
        Args:
            frame: Current frame image
        
        Returns:
            TrackingResult with bbox and confidence, or None if tracking failed
        """
        if not self.is_initialized or self.bbox is None:
            return None
        
        self.frame_count += 1
        
        # Get predicted center
        pred_cx, pred_cy = self._predict_position()
        x, y, w, h = self.bbox
        
        # Determine search parameters
        search_radius = self.search_radius
        if self.occlusion_count > 0:
            # Expand search when recovering from occlusion
            search_radius = int(search_radius * (1 + 0.5 * self.occlusion_count))
        
        # Generate scales for multi-scale search
        if self.use_multiscale:
            scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        else:
            scales = [1.0]
        
        # Extract candidates
        step_size = max(2, min(w, h) // 15)
        candidates = self.feature_extractor.extract_candidates(
            frame,
            center=(pred_cx, pred_cy),
            base_size=(w, h),
            search_radius=search_radius,
            step_size=step_size,
            scales=scales
        )
        
        if len(candidates) == 0:
            self.occlusion_count += 1
            return TrackingResult(
                bbox=self.bbox,
                confidence=0.0,
                similarity=0.0,
                is_occluded=True,
                scale=self.scale
            )
        
        # Find best match
        best_similarity = -np.inf
        best_bbox = None
        best_patch = None
        best_scale = 1.0
        
        for patch, candidate_bbox in candidates:
            similarity = self.appearance_model.compute_similarity(
                patch, method=self.similarity_method
            )
            
            if similarity is not None and similarity > best_similarity:
                best_similarity = similarity
                best_bbox = candidate_bbox
                best_patch = patch
                # Calculate scale from bbox size
                best_scale = candidate_bbox[2] / w if w > 0 else 1.0
        
        # Check for occlusion
        is_occluded = best_similarity < self.occlusion_threshold
        
        if is_occluded:
            self.occlusion_count += 1
            
            # If occluded too long, keep last known position with motion
            if self.occlusion_count <= self.max_occlusion_frames:
                # Predict position using motion
                pred_x = int(x + self.velocity[0])
                pred_y = int(y + self.velocity[1])
                self.prev_bbox = self.bbox
                self.bbox = (pred_x, pred_y, w, h)
                
                return TrackingResult(
                    bbox=self.bbox,
                    confidence=0.1,
                    similarity=best_similarity,
                    is_occluded=True,
                    scale=self.scale
                )
            else:
                # Lost tracking
                return None
        
        # Good match found
        self.occlusion_count = 0
        
        if best_bbox is not None:
            # Update motion model
            self._update_motion_model(best_bbox)
            
            # Update state
            self.prev_bbox = self.bbox
            self.bbox = best_bbox
            self.scale = best_scale
            
            # Calculate confidence
            if best_patch is not None:
                self.confidence = self.appearance_model.get_confidence(best_patch)
            else:
                self.confidence = best_similarity
            
            self.confidence_history.append(self.confidence)
            
            # Update appearance model
            should_update = (
                self.frame_count % self.update_rate == 0 and
                best_patch is not None and
                self.confidence > 0.5
            )
            
            if should_update:
                self.appearance_model.update(best_patch, learning_rate=self.learning_rate)
            
            return TrackingResult(
                bbox=self.bbox,
                confidence=self.confidence,
                similarity=best_similarity,
                is_occluded=False,
                scale=self.scale
            )
        
        return TrackingResult(
            bbox=self.bbox,
            confidence=0.0,
            similarity=best_similarity,
            is_occluded=True,
            scale=self.scale
        )
    
    def get_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Get current bounding box."""
        return self.bbox
    
    def get_confidence(self) -> float:
        """Get current tracking confidence."""
        return self.confidence
    
    def is_tracking(self) -> bool:
        """Check if tracker is initialized and tracking."""
        return self.is_initialized and self.occlusion_count <= self.max_occlusion_frames
    
    def is_occluded(self) -> bool:
        """Check if object is currently occluded."""
        return self.occlusion_count > 0
    
    def reset(self):
        """Reset tracker state."""
        self.bbox = None
        self.prev_bbox = None
        self.velocity = (0.0, 0.0)
        self.scale = 1.0
        self.is_initialized = False
        self.frame_count = 0
        self.occlusion_count = 0
        self.confidence = 1.0
        self.confidence_history = []
        self.training_patches = []
        
        self.appearance_model = AppearanceModel(
            n_components=self.appearance_model.n_components,
            variance_threshold=self.appearance_model.variance_threshold
        )
    
    def retrain(self):
        """Retrain PCA model with accumulated patches."""
        self.appearance_model.retrain()
