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
                 update_rate: int = 2,
                 learning_rate: float = 0.08,
                 occlusion_threshold: float = 0.30,
                 use_motion_model: bool = True,
                 use_multiscale: bool = True,
                 use_hog: bool = False,
                 fast_mode: bool = False):
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
            fast_mode: Enable fast mode (reduced accuracy, better speed)
        """
        # Apply fast mode optimizations
        if fast_mode:
            patch_size = (48, 48)
            search_radius = min(search_radius, 25)
            use_multiscale = False
            occlusion_threshold = 0.22  # Balanced in fast mode
        
        self.patch_size = patch_size
        self.search_radius = search_radius
        self.similarity_method = similarity_method
        self.update_rate = update_rate
        self.learning_rate = learning_rate
        self.occlusion_threshold = occlusion_threshold
        self.use_motion_model = use_motion_model
        self.use_multiscale = use_multiscale
        self.fast_mode = fast_mode
        
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
        self.max_occlusion_frames = 50  # Balanced
        
        # Confidence tracking
        self.confidence: float = 1.0
        self.confidence_history: List[float] = []
        
        # Training patches
        self.training_patches: List[np.ndarray] = []

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

        self._pca_search_interval = 12 if fast_mode else 6

        self._template: Optional[np.ndarray] = None
        self._template_bbox_size: Optional[Tuple[int, int]] = None
        self._template_update_interval = 5

        self._base_template: Optional[np.ndarray] = None
        self._base_template_bbox_size: Optional[Tuple[int, int]] = None

    def _init_flow_points(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(int(gray.shape[1]), int(x + w))
        y2 = min(int(gray.shape[0]), int(y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        mask = np.zeros_like(gray)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=160,
            qualityLevel=0.008,
            minDistance=2,
            blockSize=7,
            mask=mask,
        )
        return pts

    def _clip_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...],
    ) -> Tuple[int, int, int, int]:
        img_h, img_w = image_shape[:2]
        x, y, w, h = bbox
        w = max(1, int(w))
        h = max(1, int(h))
        x = int(np.clip(int(x), 0, max(0, img_w - w)))
        y = int(np.clip(int(y), 0, max(0, img_h - h)))
        return (x, y, w, h)

    def _get_gray_patch(
        self,
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(int(gray.shape[1]), int(x + w))
        y2 = min(int(gray.shape[0]), int(y + h))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return None
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch

    def _template_match(
        self,
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
        search_radius: int,
        use_base_only: bool = False,
    ) -> Tuple[bool, Tuple[int, int, int, int], float]:
        x, y, w, h = bbox
        if w <= 2 or h <= 2:
            return False, bbox, 0.0

        templates = []
        if use_base_only:
            if self._base_template is not None:
                templates.append(self._base_template)
            elif self._template is not None:
                templates.append(self._template)
        else:
            if self._template is not None:
                templates.append(self._template)
            if self._base_template is not None and self._base_template is not self._template:
                templates.append(self._base_template)
        if len(templates) == 0:
            return False, bbox, 0.0

        sx1 = max(0, int(x - search_radius))
        sy1 = max(0, int(y - search_radius))
        sx2 = min(int(gray.shape[1]), int(x + w + search_radius))
        sy2 = min(int(gray.shape[0]), int(y + h + search_radius))
        search_img = gray[sy1:sy2, sx1:sx2]
        if search_img.size == 0:
            return False, bbox, 0.0

        best_score = 0.0
        best_loc = None
        best_templ = None
        thresh = 0.50 if self.fast_mode else 0.55

        for templ in templates:
            if sx2 <= sx1 + templ.shape[1] + 1 or sy2 <= sy1 + templ.shape[0] + 1:
                continue
            try:
                res = cv2.matchTemplate(search_img, templ, cv2.TM_CCOEFF_NORMED)
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
            except Exception:
                continue

            score = float(max_val)
            if not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_loc = max_loc
                best_templ = templ

        if best_loc is None:
            return False, bbox, 0.0
        if best_score < thresh:
            return False, bbox, best_score

        nx = int(sx1 + best_loc[0])
        ny = int(sy1 + best_loc[1])
        new_bbox = self._clip_bbox((nx, ny, w, h), gray.shape)

        if best_templ is self._base_template:
            self._template = self._base_template

        return True, new_bbox, best_score

    def _set_flow_state(
        self,
        gray: np.ndarray,
        bbox: Tuple[int, int, int, int],
        points: Optional[np.ndarray] = None,
        reinit: bool = False,
    ) -> None:
        self.prev_gray = gray
        if reinit or points is None or (hasattr(points, "shape") and points.shape[0] < 10):
            self.prev_points = self._init_flow_points(gray, bbox)
        else:
            self.prev_points = points
    
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

        # Initialize optical flow state
        if frame is not None:
            if len(frame.shape) == 3:
                self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                self.prev_gray = frame.copy()
            self.prev_points = self._init_flow_points(self.prev_gray, bbox)

            clipped = self._clip_bbox(bbox, frame.shape)
            self._template = self._get_gray_patch(self.prev_gray, clipped)
            self._template_bbox_size = (clipped[2], clipped[3])
            self._base_template = self._template
            self._base_template_bbox_size = self._template_bbox_size
        
        x, y, w, h = self.feature_extractor.parse_bbox(bbox, frame.shape)
        
        # Extract initial patch
        patch = self.feature_extractor.extract_and_flatten(frame, (x, y, w, h))
        if patch is None:
            return False
        
        self.training_patches.append(patch)
        
        # Extract patches with variations for robust PCA
        variations = [
            # Position variations (more samples)
            (-3, -3, 1.0), (3, 3, 1.0), (-3, 3, 1.0), (3, -3, 1.0),
            (0, -5, 1.0), (0, 5, 1.0), (-5, 0, 1.0), (5, 0, 1.0),
            (-2, -2, 1.0), (2, 2, 1.0), (-2, 2, 1.0), (2, -2, 1.0),
            (0, -3, 1.0), (0, 3, 1.0), (-3, 0, 1.0), (3, 0, 1.0),
            # Scale variations
            (0, 0, 0.95), (0, 0, 1.05), (0, 0, 0.9), (0, 0, 1.1),
            (0, 0, 0.85), (0, 0, 1.15),
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
            conf = float(np.clip(self.confidence, 0.0, 1.0))
            if self.occlusion_count > 0:
                conf *= 0.25
            if conf < 0.25:
                return (cx, cy)

            pred_cx = int(cx + self.velocity[0] * conf)
            pred_cy = int(cy + self.velocity[1] * conf)
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

        max_step = max(5.0, max(old_w, old_h) * 0.75)
        new_vx = float(np.clip(new_vx, -max_step, max_step))
        new_vy = float(np.clip(new_vy, -max_step, max_step))
        
        self.velocity = (
            (1 - alpha) * self.velocity[0] + alpha * new_vx,
            (1 - alpha) * self.velocity[1] + alpha * new_vy
        )

    def _flow_step(self, gray: np.ndarray) -> Tuple[bool, float, float, Optional[np.ndarray]]:
        if self.prev_gray is None or self.prev_points is None:
            return False, 0.0, 0.0, None
        if not hasattr(self.prev_points, "shape") or self.prev_points.shape[0] < 10:
            return False, 0.0, 0.0, None

        try:
            next_pt, status, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                self.prev_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except Exception:
            return False, 0.0, 0.0, None

        if status is None or next_pt is None:
            return False, 0.0, 0.0, None

        status_fwd = status.reshape(-1) == 1
        if int(np.sum(status_fwd)) < 8:
            return False, 0.0, 0.0, None

        try:
            back_pt, status_bwd, _err_bwd = cv2.calcOpticalFlowPyrLK(
                gray,
                self.prev_gray,
                next_pt,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except Exception:
            return False, 0.0, 0.0, None

        if status_bwd is None or back_pt is None:
            return False, 0.0, 0.0, None

        status_bwd = status_bwd.reshape(-1) == 1
        prev_xy = self.prev_points[:, 0, :]
        back_xy = back_pt[:, 0, :]
        fb_err = np.linalg.norm(prev_xy - back_xy, axis=1)
        fb_ok = fb_err < 1.5
        good = status_fwd & status_bwd & fb_ok
        if int(np.sum(good)) < 8:
            return False, 0.0, 0.0, None

        prev_good = self.prev_points[good, 0, :]
        next_good = next_pt[good, 0, :]
        dxy = next_good - prev_good
        dx = float(np.median(dxy[:, 0]))
        dy = float(np.median(dxy[:, 1]))
        next_points = next_pt[good].reshape(-1, 1, 2).astype(np.float32)
        return True, dx, dy, next_points
    
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

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self.frame_count += 1

        x, y, w, h = self.bbox

        if self._template_bbox_size is not None and (w, h) != self._template_bbox_size:
            patch0 = self._get_gray_patch(gray, self.bbox)
            if patch0 is not None:
                self._template = patch0
                self._template_bbox_size = (w, h)

        flow_ok, fdx, fdy, next_points = self._flow_step(gray)
        flow_patch = None
        flow_similarity = None
        flow_bbox = None
        templ_ok = False
        templ_score = 0.0
        templ_bbox = None
        if flow_ok:
            max_step = max(3.0, float(max(w, h)) * 0.35)
            fdx = float(np.clip(fdx, -max_step, max_step))
            fdy = float(np.clip(fdy, -max_step, max_step))
            nx = int(x + fdx)
            ny = int(y + fdy)
            self.prev_bbox = self.bbox
            self.bbox = self._clip_bbox((nx, ny, w, h), frame.shape)
            flow_bbox = self.bbox
            self.velocity = (
                0.7 * self.velocity[0] + 0.3 * fdx,
                0.7 * self.velocity[1] + 0.3 * fdy,
            )
            self._set_flow_state(gray, self.bbox, points=next_points, reinit=False)

            flow_patch = self.feature_extractor.extract_and_flatten(frame, self.bbox)
            if flow_patch is not None:
                flow_similarity = self.appearance_model.compute_similarity(flow_patch, method=self.similarity_method)
                conf1 = self.appearance_model.get_confidence(flow_patch)
                conf2 = float(flow_similarity) if flow_similarity is not None else 0.0
                self.confidence = float(max(conf1, conf2))
                if self.frame_count % self.update_rate == 0 and self.confidence > 0.55:
                    self.appearance_model.update(flow_patch, learning_rate=self.learning_rate)

            templ_ok, templ_bbox, templ_score = self._template_match(
                gray,
                self.bbox,
                search_radius=max(10, int(self.search_radius * 1.2)),
            )
            if templ_ok and templ_bbox is not None:
                if self.confidence < 0.55 or (flow_similarity is not None and float(flow_similarity) < 0.35):
                    self.prev_bbox = self.bbox
                    self.bbox = templ_bbox
                    self.velocity = (self.velocity[0] * 0.7, self.velocity[1] * 0.7)
                    self._set_flow_state(gray, self.bbox, reinit=True)

                self.confidence = float(max(self.confidence, templ_score))

            if (self.frame_count % 10) == 0 or self.confidence < 0.45:
                self._set_flow_state(gray, self.bbox, reinit=True)

            if templ_ok and (self.frame_count % self._template_update_interval) == 0:
                if self.confidence > 0.70 and (flow_similarity is None or float(flow_similarity) > 0.45):
                    new_templ = self._get_gray_patch(gray, self.bbox)
                    if new_templ is not None and self._template_bbox_size == (self.bbox[2], self.bbox[3]):
                        self._template = new_templ

            do_pca_search = False
            if self.occlusion_count > 0:
                do_pca_search = True
            elif self.confidence < (0.35 if self.fast_mode else 0.45):
                do_pca_search = True
            elif (self.frame_count % self._pca_search_interval) == 0 and self.confidence < 0.6:
                do_pca_search = True

            if not do_pca_search:
                self.occlusion_count = 0
                return TrackingResult(
                    bbox=self.bbox,
                    confidence=self.confidence,
                    similarity=self.confidence,
                    is_occluded=False,
                    scale=self.scale,
                )

        # Get predicted center
        if flow_ok and self.bbox is not None:
            x0, y0, w0, h0 = self.bbox
            pred_cx = x0 + w0 // 2
            pred_cy = y0 + h0 // 2
        else:
            pred_cx, pred_cy = self._predict_position()

        if not flow_ok and self.bbox is not None and self._template is not None:
            ok, tb, ts = self._template_match(
                gray,
                self.bbox,
                search_radius=max(20, int(self.search_radius * 1.5)),
                use_base_only=True,
            )
            if ok:
                self.prev_bbox = self.bbox
                self.bbox = tb
                self.confidence = float(max(self.confidence, ts))
                self._set_flow_state(gray, self.bbox, reinit=True)

        x, y, w, h = self.bbox
        
        # Determine search parameters adaptively
        search_radius = max(self.search_radius, int(max(w, h) * 0.8))
        if self.occlusion_count > 0:
            # Expand search when recovering from occlusion
            search_radius = min(
                int(search_radius * (1 + 0.3 * self.occlusion_count)),
                int(search_radius * 1.5)
            )
        search_radius = int(search_radius)

        search_radius = min(search_radius, 90 if self.fast_mode else 120)
        
        # Generate scales for multi-scale search
        if self.use_multiscale:
            scales = [0.95, 1.0, 1.05]  # Reduced from 5 to 3 scales
        else:
            scales = [1.0]
        
        # Adaptive step size based on object size and confidence
        base_step = min(w, h) // 15  # Smaller steps for accuracy
        if self.confidence > 0.6:
            step_size = max(3, base_step)  # Larger steps when confident
        elif self.confidence > 0.3:
            step_size = max(2, base_step // 2)  # Medium steps
        else:
            step_size = max(2, base_step // 3)  # Smaller steps when uncertain
        
        # Fast mode: balance speed and accuracy
        if self.fast_mode:
            step_size = max(3, min(w, h) // 12)

        max_candidates = 160 if self.fast_mode else 240
        if self.occlusion_count > 0:
            max_candidates = 400
        
        try:
            candidate_iter = self.feature_extractor.iter_candidates(
                image=frame,
                center=(pred_cx, pred_cy),
                base_size=(w, h),
                search_radius=search_radius,
                step_size=step_size,
                scales=scales,
                max_candidates=max_candidates,
            )
        except Exception as e:
            self._set_flow_state(gray, self.bbox, reinit=True)
            return TrackingResult(
                bbox=self.bbox,
                confidence=0.0,
                similarity=0.0,
                is_occluded=True,
                scale=self.scale
            )
        
        found_any_candidate = False

        best_similarity = -np.inf
        best_bbox = None
        best_patch = None
        best_scale = 1.0

        curr_bbox = flow_bbox if flow_bbox is not None else self.bbox
        if curr_bbox is not None and flow_similarity is not None and flow_patch is not None:
            found_any_candidate = True
            best_similarity = float(flow_similarity)
            best_bbox = curr_bbox
            best_patch = flow_patch
            best_scale = 1.0

        if curr_bbox is not None:
            cbx, cby, cbw, cbh = curr_bbox
            curr_cx = cbx + cbw // 2
            curr_cy = cby + cbh // 2

            if flow_similarity is None or float(flow_similarity) < 0.30:
                max_jump = float(search_radius)
            else:
                max_jump = min(float(search_radius) * 0.6, float(max(20, max(cbw, cbh) * 0.9)))
        else:
            curr_cx, curr_cy = pred_cx, pred_cy
            max_jump = float(search_radius)
        good_enough_threshold = 0.80

        for i, (patch, candidate_bbox) in enumerate(candidate_iter):
            found_any_candidate = True
            try:
                bx, by, bw, bh = candidate_bbox
                cand_cx = bx + bw // 2
                cand_cy = by + bh // 2
                if float(np.hypot(cand_cx - curr_cx, cand_cy - curr_cy)) > max_jump:
                    continue
                similarity = self.appearance_model.compute_similarity(
                    patch, method=self.similarity_method
                )

                if similarity is not None and similarity > best_similarity:
                    best_similarity = similarity
                    best_bbox = candidate_bbox
                    best_patch = patch
                    best_scale = candidate_bbox[2] / w if w > 0 else 1.0

                    if best_similarity > good_enough_threshold:
                        break
            except Exception:
                continue

        if not found_any_candidate:
            self.occlusion_count += 1
            self._set_flow_state(gray, self.bbox, reinit=True)
            return TrackingResult(
                bbox=self.bbox,
                confidence=0.0,
                similarity=0.0,
                is_occluded=True,
                scale=self.scale
            )
        
        # Check for occlusion with adaptive threshold
        adaptive_threshold = self.occlusion_threshold
        if self.frame_count < 20:
            adaptive_threshold = self.occlusion_threshold * 0.8  # Slightly lenient early on
        
        is_occluded = best_similarity < adaptive_threshold
        if flow_ok and curr_bbox is not None and best_bbox == curr_bbox and best_similarity >= 0.20:
            is_occluded = False
        if templ_ok and templ_score >= 0.55:
            is_occluded = False
        
        if is_occluded:
            self.occlusion_count += 1

            if self.occlusion_count >= 2 and self.bbox is not None:
                global_radius = max(int(gray.shape[0]), int(gray.shape[1]))
                gok, gbb, gscore = self._template_match(
                    gray,
                    self.bbox,
                    search_radius=global_radius,
                    use_base_only=True,
                )
                if gok and gbb is not None and float(gscore) >= (0.62 if self.fast_mode else 0.65):
                    gpatch = self.feature_extractor.extract_and_flatten(frame, gbb)
                    if gpatch is not None:
                        gsim = self.appearance_model.compute_similarity(gpatch, method=self.similarity_method)
                        if gsim is not None and float(gsim) >= float(max(0.18, adaptive_threshold * 0.8)):
                            self.prev_bbox = self.bbox
                            self.bbox = self._clip_bbox(gbb, frame.shape)
                            self.velocity = (self.velocity[0] * 0.5, self.velocity[1] * 0.5)
                            self.confidence = float(max(self.confidence, gscore, float(gsim)))
                            self.occlusion_count = 0
                            self._set_flow_state(gray, self.bbox, reinit=True)
                            return TrackingResult(
                                bbox=self.bbox,
                                confidence=self.confidence,
                                similarity=self.confidence,
                                is_occluded=False,
                                scale=self.scale,
                            )

            accept_bbox = False
            if best_bbox is not None:
                bx, by, bw, bh = best_bbox
                best_cx = bx + bw // 2
                best_cy = by + bh // 2
                dist = float(np.hypot(best_cx - pred_cx, best_cy - pred_cy))
                # Require a minimum similarity and proximity to prediction
                min_sim = max(0.35, adaptive_threshold)
                max_dist = min(float(search_radius) * 0.5, float(max(15, max(w, h))))
                if best_similarity > min_sim and dist <= max_dist:
                    accept_bbox = True
            
            # If occluded too long, keep last known position with motion
            if self.occlusion_count <= self.max_occlusion_frames:
                # Still use best match even if occluded (unless similarity is very low)
                moved = False
                if accept_bbox:
                    self._update_motion_model(best_bbox)
                    self.prev_bbox = self.bbox
                    self.bbox = self._clip_bbox(best_bbox, frame.shape)
                    self.scale = best_scale
                    self.confidence = best_similarity * 0.5
                    moved = True
                else:
                    moved = False
                    if self.prev_gray is not None and self.prev_points is not None and self.prev_points.shape[0] > 0:
                        try:
                            next_pt, status, _err = cv2.calcOpticalFlowPyrLK(
                                self.prev_gray,
                                gray,
                                self.prev_points,
                                None,
                                winSize=(21, 21),
                                maxLevel=3,
                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                            )
                            if status is not None and next_pt is not None:
                                status_fwd = status.reshape(-1) == 1
                                if int(np.sum(status_fwd)) >= 5:
                                    back_pt, status_bwd, _err_bwd = cv2.calcOpticalFlowPyrLK(
                                        gray,
                                        self.prev_gray,
                                        next_pt,
                                        None,
                                        winSize=(21, 21),
                                        maxLevel=3,
                                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                                    )
                                    if status_bwd is not None and back_pt is not None:
                                        status_bwd = status_bwd.reshape(-1) == 1
                                        prev_xy = self.prev_points[:, 0, :]
                                        back_xy = back_pt[:, 0, :]
                                        fb_err = np.linalg.norm(prev_xy - back_xy, axis=1)
                                        fb_ok = fb_err < 1.5
                                        good = status_fwd & status_bwd & fb_ok

                                        if int(np.sum(good)) >= 5:
                                            prev_good = self.prev_points[good, 0, :]
                                            next_good = next_pt[good, 0, :]
                                            dxy = next_good - prev_good
                                            dx = float(np.median(dxy[:, 0]))
                                            dy = float(np.median(dxy[:, 1]))
                                            max_flow = max(3.0, float(max(w, h)) * 0.5)
                                            max_flow = min(max_flow, float(search_radius) * 0.5)
                                            dx = float(np.clip(dx, -max_flow, max_flow))
                                            dy = float(np.clip(dy, -max_flow, max_flow))
                                            pred_x = int(x + dx)
                                            pred_y = int(y + dy)
                                            self.prev_bbox = self.bbox
                                            self.bbox = self._clip_bbox((pred_x, pred_y, w, h), frame.shape)
                                            self.velocity = (
                                                0.7 * self.velocity[0] + 0.3 * dx,
                                                0.7 * self.velocity[1] + 0.3 * dy,
                                            )
                                            next_points = next_pt[good].reshape(-1, 1, 2).astype(np.float32)
                                            self._set_flow_state(gray, self.bbox, points=next_points, reinit=False)
                                            moved = True
                        except Exception:
                            moved = False

                    if not moved:
                        templ_moved = False
                        tok, tbb, tscore = self._template_match(
                            gray,
                            self.bbox,
                            search_radius=max(30, int(search_radius * 1.5)),
                            use_base_only=True,
                        )
                        if tok and tbb is not None and float(tscore) >= (0.62 if self.fast_mode else 0.65):
                            self.prev_bbox = self.bbox
                            self.bbox = self._clip_bbox(tbb, frame.shape)
                            self.velocity = (self.velocity[0] * 0.6, self.velocity[1] * 0.6)
                            templ_moved = True

                        if not templ_moved:
                            self.prev_bbox = self.bbox
                            self.bbox = (x, y, w, h)
                            self.velocity = (self.velocity[0] * 0.85, self.velocity[1] * 0.85)
                    self.confidence = 0.1

                if not moved:
                    self._set_flow_state(gray, self.bbox, reinit=True)
                elif accept_bbox:
                    self._set_flow_state(gray, self.bbox, reinit=True)
                return TrackingResult(
                    bbox=self.bbox,
                    confidence=self.confidence,
                    similarity=best_similarity if np.isfinite(best_similarity) else 0.0,
                    is_occluded=True,
                    scale=self.scale
                )
            else:
                max_step = max(5.0, float(max(w, h)))
                dx = float(np.clip(self.velocity[0], -max_step, max_step))
                dy = float(np.clip(self.velocity[1], -max_step, max_step))
                pred_x = int(x + dx)
                pred_y = int(y + dy)
                self.prev_bbox = self.bbox
                self.bbox = self._clip_bbox((pred_x, pred_y, w, h), frame.shape)
                self.confidence = 0.0
                self._set_flow_state(gray, self.bbox, reinit=True)
                return TrackingResult(
                    bbox=self.bbox,
                    confidence=self.confidence,
                    similarity=best_similarity if np.isfinite(best_similarity) else 0.0,
                    is_occluded=True,
                    scale=self.scale
                )

        # Good match found
        self.occlusion_count = 0

        if best_bbox is not None:
            if flow_ok and flow_bbox is not None and flow_similarity is not None and best_bbox != flow_bbox:
                if float(best_similarity) < float(flow_similarity) + 0.05:
                    best_bbox = flow_bbox
                    best_patch = flow_patch
                    best_similarity = float(flow_similarity)
                    best_scale = 1.0

            if flow_ok and templ_ok and templ_bbox is not None and best_bbox != templ_bbox:
                if float(best_similarity) < float(templ_score) + 0.03:
                    best_bbox = templ_bbox
                    best_similarity = float(templ_score)
                    best_scale = 1.0

            # Update motion model
            self._update_motion_model(best_bbox)

            # Update state
            self.prev_bbox = self.bbox
            self.bbox = self._clip_bbox(best_bbox, frame.shape)
            self.scale = best_scale

            if (self.frame_count % self._template_update_interval) == 0:
                if self.confidence > 0.75 and not is_occluded:
                    new_templ = self._get_gray_patch(gray, self.bbox)
                    if new_templ is not None and self._template_bbox_size == (self.bbox[2], self.bbox[3]):
                        self._template = new_templ

            # Calculate confidence
            if best_patch is not None:
                self.confidence = self.appearance_model.get_confidence(best_patch)
            else:
                self.confidence = best_similarity

            self.confidence_history.append(self.confidence)

            # Update appearance model more frequently when confident
            should_update = (
                self.frame_count % self.update_rate == 0 and
                best_patch is not None and
                self.confidence > 0.4  # Lower threshold
            )

            if should_update:
                # Adaptive learning rate based on confidence
                adaptive_lr = self.learning_rate if self.confidence > 0.6 else self.learning_rate * 0.5
                self.appearance_model.update(best_patch, learning_rate=adaptive_lr)

            self._set_flow_state(gray, self.bbox, reinit=True)
            return TrackingResult(
                bbox=self.bbox,
                confidence=self.confidence,
                similarity=best_similarity,
                is_occluded=False,
                scale=self.scale
            )

        self._set_flow_state(gray, self.bbox, reinit=True)
        return TrackingResult(
            bbox=self.bbox,
            confidence=0.0,
            similarity=best_similarity if np.isfinite(best_similarity) else 0.0,
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
        self.prev_gray = None
        self.prev_points = None
        
        self.appearance_model = AppearanceModel(
            n_components=self.appearance_model.n_components,
            variance_threshold=self.appearance_model.variance_threshold
        )
    
    def retrain(self):
        """Retrain PCA model with accumulated patches."""
        self.appearance_model.retrain()
