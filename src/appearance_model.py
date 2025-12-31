"""
PCA-based appearance model for object tracking.
Builds and maintains a PCA model to represent object appearance.
"""

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from typing import List, Optional, Tuple
from collections import deque


class AppearanceModel:
    """PCA-based appearance model for object representation."""
    
    def __init__(self, n_components: Optional[int] = None, 
                 variance_threshold: float = 0.95,
                 max_buffer_size: int = 100,
                 use_incremental: bool = False):
        """
        Initialize PCA appearance model.
        
        Args:
            n_components: Number of principal components (None for variance-based)
            variance_threshold: Minimum variance to retain (used if n_components is None)
            max_buffer_size: Maximum number of patches to keep in buffer
            use_incremental: Use incremental PCA for online learning
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.max_buffer_size = max_buffer_size
        self.use_incremental = use_incremental
        
        self.pca: Optional[PCA] = None
        self.mean: Optional[np.ndarray] = None
        self.reference_coefficients: Optional[np.ndarray] = None
        self.is_trained = False
        
        # Patch buffer for model updates
        self.patch_buffer: deque = deque(maxlen=max_buffer_size)
        
        # Statistics for adaptive thresholding
        self.similarity_history: deque = deque(maxlen=50)
        self.reconstruction_error_mean: float = 0.0
        self.reconstruction_error_std: float = 1.0
    
    def train(self, patches: List[np.ndarray]) -> bool:
        """
        Train PCA model on a collection of object patches.
        
        Args:
            patches: List of flattened patch feature vectors
        
        Returns:
            True if training successful, False otherwise
        """
        if not patches or len(patches) == 0:
            return False
        
        # Add to buffer
        for patch in patches:
            self.patch_buffer.append(patch)
        
        # Convert to numpy array
        X = np.array(list(self.patch_buffer))
        
        if X.shape[0] < 2:
            return False
        
        # Determine number of components
        n_components = self.n_components
        if n_components is None:
            n_components = min(X.shape[0] - 1, X.shape[1], 50)
        
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])
        
        if n_components <= 0:
            return False
        
        try:
            if self.use_incremental:
                self.pca = IncrementalPCA(n_components=n_components)
            else:
                self.pca = PCA(n_components=n_components)
            
            self.pca.fit(X)
            
            # If using variance threshold, find optimal components
            if self.n_components is None and not self.use_incremental:
                cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
                optimal_n = np.searchsorted(cumsum_variance, self.variance_threshold) + 1
                optimal_n = max(1, min(optimal_n, len(self.pca.components_)))
                
                if optimal_n < n_components:
                    self.pca = PCA(n_components=optimal_n)
                    self.pca.fit(X)
            
            self.mean = self.pca.mean_
            
            # Set reference from mean of all patches (more robust than first patch)
            self.reference_coefficients = np.mean(self.pca.transform(X), axis=0)
            
            # Calculate initial reconstruction error statistics
            reconstructed = self.pca.inverse_transform(self.pca.transform(X))
            errors = np.mean((X - reconstructed) ** 2, axis=1)
            self.reconstruction_error_mean = np.mean(errors)
            self.reconstruction_error_std = np.std(errors) + 1e-10
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"PCA training failed: {e}")
            return False
    
    def project(self, patch: np.ndarray) -> Optional[np.ndarray]:
        """
        Project a patch onto PCA space.
        
        Args:
            patch: Flattened patch feature vector
        
        Returns:
            PCA coefficients (projection), or None if model not trained
        """
        if not self.is_trained or self.pca is None:
            return None
        
        if patch.ndim == 1:
            patch = patch.reshape(1, -1)
        
        coefficients = self.pca.transform(patch)
        return coefficients[0] if coefficients.shape[0] == 1 else coefficients
    
    def reconstruct(self, coefficients: np.ndarray) -> Optional[np.ndarray]:
        """
        Reconstruct patch from PCA coefficients.
        
        Args:
            coefficients: PCA coefficients
        
        Returns:
            Reconstructed patch, or None if model not trained
        """
        if not self.is_trained or self.pca is None:
            return None
        
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        
        reconstructed = self.pca.inverse_transform(coefficients)
        return reconstructed[0] if reconstructed.shape[0] == 1 else reconstructed
    
    def compute_reconstruction_error(self, patch: np.ndarray) -> Optional[float]:
        """
        Compute reconstruction error for a patch.
        Lower error means better match with learned appearance.
        
        Args:
            patch: Flattened patch feature vector
        
        Returns:
            Reconstruction error (MSE), or None if model not trained
        """
        if not self.is_trained or self.pca is None:
            return None
        
        coefficients = self.project(patch)
        if coefficients is None:
            return None
        
        reconstructed = self.reconstruct(coefficients)
        if reconstructed is None:
            return None
        
        error = np.mean((patch - reconstructed) ** 2)
        return error
    
    def compute_similarity(self, patch: np.ndarray, method: str = 'combined') -> Optional[float]:
        """
        Compute similarity between patch and reference appearance.
        
        Args:
            patch: Flattened patch feature vector
            method: Similarity method ('euclidean', 'correlation', 'reconstruction', 'combined')
        
        Returns:
            Similarity score (higher is more similar), or None if model not trained
        """
        if not self.is_trained or self.reference_coefficients is None:
            return None
        
        coefficients = self.project(patch)
        if coefficients is None:
            return None
        
        if method == 'euclidean':
            distance = np.linalg.norm(coefficients - self.reference_coefficients)
            similarity = 1.0 / (1.0 + distance)
            
        elif method == 'correlation':
            ref_norm = self.reference_coefficients / (np.linalg.norm(self.reference_coefficients) + 1e-10)
            coeff_norm = coefficients / (np.linalg.norm(coefficients) + 1e-10)
            similarity = np.dot(ref_norm, coeff_norm)
            similarity = (similarity + 1) / 2  # Normalize to [0, 1]
            
        elif method == 'mahalanobis':
            if self.pca is None:
                return None
            diff = coefficients - self.reference_coefficients
            cov = np.diag(self.pca.explained_variance_ + 1e-10)
            inv_cov = np.linalg.pinv(cov)
            distance = np.sqrt(np.abs(diff @ inv_cov @ diff))
            similarity = 1.0 / (1.0 + distance)
            
        elif method == 'reconstruction':
            error = self.compute_reconstruction_error(patch)
            if error is None:
                return None
            # Normalize error using running statistics
            normalized_error = (error - self.reconstruction_error_mean) / self.reconstruction_error_std
            similarity = 1.0 / (1.0 + max(0, normalized_error))
            
        elif method == 'combined':
            # Combine PCA distance and reconstruction error
            distance = np.linalg.norm(coefficients - self.reference_coefficients)
            pca_sim = 1.0 / (1.0 + distance)
            
            error = self.compute_reconstruction_error(patch)
            if error is None:
                return pca_sim
            
            normalized_error = (error - self.reconstruction_error_mean) / self.reconstruction_error_std
            recon_sim = 1.0 / (1.0 + max(0, normalized_error))
            
            # Weighted combination
            similarity = 0.7 * pca_sim + 0.3 * recon_sim
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        # Update statistics
        self.similarity_history.append(similarity)
        
        return similarity
    
    def get_confidence(self, patch: np.ndarray) -> float:
        """
        Get tracking confidence based on reconstruction error.
        
        Args:
            patch: Flattened patch feature vector
        
        Returns:
            Confidence score (0-1), higher means more confident
        """
        error = self.compute_reconstruction_error(patch)
        if error is None:
            return 0.5
        
        # Confidence based on how close error is to expected
        z_score = abs(error - self.reconstruction_error_mean) / self.reconstruction_error_std
        confidence = np.exp(-z_score / 2)  # Exponential decay
        
        return float(np.clip(confidence, 0, 1))
    
    def update(self, patch: np.ndarray, learning_rate: float = 0.1):
        """
        Incrementally update the appearance model with a new patch.
        
        Args:
            patch: New patch to incorporate
            learning_rate: Learning rate for updating reference coefficients (0-1)
        """
        if not self.is_trained:
            return
        
        coefficients = self.project(patch)
        if coefficients is None:
            return
        
        # Update reference coefficients using exponential moving average
        self.reference_coefficients = (1 - learning_rate) * self.reference_coefficients + \
                                      learning_rate * coefficients
        
        # Update reconstruction error statistics
        error = self.compute_reconstruction_error(patch)
        if error is not None:
            self.reconstruction_error_mean = (1 - learning_rate) * self.reconstruction_error_mean + \
                                             learning_rate * error
        
        # Add to buffer for potential retraining
        self.patch_buffer.append(patch)
    
    def retrain(self, min_samples: int = 10):
        """
        Retrain PCA model using buffered patches.
        
        Args:
            min_samples: Minimum samples required for retraining
        """
        if len(self.patch_buffer) < min_samples:
            return
        
        patches = list(self.patch_buffer)
        
        # Preserve current reference
        old_reference = self.reference_coefficients.copy() if self.reference_coefficients is not None else None
        
        self.train(patches)
        
        # Blend old and new reference
        if old_reference is not None and self.reference_coefficients is not None:
            # Project old reference to new space (approximate)
            self.reference_coefficients = 0.5 * self.reference_coefficients + 0.5 * old_reference[:len(self.reference_coefficients)]
    
    def get_explained_variance(self) -> Optional[float]:
        """Get total explained variance ratio."""
        if not self.is_trained or self.pca is None:
            return None
        return float(np.sum(self.pca.explained_variance_ratio_))
    
    def get_n_components(self) -> Optional[int]:
        """Get number of principal components."""
        if not self.is_trained or self.pca is None:
            return None
        return self.pca.n_components_
    
    def get_adaptive_threshold(self, percentile: float = 10) -> float:
        """
        Get adaptive similarity threshold based on history.
        
        Args:
            percentile: Percentile for threshold (lower = stricter)
        
        Returns:
            Threshold value
        """
        if len(self.similarity_history) < 5:
            return 0.3  # Default threshold
        
        return float(np.percentile(list(self.similarity_history), percentile))
