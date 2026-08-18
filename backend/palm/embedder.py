"""
Palm embedding extractor (Track A - HOG + PCA).

Converts an aligned 224x224 palm ROI into a 128-D normalized embedding vector.
Preprocessing pipeline: Grayscale conversion -> CLAHE (Adaptive Histogram Equalization)
for shadow resilience -> HOG texture descriptor -> PCA projection -> L2 Normalization.
"""

from typing import List, Optional

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA


class PalmEmbedder:
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self._pca: Optional[PCA] = None

    @staticmethod
    def preprocess_clahe(aligned_bgr: np.ndarray) -> np.ndarray:
        """Grayscale conversion & CLAHE adaptive histogram equalization for shadow resilience."""
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _raw_hog_features(self, aligned_bgr: np.ndarray) -> np.ndarray:
        enhanced_gray = self.preprocess_clahe(aligned_bgr)
        features = hog(
            enhanced_gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return features.astype(np.float32)

    def fit(self, aligned_images: List[np.ndarray]) -> None:
        """Fit the PCA projection matrix on a batch of enrollment images."""
        if len(aligned_images) < self.embedding_dim:
            raise ValueError(
                f"Need at least {self.embedding_dim} enrollment images to fit "
                f"a {self.embedding_dim}-D PCA; got {len(aligned_images)}."
            )
        raw = np.stack([self._raw_hog_features(img) for img in aligned_images])
        self._pca = PCA(n_components=self.embedding_dim, whiten=True)
        self._pca.fit(raw)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Computes 128-D normalized embedding vector from aligned ROI image."""
        raw = self._raw_hog_features(aligned_bgr).reshape(1, -1)
        if self._pca is not None:
            vec = self._pca.transform(raw)[0]
        else:
            # Fallback mock projection if PCA joblib file isn't loaded yet
            vec = raw[0][:self.embedding_dim]
        
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self._pca, path)

    def load(self, path: str) -> None:
        import joblib
        self._pca = joblib.load(path)
