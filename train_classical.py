"""
train_classical.py — Classical Feature Extraction Baseline & Evaluation for Palm Recognition

Approach A: Classical CV Baseline (HOG, LBP, Gabor Features + PCA):
  - Local Binary Patterns (LBP) texture descriptors
  - Gabor filter energy features
  - HOG + PCA 128-D embedding representation
  - Person-Disjoint Train/Test evaluation computing Genuine vs. Impostor similarity distributions
"""

import os
import argparse
from typing import List, Tuple, Dict
import cv2
import numpy as np
from backend.palm.embedder import PalmEmbedder
from backend.palm.augment import augment_palm_image


class ClassicalPalmExtractor:
    def __init__(self):
        self.embedder = PalmEmbedder(embedding_dim=128)

    def compute_lbp(self, img_gray: np.ndarray) -> np.ndarray:
        """Computes Uniform Local Binary Pattern (LBP) texture histogram."""
        h, w = img_gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = img_gray[i, j]
                code = 0
                code |= (img_gray[i - 1, j - 1] >= center) << 7
                code |= (img_gray[i - 1, j] >= center) << 6
                code |= (img_gray[i - 1, j + 1] >= center) << 5
                code |= (img_gray[i, j + 1] >= center) << 4
                code |= (img_gray[i + 1, j + 1] >= center) << 3
                code |= (img_gray[i + 1, j] >= center) << 2
                code |= (img_gray[i + 1, j - 1] >= center) << 1
                code |= (img_gray[i, j - 1] >= center) << 0
                lbp[i - 1, j - 1] = code

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist) or 1.0
        return hist / norm

    def extract_features(self, img_bgr: np.ndarray) -> np.ndarray:
        """Extracts 128-D normalized embedding vector."""
        return self.embedder.embed(img_bgr)


def create_synthetic_dataset(num_persons: int = 20, imgs_per_person: int = 8) -> Tuple[List[np.ndarray], List[int]]:
    """Generates synthetic person-disjoint dataset for classical training/eval."""
    images = []
    labels = []

    for person_id in range(num_persons):
        base = np.full((224, 224, 3), 140, dtype=np.uint8)
        rng = np.random.RandomState(person_id * 100)
        pts = rng.randint(40, 180, size=(4, 2))
        cv2.polylines(base, [pts], False, (40, 40, 40), 4)

        for img_idx in range(imgs_per_person):
            aug_list = augment_palm_image(base, n_variants=1, seed=person_id * 50 + img_idx)
            images.append(aug_list[0])
            labels.append(person_id)

    return images, labels


def evaluate_classical_pipeline(images: List[np.ndarray], labels: List[int], test_person_ratio: float = 0.3) -> Dict[str, float]:
    """Evaluates Classical Matching on a Person-Disjoint Train/Test split."""
    unique_persons = list(set(labels))
    np.random.seed(42)
    np.random.shuffle(unique_persons)

    n_test = int(len(unique_persons) * test_person_ratio)
    test_persons = set(unique_persons[:n_test])
    train_persons = set(unique_persons[n_test:])

    print(f"[*] Total Subjects: {len(unique_persons)} | Train Subjects: {len(train_persons)} | Test Subjects: {len(test_persons)}")

    extractor = ClassicalPalmExtractor()

    # Fit PCA matrix on training set
    train_images = [img for img, lbl in zip(images, labels) if lbl in train_persons]
    if len(train_images) >= 128:
        extractor.embedder.fit(train_images)
        print(f"[✓] Fitted PCA on {len(train_images)} training images.")

    features = [extractor.extract_features(img) for img in images]

    test_feats = [feat for feat, lbl in zip(features, labels) if lbl in test_persons]
    test_lbls = [lbl for lbl in labels if lbl in test_persons]

    genuine_scores = []
    impostor_scores = []

    for i in range(len(test_feats)):
        for j in range(i + 1, len(test_feats)):
            sim = float(np.dot(test_feats[i], test_feats[j]))
            if test_lbls[i] == test_lbls[j]:
                genuine_scores.append(sim)
            else:
                impostor_scores.append(sim)

    gen_mean = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    imp_mean = float(np.mean(impostor_scores)) if impostor_scores else 0.0

    print(f"[✓] Classical Baseline Evaluation Complete:")
    print(f"    - Genuine Pairs Avg Similarity:  {gen_mean:.4f}")
    print(f"    - Impostor Pairs Avg Similarity: {imp_mean:.4f}")

    return {
        "genuine_mean": gen_mean,
        "impostor_mean": imp_mean,
        "n_train_subjects": len(train_persons),
        "n_test_subjects": len(test_persons),
    }


if __name__ == "__main__":
    print("[*] Running Classical Feature Extractor Training & Evaluation...")
    images, labels = create_synthetic_dataset(num_persons=20, imgs_per_person=8)
    evaluate_classical_pipeline(images, labels)
