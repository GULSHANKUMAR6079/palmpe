"""
fit_pca.py — PCA Fitting Utility for Classical HOG+PCA Embedder

Fits the 128-D PCA projection matrix (`pca.joblib`) used by `PalmEmbedder`
on a directory of sample/bootstrap palm photos or augmented synthetic images.

Usage:
  python fit_pca.py --photos_dir ./bootstrap_photos --out ./pca.joblib
"""

import os
import argparse
import cv2
import numpy as np
from backend.palm.embedder import PalmEmbedder
from backend.palm.augment import augment_palm_image


def generate_synthetic_bootstrap(n_palms: int = 25, imgs_per_palm: int = 6) -> list:
    """Generates synthetic aligned palm images to fit PCA if no dataset folder is supplied."""
    print(f"[*] Generating {n_palms * imgs_per_palm} synthetic aligned palm images for PCA initialization...")
    aligned_images = []
    rng = np.random.default_rng(42)

    for p in range(n_palms):
        base = np.full((224, 224, 3), 160, dtype=np.uint8)
        # Unique line structure per palm
        pts = rng.integers(30, 190, size=(4, 2))
        cv2.polylines(base, [pts], False, (50, 50, 50), 3)

        for i in range(imgs_per_palm):
            aug_list = augment_palm_image(base, n_variants=1, seed=p * 10 + i)
            aligned_images.append(aug_list[0])

    return aligned_images


def main():
    parser = argparse.ArgumentParser(description="Fit PCA projection matrix for PalmEmbedder")
    parser.add_argument("--photos_dir", type=str, default=None, help="Directory containing palm photo images")
    parser.add_argument("--out", type=str, default="pca.joblib", help="Output path for fitted PCA file")
    args = parser.parse_args()

    aligned_imgs = []

    if args.photos_dir and os.path.exists(args.photos_dir):
        print(f"[*] Reading palm images from directory: {args.photos_dir}...")
        for fname in os.listdir(args.photos_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(args.photos_dir, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    if img.shape[:2] != (224, 224):
                        img = cv2.resize(img, (224, 224))
                    aligned_imgs.append(img)
        print(f"[*] Loaded {len(aligned_imgs)} photos from directory.")

    if len(aligned_imgs) < 128:
        print("[*] Less than 128 photos provided. Generating augmented bootstrap set...")
        syn_imgs = generate_synthetic_bootstrap(n_palms=25, imgs_per_palm=6)
        aligned_imgs.extend(syn_imgs)

    print(f"[*] Fitting 128-D PCA matrix on {len(aligned_imgs)} aligned images...")
    embedder = PalmEmbedder(embedding_dim=128)
    embedder.fit(aligned_imgs)
    embedder.save(args.out)
    print(f"[✓] Successfully fitted and saved PCA matrix to: {args.out}")


if __name__ == "__main__":
    main()
