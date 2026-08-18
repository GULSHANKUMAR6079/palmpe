"""
evaluate.py — Biometric Evaluation Suite (FAR, FRR, EER, ROC & DET Plotting)

Calculates commercial biometric verification metrics on genuine and impostor pair similarity scores:
  - FAR (False Acceptance Rate)
  - FRR (False Rejection Rate)
  - EER (Equal Error Rate)
  - Generates ROC curve (`roc_curve.png`) and DET curve (`det_curve.png`)
"""

import os
import argparse
from typing import Tuple, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class BiometricEvaluator:
    def __init__(self, thresholds: np.ndarray = np.linspace(0.0, 1.0, 500)):
        self.thresholds = thresholds

    def compute_far_frr(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        far_list = []
        frr_list = []

        for t in self.thresholds:
            frr = float(np.mean(genuine_scores < t)) if len(genuine_scores) > 0 else 0.0
            far = float(np.mean(impostor_scores >= t)) if len(impostor_scores) > 0 else 0.0
            far_list.append(far)
            frr_list.append(frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)

        idx = int(np.argmin(np.abs(far_arr - frr_arr)))
        eer = float((far_arr[idx] + frr_arr[idx]) / 2.0)
        optimal_thresh = float(self.thresholds[idx])

        return far_arr, frr_arr, eer, optimal_thresh

    def plot_roc_curve(self, far: np.ndarray, frr: np.ndarray, eer: float, save_path: str = "roc_curve.png"):
        tar = 1.0 - frr
        plt.figure(figsize=(6, 5))
        plt.plot(far, tar, color='#38BDF8', lw=2, label=f'ROC Curve (EER = {eer*100:.2f}%)')
        plt.plot([0, 1], [0, 1], color='#64748B', linestyle='--', label='Random Chance')
        plt.xscale('log')
        plt.xlim([1e-4, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Acceptance Rate (FAR)', fontsize=10)
        plt.ylabel('True Acceptance Rate (TAR = 1 - FRR)', fontsize=10)
        plt.title('Palm Biometric ROC Curve', fontsize=12, fontweight='bold')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[✓] Saved ROC Curve plot to: {save_path}")

    def plot_det_curve(self, far: np.ndarray, frr: np.ndarray, eer: float, save_path: str = "det_curve.png"):
        plt.figure(figsize=(6, 5))
        plt.plot(far, frr, color='#818CF8', lw=2, label=f'DET Curve (EER = {eer*100:.2f}%)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([1e-3, 1.0])
        plt.ylim([1e-3, 1.0])
        plt.xlabel('False Acceptance Rate (FAR)', fontsize=10)
        plt.ylabel('False Rejection Rate (FRR)', fontsize=10)
        plt.title('Palm Biometric Detection Error Tradeoff (DET)', fontsize=12, fontweight='bold')
        plt.grid(True, which='both', linestyle=':', alpha=0.5)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[✓] Saved DET Curve plot to: {save_path}")

    def evaluate_biometric_system(
        self,
        genuine_scores: np.ndarray,
        impostor_scores: np.ndarray,
        roc_path: str = "roc_curve.png",
        det_path: str = "det_curve.png"
    ) -> Dict[str, float]:
        far, frr, eer, opt_thresh = self.compute_far_frr(genuine_scores, impostor_scores)
        self.plot_roc_curve(far, frr, eer, save_path=roc_path)
        self.plot_det_curve(far, frr, eer, save_path=det_path)

        print("\n========================================================")
        print("          PALMPAY BIOMETRIC EVALUATION REPORT           ")
        print("========================================================")
        print(f" Genuine Pairs Tested:   {len(genuine_scores):,}")
        print(f" Impostor Pairs Tested:  {len(impostor_scores):,}")
        print(f" Equal Error Rate (EER): {eer * 100:.2f}%")
        print(f" Optimal Threshold:      {opt_thresh:.4f}")
        print("========================================================\n")

        return {
            "eer": eer,
            "optimal_threshold": opt_thresh,
        }


if __name__ == "__main__":
    print("[*] Running Biometric Evaluation Suite on Benchmark Pairs...")
    np.random.seed(42)
    genuine = np.clip(np.random.normal(0.85, 0.05, size=1500), 0.0, 1.0)
    impostor = np.clip(np.random.normal(0.40, 0.08, size=15000), 0.0, 1.0)

    evaluator = BiometricEvaluator()
    evaluator.evaluate_biometric_system(genuine, impostor)
