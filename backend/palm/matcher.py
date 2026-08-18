"""
Dual-threshold identity matcher over enrolled palm embeddings.

Implements two thresholds for robust biometric decisioning:
  - MATCH_THRESHOLD_HIGH (e.g. 0.82): High confidence match -> status = "matched" / "paid"
  - MATCH_THRESHOLD_LOW (e.g. 0.70): Borderline match -> status = "borderline" (requires Step-Up PIN)
  - Below LOW threshold: Rejected -> status = "unmatched" / "rejected_mismatch"
"""

import os
import threading
from typing import List, Optional, Tuple

import numpy as np


class PalmMatcher:
    def __init__(
        self,
        threshold_high: Optional[float] = None,
        threshold_low: Optional[float] = None,
    ):
        self.threshold_high = threshold_high if threshold_high is not None else float(
            os.environ.get("MATCH_THRESHOLD_HIGH", "0.82")
        )
        self.threshold_low = threshold_low if threshold_low is not None else float(
            os.environ.get("MATCH_THRESHOLD_LOW", "0.70")
        )
        self._embeddings: List[np.ndarray] = []
        self._customer_ids: List[int] = []
        self._lock = threading.Lock()

    def add(self, customer_id: int, embedding: np.ndarray) -> None:
        """Add unit-normalized customer embedding vector to in-memory index."""
        norm = np.linalg.norm(embedding) or 1.0
        norm_emb = (embedding / norm).astype(np.float32)
        with self._lock:
            if self._embeddings and self._embeddings[0].shape[0] != norm_emb.shape[0]:
                print(
                    f"[!] Matcher warning: Ignoring mismatched {norm_emb.shape[0]}-D embedding "
                    f"(expected {self._embeddings[0].shape[0]}-D)"
                )
                return
            self._embeddings.append(norm_emb)
            self._customer_ids.append(customer_id)

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[int], float, str]:
        """
        Identify top match across enrolled customers.
        Returns: (customer_id, confidence_score, status_label)
        status_label: "matched" | "borderline" | "unmatched"
        """
        with self._lock:
            if not self._embeddings:
                return None, 0.0, "unmatched"
            embeddings_matrix = np.stack(self._embeddings)
            customer_ids = list(self._customer_ids)

        query_norm = np.linalg.norm(embedding) or 1.0
        query = (embedding / query_norm).astype(np.float32)

        if embeddings_matrix.shape[1] != query.shape[0]:
            print(
                f"[!] Matcher query dimension mismatch: matrix is {embeddings_matrix.shape[1]}-D, "
                f"query is {query.shape[0]}-D"
            )
            return None, 0.0, "unmatched"

        sims = embeddings_matrix @ query
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        matched_cid = customer_ids[best_idx]

        if best_score >= self.threshold_high:
            return matched_cid, best_score, "matched"
        elif best_score >= self.threshold_low:
            return matched_cid, best_score, "borderline"
        else:
            return None, best_score, "unmatched"

    def verify(self, customer_id: int, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Verify that a second scan matches the specific customer_id identified in Scan 1.
        Returns: (status_label, confidence_score)
        status_label: "paid" | "borderline" | "rejected_mismatch"
        """
        with self._lock:
            embeddings_list = list(self._embeddings)
            customer_ids = list(self._customer_ids)

        query_norm = np.linalg.norm(embedding) or 1.0
        query = (embedding / query_norm).astype(np.float32)

        best_score = -1.0
        for cid, emb in zip(customer_ids, embeddings_list):
            if cid == customer_id:
                score = float(emb @ query)
                best_score = max(best_score, score)

        if best_score >= self.threshold_high:
            return "paid", best_score
        elif best_score >= self.threshold_low:
            return "borderline", best_score
        else:
            return "rejected_mismatch", best_score
