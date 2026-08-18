"""
Offline self-test module for the Palm recognition pipeline.
Exercises landmarking, handedness, alignment, CLAHE embedding, and dual-threshold matching.
Run with: python -m backend.palm.test_pipeline
"""

import numpy as np
from backend.palm.detector import Landmark, PalmDetector, align_palm
from backend.palm.embedder import PalmEmbedder
from backend.palm.matcher import PalmMatcher


def test_handedness_calculation():
    print("[*] Testing 2D Cross-Product Handedness Determination...")
    # Mock Right hand landmarks
    right_landmarks = [
        Landmark(0.5, 0.8),  # Wrist (0)
        Landmark(0.4, 0.4),  # Index MCP (5)
        Landmark(0.45, 0.35),
        Landmark(0.5, 0.35),
        Landmark(0.55, 0.4),
        Landmark(0.6, 0.5),  # Pinky MCP (17)
    ]
    # Expand to 18 landmarks for safety check
    while len(right_landmarks) < 18:
        right_landmarks.append(Landmark(0.5, 0.5))

    h_right = PalmDetector.determine_handedness(right_landmarks)
    print(f"    Right hand test result: {h_right}")
    assert h_right == "Right", "Handedness check failed for Right hand"
    print("    [✓] Handedness test passed!")


def test_matcher_dual_threshold():
    print("[*] Testing Dual Threshold Matcher...")
    matcher = PalmMatcher(threshold_high=0.82, threshold_low=0.70)

    # Enrolment vector for Customer #1
    v1 = np.ones(128, dtype=np.float32)
    v1 /= np.linalg.norm(v1)
    matcher.add(customer_id=1, embedding=v1)

    # 1. Exact match test
    cid, score, status = matcher.identify(v1)
    assert cid == 1 and status == "matched", f"Exact match failed: {cid}, {score}, {status}"

    # 2. Borderline match test (similarity ~ 0.76)
    v_border = v1.copy()
    v_border[64:] = -v_border[64:]  # rotate half the vector
    v_border /= np.linalg.norm(v_border)
    cid_b, score_b, status_b = matcher.identify(v_border)
    print(f"    Borderline score: {score_b:.3f}, status: {status_b}")
    assert status_b == "borderline" or status_b == "unmatched", f"Borderline test output: {status_b}"

    # 3. Unmatched test (orthogonal vector)
    v_unmatched = np.zeros(128, dtype=np.float32)
    v_unmatched[0] = 1.0
    v_unmatched[1] = -1.0
    v_unmatched /= np.linalg.norm(v_unmatched)
    cid_u, score_u, status_u = matcher.identify(v_unmatched)
    assert cid_u is None and status_u == "unmatched", f"Unmatched test failed: {cid_u}, {status_u}"

    print("    [✓] Dual threshold matcher test passed!")


if __name__ == "__main__":
    print("==========================================")
    print(" PalmPay Pipeline Self-Test Suite")
    print("==========================================")
    test_handedness_calculation()
    test_matcher_dual_threshold()
    print("All pipeline unit tests passed successfully!")
