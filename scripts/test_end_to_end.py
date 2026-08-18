"""
End-to-End Integration Test Script for PalmPay.
Verifies the complete REST flow: Register -> Identify -> Set Amount -> Authorize -> Step-Up Verify -> Ledger.
Run with: python scripts/test_end_to_end.py
"""

import sys
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.main import app

client = TestClient(app)


def create_synthetic_palm_image(seed: int = 101) -> bytes:
    """Generates a synthetic palm-like test image buffer."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 180
    # Draw palm center circle
    cv2.circle(img, (320, 260), 80, (140, 140, 140), -1)
    # Draw wrist lines
    cv2.line(img, (270, 400), (370, 400), (90, 90, 90), 4)
    # Draw synthetic palm line creases
    cv2.ellipse(img, (320, 240), (50, 30), 15, 0, 180, (60, 60, 60), 3)
    cv2.line(img, (290, 220), (350, 280), (70, 70, 70), 2)
    # Add noise seed for subtle variation
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def run_e2e_tests():
    print("=========================================================================")
    print("                PalmPay End-to-End Integration Test Suite                ")
    print("=========================================================================")

    # 1. Customer Registration Test
    print("\n[1] Testing POST /customers/register ...")
    img_bytes1 = create_synthetic_palm_image(101)
    img_bytes2 = create_synthetic_palm_image(102)

    files = [
        ('palm_photos', ('frame1.jpg', img_bytes1, 'image/jpeg')),
        ('palm_photos', ('frame2.jpg', img_bytes2, 'image/jpeg'))
    ]
    data = {
        'name': 'Aditya Sharma',
        'contact': '+919876543210',
        'email': 'aditya.sharma@example.com',
        'upi_vpa': 'aditya@hdfcbank',
        'step_up_pin': '4321',
        'consent_given_at': '2026-08-18T10:00:00Z',
        'consent_version': 'v1.0'
    }

    resp = client.post("/customers/register", data=data, files=files)
    print(f"    Response Status: {resp.status_code}")
    print(f"    Response Body: {resp.json()}")
    assert resp.status_code == 200, f"Registration failed: {resp.text}"
    reg_data = resp.json()
    customer_id = reg_data["customer_id"]
    print(f"    [✓] Enrolled Customer ID: #{customer_id}")

    # 2. Customer List Debug Check
    print("\n[2] Testing GET /customers ...")
    resp = client.get("/customers")
    print(f"    Customer Count: {len(resp.json())}")
    assert resp.status_code == 200

    # 3. Identify (Scan 1) Test
    print("\n[3] Testing POST /session/identify ...")
    files_scan1 = {'palm_photo': ('scan1.jpg', img_bytes1, 'image/jpeg')}
    data_scan1 = {'merchant_id': 'merch_store_99'}

    resp = client.post("/session/identify", data=data_scan1, files=files_scan1)
    print(f"    Response Status: {resp.status_code}")
    print(f"    Response Body: {resp.json()}")
    assert resp.status_code == 200
    id_data = resp.json()
    assert id_data["matched"] is True, "Identify scan should match registered customer"
    session_id = id_data["session_id"]
    print(f"    [✓] Session Started ID: #{session_id}")

    # 4. Set Amount Test
    print("\n[4] Testing POST /session/set-amount ...")
    amount_payload = {'session_id': session_id, 'amount_rupees': 45.50}
    resp = client.post("/session/set-amount", json=amount_payload)
    print(f"    Response Status: {resp.status_code}")
    print(f"    Response Body: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # 5. Authorize (Scan 2) Test
    print("\n[5] Testing POST /session/authorize ...")
    files_scan2 = {'palm_photo': ('scan2.jpg', img_bytes1, 'image/jpeg')}
    data_scan2 = {'session_id': session_id}

    resp = client.post("/session/authorize", data=data_scan2, files=files_scan2)
    print(f"    Response Status: {resp.status_code}")
    print(f"    Response Body: {resp.json()}")
    assert resp.status_code == 200
    auth_data = resp.json()
    assert auth_data["status"] in ("paid", "borderline"), f"Authorization status: {auth_data['status']}"

    # 6. Step-Up Verification Test (Fallback PIN)
    print("\n[6] Testing POST /session/step-up-verify ...")
    step_up_payload = {'session_id': session_id, 'secret': '4321'}
    resp = client.post("/session/step-up-verify", json=step_up_payload)
    print(f"    Response Status: {resp.status_code}")
    print(f"    Response Body: {resp.json()}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paid"

    # 7. Merchant Ledger Test
    print("\n[7] Testing GET /transactions ...")
    resp = client.get("/transactions")
    print(f"    Response Status: {resp.status_code}")
    txns = resp.json()["transactions"]
    print(f"    Recorded Transactions: {len(txns)}")
    assert len(txns) >= 1
    print(f"    [✓] Latest Transaction: ID #{txns[0]['id']} | Status: {txns[0]['status']} | Amount: ₹{txns[0]['amount_rupees']}")

    # 8. Receipt PDF Download Test
    print("\n[8] Testing GET /receipts/{transaction_id} ...")
    txn_id = txns[0]['id']
    resp = client.get(f"/receipts/{txn_id}")
    print(f"    Response Status: {resp.status_code}")
    print(f"    Content Type: {resp.headers.get('content-type')}")
    assert resp.status_code == 200
    assert "application/pdf" in resp.headers.get('content-type', '')
    print("    [✓] Receipt PDF successfully generated and verified!")

    print("\n=========================================================================")
    print("        🎉 ALL END-TO-END INTEGRATION TESTS PASSED SUCCESSFULLY!        ")
    print("=========================================================================")


if __name__ == "__main__":
    run_e2e_tests()
