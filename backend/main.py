"""
PalmPay FastAPI Main Application.
Implements the exact REST + WebSocket API contract defined in INTEGRATION.md.
"""

import hashlib
import io
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, Request, Response,
    UploadFile, WebSocket, WebSocketDisconnect, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.models import Customer, PalmEmbedding, Transaction, TransactionStatus
from backend.palm.augment import augment_palm_image
from backend.palm.detector import PalmDetector, align_palm
from backend.palm.embedder import PalmEmbedder
from backend.palm.matcher import PalmMatcher
from backend.payments.razorpay_client import RazorpayMandateClient
from backend.receipt import generate_receipt, mask_vpa
from backend.schemas import (
    AuthorizeResponse, CustomerStateResponse, CustomerUpdateRequest,
    IdentifyResponse, MandateApprovedRequest, PairingTokenResponse,
    RegisterResponse, SetAmountRequest, SetAmountResponse, StepUpVerifyRequest,
    TransactionItem, TransactionListResponse,
)
from dotenv import load_dotenv

load_dotenv()

# Ensure Database Tables Exist
Base.metadata.create_all(bind=engine)

# Config & Paths
RECEIPTS_DIR = os.environ.get("RECEIPTS_DIR", "./receipts")
MODEL_PATH = os.environ.get("HAND_LANDMARKER_MODEL", "hand_landmarker.task")
if not os.path.exists(MODEL_PATH) and os.path.exists(r"d:\Cognizant\palm-pay\hand_landmarker.task"):
    MODEL_PATH = r"d:\Cognizant\palm-pay\hand_landmarker.task"

PCA_PATH = os.environ.get("PALM_PCA_PATH", "./pca.joblib")
if not os.path.exists(PCA_PATH) and os.path.exists(r"d:\Cognizant\palm-pay\pca.joblib"):
    PCA_PATH = r"d:\Cognizant\palm-pay\pca.joblib"
PAYMENT_CAP_RUPEES = 100.0
TEMPLATES_PER_CUSTOMER = 5

# Validation Regexes
EMAIL_REGEX = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
PHONE_REGEX = r"^[0-9]{10}$"
VPA_REGEX = r"^[\w.-]+@[\w.-]+$"

app = FastAPI(
    title="PalmPay Biometric Micro-Payment Backend",
    description="RBI-Compliant Palm-Print Biometric Autopay Terminal",
    version="1.0.0"
)

# CORS Configuration
cors_origins_str = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000")
origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Pipeline Components
detector = PalmDetector(model_path=MODEL_PATH)
embedder = PalmEmbedder(embedding_dim=128)
if os.path.exists(PCA_PATH):
    try:
        embedder.load(PCA_PATH)
        print(f"[*] Loaded PCA projection from {PCA_PATH}")
    except Exception as e:
        print(f"[!] Warning: Could not load PCA file {PCA_PATH}: {e}")

matcher = PalmMatcher()
razorpay_client = RazorpayMandateClient()
AUTO_APPROVE_MANDATE = os.environ.get("AUTO_APPROVE_MANDATE", "true").lower() == "true"


@app.on_event("startup")
def startup_load_embeddings():
    """Load existing enrolled customer embeddings into memory index at startup."""
    db = next(get_db())
    count = 0
    for emb_row in db.query(PalmEmbedding).all():
        vec = np.array(emb_row.vector, dtype=np.float32)
        matcher.add(customer_id=emb_row.customer_id, embedding=vec)
        count += 1
    print(f"[*] Matcher loaded {count} customer palm embeddings into memory.")


def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode('utf-8')).hexdigest()


def _read_upload_as_bgr(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode uploaded image bytes")
    return img


def _detect_align_embed(file_bytes: bytes):
    frame = _read_upload_as_bgr(file_bytes)
    landmarks, handedness, score = detector.detect_with_meta(frame)
    if landmarks is None:
        raise HTTPException(422, "No hand detected in camera frame")

    is_open, open_msg = detector.is_open_palm(landmarks)
    if not is_open:
        raise HTTPException(422, open_msg)

    is_live, liveness_score, live_msg = detector.verify_liveness(frame, landmarks)
    if not is_live:
        raise HTTPException(422, f"🫀 Liveness Check Failed: {live_msg}")

    aligned = align_palm(frame, landmarks)
    if aligned is None:
        raise HTTPException(422, "Please hold your hand vertically upright inside posture guide")

    embedding = embedder.embed(aligned)
    return embedding, handedness, liveness_score, frame, aligned


# ---------------------------------------------------------------------------
# 2.1 POST /customers/register — Enrollment
# ---------------------------------------------------------------------------

@app.post("/customers/register", response_model=RegisterResponse)
def register_customer(
    name: str = Form(...),
    contact: str = Form(...),
    email: str = Form(...),
    upi_vpa: str = Form(...),
    step_up_pin: Optional[str] = Form(None),
    consent_given_at: Optional[str] = Form(None),
    consent_version: Optional[str] = Form("v1.0"),
    palm_photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    name_clean = name.strip()
    contact_clean = contact.strip()
    email_clean = email.strip().lower()
    upi_vpa_clean = upi_vpa.strip()

    if not name_clean:
        raise HTTPException(400, "Name is required")
    if not re.match(EMAIL_REGEX, email_clean):
        raise HTTPException(400, "Invalid email address format")
    if not re.match(PHONE_REGEX, contact_clean):
        raise HTTPException(400, "Mobile number must be digits only and exactly 10 digits")
    if not re.match(VPA_REGEX, upi_vpa_clean):
        raise HTTPException(400, "Invalid UPI ID format (e.g. name@bank)")

    existing = db.query(Customer).filter(
        (Customer.email == email_clean) | (Customer.contact == contact_clean)
    ).first()
    if existing:
        if existing.email == email_clean:
            raise HTTPException(400, f"Customer with email '{email_clean}' already exists")
        else:
            raise HTTPException(400, f"Customer with phone '{contact_clean}' already exists")

    if not palm_photos or len(palm_photos) < 1:
        raise HTTPException(400, "Please upload at least 1 palm photo for enrollment")

    raw_frames = [photo.file.read() for photo in palm_photos]
    bgr_frames = [_read_upload_as_bgr(b) for b in raw_frames]

    # Multi-frame motion parallax liveness check
    if len(bgr_frames) > 1:
        is_mf_live, mf_msg = detector.verify_multiframe_liveness(bgr_frames)
        if not is_mf_live:
            raise HTTPException(422, f"🫀 Multi-frame Liveness Check Failed: {mf_msg}")

    aligned_real = []
    primary_handedness = "Right"

    for frame in bgr_frames:
        landmarks, handedness, score = detector.detect_with_meta(frame)
        if landmarks is None:
            raise HTTPException(status_code=422, detail="No hand detected in uploaded palm photo")

        is_open, open_msg = detector.is_open_palm(landmarks)
        if not is_open:
            raise HTTPException(status_code=422, detail=f"Registration Posture Failed: {open_msg}")

        is_live, liveness_score, live_msg = detector.verify_liveness(frame, landmarks)
        if not is_live:
            raise HTTPException(status_code=422, detail=f"Registration Liveness Failed: {live_msg}")

        aligned = align_palm(frame, landmarks)
        if aligned is None:
            raise HTTPException(status_code=422, detail="Please hold your hand vertically upright inside posture guide")

        aligned_real.append(aligned)
        primary_handedness = handedness

    # Top up templates with augmented variations if fewer than TEMPLATES_PER_CUSTOMER
    templates = list(aligned_real)
    if len(templates) < TEMPLATES_PER_CUSTOMER:
        needed = TEMPLATES_PER_CUSTOMER - len(templates)
        per_photo = -(-needed // len(aligned_real))
        for i, img in enumerate(aligned_real):
            templates.extend(augment_palm_image(img, n_variants=per_photo, seed=i))
        templates = templates[:max(TEMPLATES_PER_CUSTOMER, len(aligned_real))]

    embeddings = [embedder.embed(img) for img in templates]

    # Biometric Deduplication Check
    dedup_cid, dedup_score, dedup_status = matcher.identify(embeddings[0])
    if dedup_cid is not None and dedup_status in ("matched", "borderline"):
        existing_cust = db.query(Customer).get(dedup_cid)
        cust_name = existing_cust.name if existing_cust else f"#{dedup_cid}"
        raise HTTPException(
            status_code=400,
            detail=f"⚠️ BIOMETRIC DEDUPLICATION FAILED: This palm is ALREADY enrolled under '{cust_name}' "
            f"(Confidence: {dedup_score*100:.1f}%)!"
        )

    try:
        dt_consent = datetime.fromisoformat(consent_given_at) if consent_given_at else datetime.utcnow()

        customer = Customer(
            name=name_clean,
            contact=contact_clean,
            email=email_clean,
            upi_vpa=upi_vpa_clean,
            step_up_pin_hash=None,
            consent_given_at=dt_consent,
            consent_version=consent_version,
            registered_handedness=primary_handedness,
        )
        db.add(customer)
        db.flush()

        rp_customer_id = razorpay_client.create_customer(
            name=name_clean, contact=contact_clean, email=email_clean
        )
        mandate_order_id = razorpay_client.create_mandate_order(
            razorpay_customer_id=rp_customer_id,
            mandate_limit_paise=int(PAYMENT_CAP_RUPEES * 100),
        )

        customer.razorpay_customer_id = rp_customer_id
        customer.mandate_order_id = mandate_order_id
        if AUTO_APPROVE_MANDATE:
            customer.mandate_token_id = f"mock_token_{customer.id}"

        for vec in embeddings:
            db.add(PalmEmbedding(customer_id=customer.id, vector=vec.tolist()))

        db.commit()
        db.refresh(customer)

        for vec in embeddings:
            matcher.add(customer_id=customer.id, embedding=vec)

        return RegisterResponse(
            customer_id=customer.id,
            mandate_order_id=mandate_order_id,
            message="Palm enrolled successfully. Mandate approval required."
        )
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Registration failed during database/gateway setup: {str(e)}")


# ---------------------------------------------------------------------------
# 2.2 POST /session/identify — Scan 1
# ---------------------------------------------------------------------------

@app.post("/session/identify", response_model=IdentifyResponse)
def identify(
    merchant_id: str = Form(...),
    palm_photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    embedding, handedness, liveness_score, raw_frame, aligned_roi = _detect_align_embed(palm_photo.file.read())
    cid, confidence, status_label = matcher.identify(embedding)

    if cid is None or status_label == "unmatched":
        return IdentifyResponse(
            matched=False,
            status="unmatched",
            requires_step_up=False,
            confidence=confidence,
            message="Palm not recognized. Please retry or register."
        )

    customer = db.query(Customer).get(cid)
    if not customer:
        return IdentifyResponse(matched=False, status="unmatched", confidence=confidence)

    if customer.registered_handedness and handedness and customer.registered_handedness != handedness:
        raise HTTPException(
            status_code=400,
            detail=f"⚠️ HANDEDNESS MISMATCH: Customer '{customer.name}' registered with {customer.registered_handedness} hand, "
            f"but presented {handedness} hand! Please scan with your {customer.registered_handedness} hand."
        )

    txn = Transaction(
        customer_id=customer.id,
        merchant_id=merchant_id,
        status=TransactionStatus.IDENTIFIED,
        identify_confidence=confidence,
        identify_handedness=handedness,
        identify_embedding=embedding.tolist(),
    )
    db.add(txn)
    db.commit()
    db.refresh(txn)

    requires_step_up = (status_label == "borderline")
    step_up_prompt = "One-Touch Verification Required" if requires_step_up else None

    return IdentifyResponse(
        matched=True,
        status=status_label,
        requires_step_up=requires_step_up,
        step_up_prompt=step_up_prompt,
        customer_id=customer.id,
        name=customer.name,
        masked_upi=mask_vpa(customer.upi_vpa),
        confidence=confidence,
        session_id=txn.id,
        handedness=handedness,
    )


# ---------------------------------------------------------------------------
# 2.3 POST /session/set-amount
# ---------------------------------------------------------------------------

@app.post("/session/set-amount", response_model=SetAmountResponse)
def set_amount(req: SetAmountRequest, db: Session = Depends(get_db)):
    txn = db.query(Transaction).get(req.session_id)
    if not txn or txn.status != TransactionStatus.IDENTIFIED:
        raise HTTPException(status_code=409, detail="Session not in a state that accepts an amount")

    if req.amount_rupees <= 0 or req.amount_rupees > PAYMENT_CAP_RUPEES:
        raise HTTPException(status_code=400, detail=f"Amount must be between Rs 0 and Rs {PAYMENT_CAP_RUPEES}")

    txn.amount_rupees = req.amount_rupees
    txn.status = TransactionStatus.AMOUNT_SET
    db.commit()
    return SetAmountResponse(ok=True, amount_rupees=req.amount_rupees)


# ---------------------------------------------------------------------------
# 2.4 POST /session/authorize — Scan 2
# ---------------------------------------------------------------------------

@app.post("/session/authorize", response_model=AuthorizeResponse)
def authorize(
    session_id: int = Form(...),
    palm_photo: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    txn = db.query(Transaction).get(session_id)
    if not txn or txn.status not in (TransactionStatus.AMOUNT_SET, TransactionStatus.REJECTED_MISMATCH, TransactionStatus.BORDERLINE):
        raise HTTPException(status_code=409, detail="Session not ready for authorization")

    embedding, handedness, liveness_score, raw_frame, aligned_roi = _detect_align_embed(palm_photo.file.read())

    # 1. Strict Handedness Check
    if txn.identify_handedness and handedness and txn.identify_handedness != handedness:
        txn.status = TransactionStatus.REJECTED_MISMATCH
        db.commit()
        return AuthorizeResponse(
            status="rejected_mismatch",
            reason=f"⚠️ HANDEDNESS MISMATCH! Step 1 used {txn.identify_handedness} hand, but authorization used {handedness} hand."
        )

    # 2. Strict Customer Identity Lock
    scanned_cid, cid_score, scanned_status = matcher.identify(embedding)
    if scanned_cid is not None and scanned_cid != txn.customer_id:
        step1_cust = db.query(Customer).get(txn.customer_id)
        step1_name = step1_cust.name if step1_cust else f"#{txn.customer_id}"
        txn.status = TransactionStatus.REJECTED_MISMATCH
        db.commit()
        return AuthorizeResponse(
            status="rejected_mismatch",
            reason=f"⚠️ DIFFERENT USER DETECTED! Session started by '{step1_name}', but hand belongs to another user."
        )

    # 3. Session Vector Similarity
    if txn.identify_embedding:
        step1_vec = np.array(txn.identify_embedding, dtype=np.float32)
        step1_norm = step1_vec / (np.linalg.norm(step1_vec) or 1.0)
        step3_norm = embedding / (np.linalg.norm(embedding) or 1.0)
        session_sim = float(step1_norm @ step3_norm)
        if session_sim < 0.68:
            txn.status = TransactionStatus.REJECTED_MISMATCH
            db.commit()
            return AuthorizeResponse(
                status="rejected_mismatch",
                reason=f"⚠️ PALM MISMATCH (Similarity {session_sim:.2f})! Step 3 hand does not match Step 1 hand."
            )

    verify_status, verify_score = matcher.verify(txn.customer_id, embedding)
    txn.authorize_confidence = verify_score

    if verify_status == "borderline":
        txn.status = TransactionStatus.BORDERLINE
        db.commit()
        return AuthorizeResponse(
            status="borderline",
            requires_step_up=True,
            step_up_prompt="One-Touch Verification Required",
            reason="Borderline biometric match score. One-touch authorization required."
        )

    if verify_status != "paid":
        txn.status = TransactionStatus.REJECTED_MISMATCH
        db.commit()
        return AuthorizeResponse(
            status="rejected_mismatch",
            reason="⚠️ PALM MISMATCH — PAYMENT REJECTED! Hand presented does not match customer identified in Step 1."
        )

    if not txn.amount_rupees or txn.amount_rupees <= 0:
        raise HTTPException(status_code=400, detail="Payment amount has not been confirmed in Step 2. Please confirm amount first.")

    # Execute Payment Charge
    customer = db.query(Customer).get(txn.customer_id)
    token_id = customer.mandate_token_id or f"mock_token_{customer.id}"

    try:
        payment = razorpay_client.charge_with_token(
            token_id=token_id,
            razorpay_customer_id=customer.razorpay_customer_id or f"cust_{customer.id}",
            amount_rupees=txn.amount_rupees,
            mandate_limit_paise=customer.mandate_limit_paise,
        )
    except Exception as e:
        txn.status = TransactionStatus.FAILED
        db.commit()
        return AuthorizeResponse(status="failed", reason=str(e))

    txn.status = TransactionStatus.PAID
    txn.razorpay_payment_id = payment["id"]

    receipt_path = generate_receipt(
        out_dir=RECEIPTS_DIR,
        transaction_id=txn.id,
        customer_name=customer.name,
        masked_upi=mask_vpa(customer.upi_vpa),
        amount_rupees=txn.amount_rupees,
        merchant_id=txn.merchant_id,
        razorpay_payment_id=payment["id"],
    )
    txn.receipt_path = receipt_path
    db.commit()

    return AuthorizeResponse(
        status="paid",
        amount_rupees=txn.amount_rupees,
        razorpay_payment_id=payment["id"],
        receipt_url=f"/receipts/{txn.id}",
    )


# ---------------------------------------------------------------------------
# 2.5 POST /session/step-up-verify — Fallback PIN verification
# ---------------------------------------------------------------------------

@app.post("/session/step-up-verify", response_model=AuthorizeResponse)
def step_up_verify(req: StepUpVerifyRequest, db: Session = Depends(get_db)):
    txn = db.query(Transaction).get(req.session_id)
    if not txn or txn.status not in (TransactionStatus.IDENTIFIED, TransactionStatus.AMOUNT_SET, TransactionStatus.BORDERLINE):
        raise HTTPException(status_code=409, detail="Session not in a state allowing step-up verification")

    customer = db.query(Customer).get(txn.customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    if not txn.amount_rupees or txn.amount_rupees <= 0:
        raise HTTPException(status_code=400, detail="Payment amount has not been confirmed in Step 2. Please confirm amount first.")

    # Proceed to charge payment
    token_id = customer.mandate_token_id or f"mock_token_{customer.id}"
    amount = txn.amount_rupees

    try:
        payment = razorpay_client.charge_with_token(
            token_id=token_id,
            razorpay_customer_id=customer.razorpay_customer_id or f"cust_{customer.id}",
            amount_rupees=amount,
            mandate_limit_paise=customer.mandate_limit_paise,
        )
    except Exception as e:
        txn.status = TransactionStatus.FAILED
        db.commit()
        return AuthorizeResponse(status="failed", reason=str(e))

    txn.status = TransactionStatus.PAID
    txn.razorpay_payment_id = payment["id"]

    receipt_path = generate_receipt(
        out_dir=RECEIPTS_DIR,
        transaction_id=txn.id,
        customer_name=customer.name,
        masked_upi=mask_vpa(customer.upi_vpa),
        amount_rupees=amount,
        merchant_id=txn.merchant_id,
        razorpay_payment_id=payment["id"],
    )
    txn.receipt_path = receipt_path
    db.commit()

    return AuthorizeResponse(
        status="paid",
        amount_rupees=amount,
        razorpay_payment_id=payment["id"],
        receipt_url=f"/receipts/{txn.id}",
    )


# ---------------------------------------------------------------------------
# 2.6 GET /transactions — Merchant Ledger
# ---------------------------------------------------------------------------

@app.get("/transactions", response_model=TransactionListResponse)
def list_transactions(db: Session = Depends(get_db)):
    txns = db.query(Transaction).order_by(Transaction.created_at.desc()).all()
    items = []
    for t in txns:
        c_name = t.customer.name if t.customer else "Unknown"
        c_upi = mask_vpa(t.customer.upi_vpa) if t.customer else "N/A"
        m_token = t.customer.mandate_token_id if t.customer else None
        items.append(TransactionItem(
            id=t.id,
            created_at=t.created_at,
            customer_name=c_name,
            masked_upi=c_upi,
            amount_rupees=t.amount_rupees,
            status=t.status.value if hasattr(t.status, 'value') else str(t.status),
            razorpay_payment_id=t.razorpay_payment_id,
            mandate_token_id=m_token,
            authorize_confidence=t.authorize_confidence or t.identify_confidence,
        ))
    return TransactionListResponse(transactions=items)


# ---------------------------------------------------------------------------
# 2.7 Admin Customer CRUD Routes
# ---------------------------------------------------------------------------

@app.get("/customers", response_model=List[CustomerStateResponse])
def list_customers(db: Session = Depends(get_db)):
    customers = db.query(Customer).all()
    res = []
    for c in customers:
        res.append(CustomerStateResponse(
            id=c.id,
            name=c.name,
            contact=c.contact,
            email=c.email,
            masked_upi=mask_vpa(c.upi_vpa),
            razorpay_customer_id=c.razorpay_customer_id,
            mandate_order_id=c.mandate_order_id,
            mandate_token_id=c.mandate_token_id,
            mandate_approved=c.mandate_token_id is not None,
            embedding_count=len(c.embeddings),
            created_at=c.created_at,
        ))
    return res


@app.get("/customers/{customer_id}", response_model=CustomerStateResponse)
def get_customer(customer_id: int, db: Session = Depends(get_db)):
    c = db.query(Customer).get(customer_id)
    if not c:
        raise HTTPException(404, "Customer not found")
    return CustomerStateResponse(
        id=c.id,
        name=c.name,
        contact=c.contact,
        email=c.email,
        masked_upi=mask_vpa(c.upi_vpa),
        razorpay_customer_id=c.razorpay_customer_id,
        mandate_order_id=c.mandate_order_id,
        mandate_token_id=c.mandate_token_id,
        mandate_approved=c.mandate_token_id is not None,
        embedding_count=len(c.embeddings),
        created_at=c.created_at,
    )


@app.put("/customers/{customer_id}", response_model=CustomerStateResponse)
def update_customer(customer_id: int, req: CustomerUpdateRequest, db: Session = Depends(get_db)):
    c = db.query(Customer).get(customer_id)
    if not c:
        raise HTTPException(404, "Customer not found")

    if req.name:
        c.name = req.name.strip()
    if req.contact:
        contact_clean = req.contact.strip()
        if not re.match(PHONE_REGEX, contact_clean):
            raise HTTPException(400, "Mobile number must be digits only and exactly 10 digits")
        c.contact = contact_clean
    if req.email:
        c.email = req.email.strip().lower()
    if req.upi_vpa:
        c.upi_vpa = req.upi_vpa.strip()
    if req.step_up_pin:
        pass

    db.commit()
    db.refresh(c)

    return CustomerStateResponse(
        id=c.id,
        name=c.name,
        contact=c.contact,
        email=c.email,
        masked_upi=mask_vpa(c.upi_vpa),
        razorpay_customer_id=c.razorpay_customer_id,
        mandate_order_id=c.mandate_order_id,
        mandate_token_id=c.mandate_token_id,
        mandate_approved=c.mandate_token_id is not None,
        embedding_count=len(c.embeddings),
        created_at=c.created_at,
    )


@app.delete("/customers/{customer_id}")
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    c = db.query(Customer).get(customer_id)
    if not c:
        raise HTTPException(404, "Customer not found")

    # Retain transaction audit records by setting customer_id to null
    db.query(Transaction).filter(Transaction.customer_id == customer_id).update({"customer_id": None})
    db.delete(c)
    db.commit()

    # Re-build in-memory matcher index
    global matcher
    matcher = PalmMatcher()
    startup_load_embeddings()

    return {"ok": True, "message": f"Customer #{customer_id} deleted. Embeddings purged."}


@app.get("/receipts/{transaction_id}")
def get_receipt(transaction_id: int, db: Session = Depends(get_db)):
    txn = db.query(Transaction).get(transaction_id)
    if not txn or not txn.receipt_path or not os.path.exists(txn.receipt_path):
        raise HTTPException(404, "Receipt PDF not found")
    return FileResponse(txn.receipt_path, media_type="application/pdf")


# ---------------------------------------------------------------------------
# 2.8 WebSocket Relay Hub & Pairing
# ---------------------------------------------------------------------------

active_pairings: Dict[str, Dict[str, Any]] = {}
active_websockets: Dict[str, Set[WebSocket]] = {}


@app.post("/terminals/{terminal_id}/pairing-token", response_model=PairingTokenResponse)
def create_pairing_token(terminal_id: str):
    token = f"pair_{uuid.uuid4().hex[:12]}"
    pair_url = f"ws://localhost:8000/ws/session/{token}"
    active_pairings[token] = {
        "terminal_id": terminal_id,
        "token": token,
        "created_at": datetime.utcnow(),
    }
    return PairingTokenResponse(
        terminal_id=terminal_id,
        token=token,
        pair_url=pair_url,
        expires_in=300
    )


@app.websocket("/ws/session/{pairing_token}")
async def websocket_relay_endpoint(websocket: WebSocket, pairing_token: str):
    await websocket.accept()

    if pairing_token not in active_websockets:
        active_websockets[pairing_token] = set()
    active_websockets[pairing_token].add(websocket)

    try:
        # Send initial pairing success notification
        await websocket.send_json({
            "type": "pairing_success",
            "token": pairing_token,
            "message": "Connected to PalmPay WebSocket Relay Hub"
        })

        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            # Broadcast message to all other connected clients on this token
            target_sockets = active_websockets.get(pairing_token, set())
            for ws in list(target_sockets):
                if ws != websocket:
                    try:
                        await ws.send_json(data)
                    except Exception:
                        pass
    except WebSocketDisconnect:
        if pairing_token in active_websockets:
            active_websockets[pairing_token].discard(websocket)


# ---------------------------------------------------------------------------
# Host Standalone Test Frontend on /test and /
# ---------------------------------------------------------------------------

TEST_FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "test.html")

@app.get("/test", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
def serve_test_frontend():
    if os.path.exists(TEST_FRONTEND_PATH):
        with open(TEST_FRONTEND_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>PalmPay Backend Running. Test Frontend html not found.</h2>")
