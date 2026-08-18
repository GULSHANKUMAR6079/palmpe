from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class RegisterResponse(BaseModel):
    customer_id: int
    mandate_order_id: str
    message: str = "Palm enrolled successfully. Mandate approval required."


class IdentifyResponse(BaseModel):
    matched: bool
    status: str = Field(..., description="matched | borderline | unmatched")
    requires_step_up: bool = False
    step_up_prompt: Optional[str] = None
    customer_id: Optional[int] = None
    name: Optional[str] = None
    masked_upi: Optional[str] = None
    confidence: float = 0.0
    session_id: Optional[int] = None
    message: Optional[str] = None
    handedness: Optional[str] = None


class SetAmountRequest(BaseModel):
    session_id: int
    amount_rupees: float


class SetAmountResponse(BaseModel):
    ok: bool = True
    amount_rupees: float


class AuthorizeResponse(BaseModel):
    status: str = Field(..., description="paid | borderline | rejected_mismatch | failed")
    requires_step_up: bool = False
    step_up_prompt: Optional[str] = None
    amount_rupees: Optional[float] = None
    razorpay_payment_id: Optional[str] = None
    receipt_url: Optional[str] = None
    reason: Optional[str] = None


class StepUpVerifyRequest(BaseModel):
    session_id: int
    secret: str


class TransactionItem(BaseModel):
    id: int
    created_at: datetime
    customer_name: Optional[str] = None
    masked_upi: Optional[str] = None
    amount_rupees: Optional[float] = None
    status: str
    razorpay_payment_id: Optional[str] = None
    mandate_token_id: Optional[str] = None
    authorize_confidence: Optional[float] = None


class TransactionListResponse(BaseModel):
    transactions: List[TransactionItem]


class CustomerStateResponse(BaseModel):
    id: int
    name: str
    contact: str
    email: Optional[str] = None
    masked_upi: str
    razorpay_customer_id: Optional[str] = None
    mandate_order_id: Optional[str] = None
    mandate_token_id: Optional[str] = None
    mandate_approved: bool
    embedding_count: int
    created_at: datetime


class CustomerUpdateRequest(BaseModel):
    name: Optional[str] = None
    contact: Optional[str] = None
    email: Optional[str] = None
    upi_vpa: Optional[str] = None
    step_up_pin: Optional[str] = None


class MandateApprovedRequest(BaseModel):
    customer_id: int
    token_id: str


class PairingTokenResponse(BaseModel):
    terminal_id: str
    token: str
    pair_url: str
    expires_in: int = 300
