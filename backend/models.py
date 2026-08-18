import enum
from datetime import datetime

from sqlalchemy import (
    JSON, Column, DateTime, Enum, Float, ForeignKey, Integer, String,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    contact = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=True, unique=True)
    upi_vpa = Column(String, nullable=False)

    razorpay_customer_id = Column(String, nullable=True)
    mandate_order_id = Column(String, nullable=True)
    mandate_token_id = Column(String, nullable=True)
    mandate_limit_paise = Column(Integer, default=10000)  # Rs 100 default cap
    
    step_up_pin_hash = Column(String, nullable=True)
    consent_given_at = Column(DateTime, nullable=True)
    consent_version = Column(String, nullable=True, default="v1.0")
    registered_handedness = Column(String, nullable=True)  # "Left" or "Right"

    created_at = Column(DateTime, default=datetime.utcnow)

    embeddings = relationship("PalmEmbedding", back_populates="customer", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="customer")


class PalmEmbedding(Base):
    __tablename__ = "palm_embeddings"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id", ondelete="CASCADE"), nullable=False)
    vector = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="embeddings")


class TransactionStatus(str, enum.Enum):
    IDENTIFIED = "identified"
    AMOUNT_SET = "amount_set"
    AUTHORIZED = "authorized"
    PAID = "paid"
    BORDERLINE = "borderline"
    FAILED = "failed"
    REJECTED_MISMATCH = "rejected_mismatch"


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    merchant_id = Column(String, nullable=False)

    amount_rupees = Column(Float, nullable=True)
    status = Column(Enum(TransactionStatus), default=TransactionStatus.IDENTIFIED)

    identify_confidence = Column(Float, nullable=True)
    authorize_confidence = Column(Float, nullable=True)

    identify_handedness = Column(String, nullable=True)
    identify_embedding = Column(JSON, nullable=True)

    razorpay_payment_id = Column(String, nullable=True)
    receipt_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="transactions")
