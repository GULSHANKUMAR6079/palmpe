"""
CLI utility to inspect enrolled PalmPay customers and mandate statuses.
Run with: python scripts/list_customers.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import SessionLocal
from backend.models import Customer, PalmEmbedding
from backend.receipt import mask_vpa


def main():
    db = SessionLocal()
    try:
        customers = db.query(Customer).all()
        print("=========================================================================")
        print("                        ENROLLED PALMPAY CUSTOMERS                       ")
        print("=========================================================================")
        if not customers:
            print("No customers enrolled in local database yet.")
            return

        for c in customers:
            print(f"ID: #{c.id} | Name: {c.name}")
            print(f"  Phone: {c.contact} | Email: {c.email}")
            print(f"  UPI VPA: {mask_vpa(c.upi_vpa)} (Raw: {c.upi_vpa})")
            print(f"  Handedness: {c.registered_handedness or 'Unknown'}")
            print(f"  Mandate Order ID: {c.mandate_order_id or 'None'}")
            print(f"  Mandate Token ID: {c.mandate_token_id or 'Pending Approval'}")
            print(f"  Step-Up PIN Set: {'YES' if c.step_up_pin_hash else 'NO'}")
            print(f"  Embeddings Enrolled: {len(c.embeddings)}")
            print(f"  Enrolled At: {c.created_at}")
            print("-------------------------------------------------------------------------")
    finally:
        db.close()


if __name__ == "__main__":
    main()
