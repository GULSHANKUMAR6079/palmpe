"""
PDF Receipt Generator using ReportLab.
Generates transaction certificates for completed palm payments.
"""

import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def mask_vpa(vpa: str) -> str:
    """Masks UPI VPA for privacy: e.g. aditya@hdfcbank -> adi****@hdfcbank"""
    if not vpa or "@" not in vpa:
        return vpa or ""
    handle, handle_domain = vpa.split("@", 1)
    if len(handle) <= 3:
        masked_handle = handle[0] + "***" if handle else "***"
    else:
        masked_handle = handle[:3] + "****"
    return f"{masked_handle}@{handle_domain}"


def generate_receipt(
    out_dir: str,
    transaction_id: int,
    customer_name: str,
    masked_upi: str,
    amount_rupees: float,
    merchant_id: str,
    razorpay_payment_id: str,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = f"receipt_{transaction_id}.pdf"
    filepath = os.path.join(out_dir, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0F172A'),
        alignment=1,
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#64748B'),
        alignment=1,
        spaceAfter=25
    )

    cell_label_style = ParagraphStyle(
        'CellLabel',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        fontName='Helvetica-Bold'
    )

    cell_value_style = ParagraphStyle(
        'CellValue',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#0F172A')
    )

    elements = [
        Paragraph("PalmPay Transaction Receipt", title_style),
        Paragraph("Biometric Micro-Payment Authorization Certificate", subtitle_style),
        Spacer(1, 10)
    ]

    data = [
        [Paragraph("Transaction ID", cell_label_style), Paragraph(f"#{transaction_id}", cell_value_style)],
        [Paragraph("Date & Time", cell_label_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"), cell_value_style)],
        [Paragraph("Merchant ID", cell_label_style), Paragraph(str(merchant_id), cell_value_style)],
        [Paragraph("Customer Name", cell_label_style), Paragraph(str(customer_name), cell_value_style)],
        [Paragraph("Masked UPI VPA", cell_label_style), Paragraph(str(masked_upi), cell_value_style)],
        [Paragraph("Amount Paid", cell_label_style), Paragraph(f"Rs. {amount_rupees:.2f}", cell_value_style)],
        [Paragraph("Razorpay Payment ID", cell_label_style), Paragraph(str(razorpay_payment_id), cell_value_style)],
        [Paragraph("Biometric Verification", cell_label_style), Paragraph("SUCCESS (Dual Scan Palm Verification)", cell_value_style)],
    ]

    table = Table(data, colWidths=[160, 320])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#0F172A')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
    ]))

    elements.append(table)
    doc.build(elements)
    return filepath
