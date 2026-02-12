"""
Data setup: Generate a sample HR manual PDF for testing the Knowledge Engine.
Uses fpdf2 to create a small PDF with policy content (e.g. leave, childcare).
"""
import sys
from pathlib import Path

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fpdf import FPDF
from fpdf.enums import XPos, YPos


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / "hr_manual_sample.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.set_title("HR Policy Manual (Sample)")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(True, margin=20)
    # Use explicit width so we never get "not enough horizontal space"
    w = pdf.epw  # effective page width (between margins)

    sections = [
        ("Leave Policy", [
            "Employees are entitled to 22 days of annual leave per calendar year.",
            "Paternity Leave: 2 weeks paid leave for new fathers, to be taken within 6 months of the child's birth.",
            "Maternity Leave: 26 weeks paid leave for new mothers. Additional unpaid leave may be agreed with your manager.",
        ]),
        ("Childcare and Family", [
            "We offer flexible working arrangements for parents. You may request remote work or adjusted hours.",
            "Childcare vouchers are available; contact HR for the current scheme.",
            "If you need to care for a dependent in an emergency, use the dedicated emergency leave policy.",
        ]),
        ("Expenses", [
            "Submit expense claims within 30 days of the purchase. Use the internal portal.",
            "Receipts are required for any claim over £25.",
        ]),
    ]

    for title, paragraphs in sections:
        pdf.set_font("Helvetica", "B", size=14)
        pdf.cell(w, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=11)
        for p in paragraphs:
            pdf.multi_cell(w, 7, p)
        pdf.ln(4)

    pdf.output(str(out))
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
