from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

def export_to_pdf(data_summary, filename="trade_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Master Quant Strategic Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on: {pd.Timestamp.now()}")

    c.line(50, height - 75, width - 50, height - 75)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 100, "1. Strategic Summary")

    c.setFont("Helvetica", 12)
    y = height - 120

    # We expect data_summary to be a dict
    for key, value in data_summary.items():
        if isinstance(value, dict):
             c.setFont("Helvetica-Bold", 12)
             c.drawString(50, y, f"{key}:")
             y -= 20
             c.setFont("Helvetica", 11)
             for sub_k, sub_v in value.items():
                 c.drawString(70, y, f"{sub_k}: {sub_v}")
                 y -= 15
        else:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y, f"{key}:")
            c.setFont("Helvetica", 11)
            c.drawString(150, y, f"{value}")
            y -= 20

        if y < 80:
            c.showPage()
            y = height - 50

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "Disclaimer: Trading involves risk. AI suggestions are for informational purposes only.")

    c.save()
    return filename

def export_to_json(data, filename="trade_snapshot.json"):
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    return filename
