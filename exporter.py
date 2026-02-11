from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd

def export_to_pdf(data_summary, filename="trade_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Quantitative Trading Report")

    c.setFont("Helvetica", 12)
    y = height - 80
    for key, value in data_summary.items():
        c.drawString(50, y, f"{key}: {value}")
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    return filename

def export_to_json(data, filename="trade_snapshot.json"):
    import json
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    return filename
