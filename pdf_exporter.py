import logging
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from typing import Dict, Any, List

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def export_overview_to_pdf(filename: str, kpis: Dict[str, Any], fleet: List[Dict[str, Any]]):
    """
    Экспорт KPI и информации о флоте в PDF отчет.
    
    :param filename: Путь к итоговому PDF файлу
    :param kpis: Словарь ключевых показателей эффективности
    :param fleet: Список судов с текущим положением
    """
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Заголовок
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Отчет о флоте")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # KPI
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 100, "Ключевые показатели:")
    c.setFont("Helvetica", 10)
    y = height - 120
    for key, value in kpis.items():
        c.drawString(60, y, f"{key}: {value}")
        y -= 15

    # Fleet info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 20, "Текущее положение флота:")
    y -= 40
    c.setFont("Helvetica", 9)
    for vessel in fleet:
        vessel_str = f"{vessel.get('name', 'Неизвестно')} — {vessel.get('status', '')}, порт {vessel.get('current_port', '')}"
        c.drawString(60, y, vessel_str)
        y -= 15
        if y < 100:  # новая страница
            c.showPage()
            y = height - 50

    c.save()
    return filename