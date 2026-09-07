import os
import platform
import pytesseract
from PIL import Image

# Путь к Tesseract для Windows
if platform.system() == 'Windows':
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
    ]
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


def perform_tesseract_ocr(image_path: str, lang: str = 'rus') -> str:
    """
    Performs OCR on an image using Tesseract 5 (LSTM).

    Args:
        image_path: Path to the input image file.
        lang: Language code for Tesseract (default: 'rus').

    Returns:
        Extracted text from the image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        # FIX: Используем context manager для закрытия изображения
        with Image.open(image_path) as image:
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
            return text.strip()
    except pytesseract.TesseractNotFoundError as e:
        raise Exception(
            "Tesseract OCR не установлен или не найден в системе. "
            "Для Windows скачайте и установите Tesseract с: "
            "https://github.com/UB-Mannheim/tesseract/wiki\n"
            "После установки убедитесь, что путь к tesseract.exe добавлен в PATH "
            "или укажите путь в файле tesseract_ocr.py"
        ) from e
    except Exception as e:
        raise Exception(f"Error performing Tesseract OCR: {e}") from e