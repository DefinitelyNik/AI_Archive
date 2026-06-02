import pytesseract
from PIL import Image
import os
import platform

# Путь к Tesseract для Windows (измените на свой путь, если нужно)
if platform.system() == 'Windows':
    # Типичные пути установки Tesseract на Windows
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
    ]

    # Проверяем, существует ли какой-то из путей
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break


def perform_tesseract_ocr(image_path: str, lang: str = 'rus') -> str:
    """
    Performs OCR on an image using Tesseract 5 (LSTM) and returns the extracted text.

    Args:
        image_path (str): Path to the input image file.
        lang (str): Language code for Tesseract (default: 'rus' for Russian).

    Returns:
        str: Extracted text from the image.
    """
    try:
        # Открываем изображение
        image = Image.open(image_path)

        # Выполняем OCR с использованием Tesseract 5 (LSTM)
        # Tesseract 5 по умолчанию использует LSTM нейронную сеть
        # PSM 6 - Предполагаем один равномерный блок текста
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)

        return text.strip()
    except Exception as e:
        error_msg = str(e)
        if "TesseractNotFoundError" in error_msg or "not found" in error_msg.lower():
            raise Exception(
                "Tesseract OCR не установлен или не найден в системе. "
                "Для Windows скачайте и установите Tesseract с: "
                "https://github.com/UB-Mannheim/tesseract/wiki\n"
                "После установки убедитесь, что путь к tesseract.exe добавлен в PATH "
                "или укажите путь в файле tesseract_ocr.py"
            )
        raise Exception(f"Error performing Tesseract OCR: {error_msg}")
