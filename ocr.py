import os
import easyocr
import torch

# FIX: Ленивая инициализация — модель загружается только при первом вызове
_ocr_reader = None


def get_ocr_reader():
    """Lazy initialization of EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        gpu = torch.cuda.is_available()
        _ocr_reader = easyocr.Reader(['ru'], gpu=gpu)
    return _ocr_reader


def perform_ocr(image_path: str) -> str:
    """
    Performs OCR on an image and returns the extracted text.

    Args:
        image_path: Path to the input image file.

    Returns:
        Extracted text from the image, joined by spaces.
    """
    # FIX: Проверяем существование файла
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    reader = get_ocr_reader()
    result = reader.readtext(image_path)
    text = " ".join([item[1] for item in result])
    return text