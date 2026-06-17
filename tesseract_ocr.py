import os
import platform

from PIL import Image


def _load_pytesseract():
    import pytesseract

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

    return pytesseract


def perform_tesseract_ocr(image_path: str, lang: str = 'rus') -> str:
    """
    Performs OCR on an image using Tesseract 5 (LSTM).

    Args:
        image_path (str): Path to the input image file.
        lang (str): Language code for Tesseract.

    Returns:
        str: Extracted text from the image.
    """
    try:
        pytesseract = _load_pytesseract()
        image = Image.open(image_path)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=custom_config
        )

        return text.strip()
    except Exception as exc:
        error_msg = str(exc)
        if "TesseractNotFoundError" in error_msg or "not found" in error_msg.lower():
            raise Exception(
                "Tesseract OCR is not installed or was not found. "
                "Install Tesseract and make sure tesseract.exe is available "
                "in PATH, or configure its path in tesseract_ocr.py."
            )
        raise Exception(f"Error performing Tesseract OCR: {error_msg}")
