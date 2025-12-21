import easyocr

ocr_reader = easyocr.Reader(['ru'], gpu=True)


def perform_ocr(image_path: str) -> str:
    """
    Performs OCR on an image and returns the extracted text.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Extracted text from the image, joined by spaces.

    Example:
        >>> text = perform_ocr('path/to/image.jpg')
        >>> print(text)
        'Hello world'
    """
    result = ocr_reader.readtext(image_path)
    text = " ".join([item[1] for item in result])
    return text
