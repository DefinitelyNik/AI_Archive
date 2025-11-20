import easyocr

ocr_reader = easyocr.Reader(['ru'], gpu=True)

def perform_ocr(image_path):
    result = ocr_reader.readtext(image_path)
    text = " ".join([item[1] for item in result])
    return text