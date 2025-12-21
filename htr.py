import cv2
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import easyocr
import statistics


def group_by_lines(detection_results: list, y_tolerance: int = 10) -> list:
    """
    Groups detected text fragments into lines based on Y-coordinates.

    Args:
        detection_results (list): Results from EasyOCR readtext.
        y_tolerance (int): Tolerance for grouping fragments into lines.

    Returns:
        list: List of lines, where each line
        is a list of (bbox, text, prob) tuples.
    """
    sorted_results = sorted(
        detection_results,
        key=lambda x: min([pt[1] for pt in x[0]]))

    lines = []
    current_line = []
    current_line_y_center = None

    for bbox, text, prob in sorted_results:
        y_cords = [pt[1] for pt in bbox]
        y_min, y_max = min(y_cords), max(y_cords)
        y_center = (y_min + y_max) / 2

        if current_line_y_center is None:
            current_line.append((bbox, text, prob))
            current_line_y_center = y_center
        elif abs(y_center - current_line_y_center) <= y_tolerance:
            current_line.append((bbox, text, prob))
            centers = [
                (min([pt[1] for pt in b]) + max([pt[1] for pt in b]))
                / 2 for b, t, p in current_line]
            current_line_y_center = statistics.mean(centers)
        else:
            if current_line:
                lines.append(current_line)
            current_line = [(bbox, text, prob)]
            current_line_y_center = y_center

    if current_line:
        lines.append(current_line)

    return lines


def perform_htr(image_path: str,
                model_name: str = "kazars24/trocr-base-handwritten-ru",
                y_tolerance: int = 10) -> tuple:
    """
    Performs Handwritten Text Recognition (HTR) on a multi-line image.

    Args:
        image_path (str): Path to the input image file.
        model_name (str): Name of the TrOCR model to use.
        y_tolerance (int): Tolerance for grouping text fragments into lines.

    Returns:
        tuple: A tuple containing:
            - list: Recognized lines of text.
            - str: Full text joined by newlines.

    Example:
        >>> lines, full_text = perform_htr('path/to/image.jpg')
        >>> print(lines)
        ['Line 1 text', 'Line 2 text']
        >>> print(full_text)
        'Line 1 text\\nLine 2 text'
    """
    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    reader = easyocr.Reader(['ru'], gpu=True)
    detection_results = reader.readtext(image_rgb, paragraph=False)

    if not detection_results:
        return [], ""

    grouped_lines = group_by_lines(detection_results, y_tolerance=y_tolerance)

    processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    final_recognized_lines = []
    full_text_parts = []

    for line_idx, line_fragments in enumerate(grouped_lines):
        if not line_fragments:
            final_recognized_lines.append("")
            full_text_parts.append("")
            continue

        x_cords = []
        y_cords = []
        for bbox, _, _ in line_fragments:
            x_cords.extend([pt[0] for pt in bbox])
            y_cords.extend([pt[1] for pt in bbox])

        x_min = min(x_cords)
        x_max = max(x_cords)
        y_min = min(y_cords)
        y_max = max(y_cords)

        if x_max <= x_min or y_max <= y_min:
            final_recognized_lines.append("")
            full_text_parts.append("")
            continue

        line_image = pil_image.crop((x_min, y_min, x_max, y_max))

        pixel_values = processor(images=line_image,
                                 return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = model.generate(pixel_values)

        line_text = processor.batch_decode(outputs,
                                           skip_special_tokens=True)[0]
        final_recognized_lines.append(line_text)
        full_text_parts.append(line_text)

    full_text = "\n".join(full_text_parts)
    return final_recognized_lines, full_text
