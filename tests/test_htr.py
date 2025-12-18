import pytest
from unittest.mock import patch, MagicMock, mock_open
import cv2
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from htr import perform_htr, group_by_lines

def test_group_by_lines():
    """Тест функции группировки строк"""
    detections = [
        ([[10, 10], [100, 10], [100, 30], [10, 30]], 'Строка1', 0.9),
        ([[10, 15], [100, 15], [100, 35], [10, 35]], 'Строка2', 0.85),
        ([[10, 50], [100, 50], [100, 70], [10, 70]], 'Строка3', 0.8)
    ]

    result = group_by_lines(detections, y_tolerance=10)
    assert len(result) == 2
    assert len(result[0]) == 2  # 2 фрагмента в первой строке
    assert len(result[1]) == 1  # 1 фрагмент во второй строке

def test_group_by_lines_empty():
    """Тест с пустыми детекциями"""
    result = group_by_lines([])
    assert result == []

@patch('htr.cv2.imread', return_value=None)
def test_perform_htr_file_not_found(mock_imread):
    """Тест perform_htr с несуществующим файлом"""
    with pytest.raises(cv2.error):
        perform_htr('nonexistent.jpg')

@patch('htr.cv2.imread')
@patch('htr.cv2.cvtColor')
@patch('htr.Image.fromarray')
@patch('htr.easyocr.Reader')
def test_perform_htr_no_detections(mock_reader_class, mock_fromarray, mock_cvtColor, mock_imread):
    """Тест perform_htr с пустыми детекциями"""
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_pil_image = MagicMock()
    mock_fromarray.return_value = mock_pil_image
    mock_reader = MagicMock()
    mock_reader_class.return_value = mock_reader
    mock_reader.readtext.return_value = []

    lines, full_text = perform_htr('dummy_path.jpg')
    assert lines == []
    assert full_text == ''