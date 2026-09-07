"""Tests for the EasyOCR wrapper."""

from unittest.mock import MagicMock, patch

from ocr import perform_ocr


def test_perform_ocr():
    """OCR joins recognized fragments in reading order."""
    reader = MagicMock()
    reader.readtext.return_value = [
        [[[10, 10], [100, 10], [100, 50], [10, 50]], 'Привет', 0.9],
        [[[10, 60], [100, 60], [100, 100], [10, 100]], 'Мир', 0.85],
    ]

    with patch('ocr.os.path.exists', return_value=True), patch(
        'ocr.get_ocr_reader', return_value=reader
    ):
        text = perform_ocr('dummy_path.jpg')

    reader.readtext.assert_called_once_with('dummy_path.jpg')
    assert text == 'Привет Мир'


def test_perform_ocr_empty_result():
    """OCR returns an empty string when no text fragments are detected."""
    reader = MagicMock()
    reader.readtext.return_value = []

    with patch('ocr.os.path.exists', return_value=True), patch(
        'ocr.get_ocr_reader', return_value=reader
    ):
        text = perform_ocr('dummy_path.jpg')

    assert text == ''
