"""Tests for handwritten text recognition helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from htr import group_by_lines, perform_htr


def test_group_by_lines():
    """Nearby detections are grouped into one text line."""
    detections = [
        ([[10, 10], [100, 10], [100, 30], [10, 30]], 'Строка1', 0.9),
        ([[10, 15], [100, 15], [100, 35], [10, 35]], 'Строка2', 0.85),
        ([[10, 50], [100, 50], [100, 70], [10, 70]], 'Строка3', 0.8),
    ]

    result = group_by_lines(detections, y_tolerance=10)
    assert len(result) == 2
    assert len(result[0]) == 2
    assert len(result[1]) == 1


def test_group_by_lines_empty():
    assert group_by_lines([]) == []


def test_perform_htr_file_not_found():
    """An absent image is rejected before OpenCV/model initialization."""
    with pytest.raises(FileNotFoundError):
        perform_htr('nonexistent.jpg')


def test_perform_htr_no_detections():
    """No EasyOCR detections returns an empty recognition result."""
    reader = MagicMock()
    reader.readtext.return_value = []
    processor = MagicMock()
    model = MagicMock()

    with patch('htr.os.path.exists', return_value=True), patch(
        'htr.cv2.imread', return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    ), patch(
        'htr.cv2.cvtColor', return_value=np.zeros((100, 100, 3), dtype=np.uint8)
    ), patch('htr.Image.fromarray'), patch(
        'htr._load_htr_models', return_value=(reader, processor, model)
    ):
        lines, full_text = perform_htr('dummy_path.jpg')

    assert lines == []
    assert full_text == ''
