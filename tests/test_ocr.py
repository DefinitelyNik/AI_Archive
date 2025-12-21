from unittest.mock import patch
from ocr import perform_ocr


@patch('ocr.ocr_reader')
def test_perform_ocr(mock_reader):
    """Тест функции perform_ocr"""
    mock_result = [
        [[[10, 10], [100, 10], [100, 50], [10, 50]], 'Привет', 0.9],
        [[[10, 60], [100, 60], [100, 100], [10, 100]], 'Мир', 0.85]
    ]
    mock_reader.readtext.return_value = mock_result

    text = perform_ocr('dummy_path.jpg')

    mock_reader.readtext.assert_called_once_with('dummy_path.jpg')
    assert text == 'Привет Мир'


@patch('ocr.ocr_reader')
def test_perform_ocr_empty_result(mock_reader):
    """Тест perform_ocr с пустым результатом"""
    mock_reader.readtext.return_value = []

    text = perform_ocr('dummy_path.jpg')
    assert text == ''
