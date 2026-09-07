"""Tests for NER, date detection and text normalization."""

from unittest.mock import MagicMock, patch

from ner import annotate_text, find_dates, perform_ner, translate_text


def test_perform_ner(monkeypatch):
    """NER loads models lazily and passes markup to the annotation function."""
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_navec = MagicMock()
    monkeypatch.setattr('ner._navec', None)
    monkeypatch.setattr('ner._ner_model', None)

    with patch('ner.os.path.exists', return_value=True), patch(
        'ner.Navec.load', return_value=mock_navec
    ), patch('ner.NER.load', return_value=mock_model), patch(
        'ner.annotate_text', return_value='annotated text'
    ):
        assert perform_ner('Тест текст') == 'annotated text'

    mock_model.navec.assert_called_once_with(mock_navec)


def test_find_dates():
    text = '''Я родился 15.03.1990,
              а в 1995 году уехал.
              В 2000-х годах жил в Москве.'''
    dates = find_dates(text)

    assert len(dates) >= 2
    assert any(
        '1990' in text[start:stop]
        for start, stop, label in dates
        if label == 'date'
    )


def test_find_dates_empty():
    assert find_dates('') == []


def test_annotate_text_uses_expected_ner_class_and_escapes_source():
    mock_markup = MagicMock()
    mock_span = MagicMock()
    mock_span.start = 0
    mock_span.stop = 5
    mock_span.type = 'PER'
    mock_markup.spans = [mock_span]
    mock_markup.text = 'Тест <script>alert(1)</script>'

    with patch('ner.find_dates', return_value=[]):
        result = annotate_text(mock_markup)

    assert '<mark class="ner-per">Тест </mark>' in result
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in result
    assert '<script>' not in result


def test_translate_text():
    input_text = 'Нѣкоторый текст съ дореволюціонными буквами и ѣ'
    expected = 'Некоторый текст с дореволюционными буквами и е'
    assert translate_text(input_text) == expected
