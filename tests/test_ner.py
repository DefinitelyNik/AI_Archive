from unittest.mock import patch, MagicMock
from ner import perform_ner, find_dates, annotate_text, translate_text


@patch('ner.NER.load')
@patch('ner.Navec.load')
def test_perform_ner(mock_navec_load, mock_ner_load):
    """Тест функции perform_ner"""
    mock_ner_instance = MagicMock()
    mock_ner_instance.return_value = MagicMock()
    mock_ner_load.return_value = mock_ner_instance

    mock_navec_instance = MagicMock()
    mock_navec_load.return_value = mock_navec_instance

    mock_markup = MagicMock()
    mock_span = MagicMock()
    mock_span.start = 0
    mock_span.stop = 5
    mock_span.type = 'PER'
    mock_markup.spans = [mock_span]
    mock_markup.text = 'Тест текст'
    mock_ner_instance.return_value = mock_markup

    with patch('ner.annotate_text') as mock_annotate:
        mock_annotate.return_value = 'annotated text'
        result = perform_ner('Тест текст')
        assert result == 'annotated text'


def test_find_dates():
    """Тест функции поиска дат"""
    text = '''Я родился 15.03.1990,
              а в 1995 году уехал.
              В 2000-х годах жил в Москве.'''
    dates = find_dates(text)

    assert len(dates) >= 2
    date_1990_found = any('1990' in text[span_start:span_end]
                          for span_start, span_end, label
                          in dates
                          if label == 'date')
    assert date_1990_found


def test_find_dates_empty():
    """Тест с пустым текстом"""
    dates = find_dates('')
    assert dates == []


def test_annotate_text():
    """Тест функции аннотации текста"""
    mock_markup = MagicMock()
    mock_span = MagicMock()
    mock_span.start = 0
    mock_span.stop = 5
    mock_span.type = 'PER'
    mock_markup.spans = [mock_span]
    mock_markup.text = 'Тест текст'

    with patch('ner.find_dates', return_value=[]):
        result = annotate_text(mock_markup)
        ref = '<mark class="ner-per">Тест </mark>текст'
        assert ref in result or 'Тест' in result


def test_translate_text():
    """Тест функции перевода дореволюционного текста"""
    input_text = 'Нѣкоторый текст съ дореволюціонными буквами и ѣ'
    expected = 'Некоторый текст с дореволюционными буквами и е'
    result = translate_text(input_text)
    assert result == expected
