from ner import translate_text

def test_translate_text():
    """Тест перевода дореволюционного русского"""
    input_text = "Ѣгусь по улицѣ съ собакой Ѳомой."
    expected = "Егусь по улице с собакой Фомой."
    result = translate_text(input_text)
    assert result == expected

def test_translate_text_no_changes():
    """Тест текста без дореволюционных букв"""
    input_text = "Иду по улице с собакой."
    result = translate_text(input_text)
    assert result == input_text
