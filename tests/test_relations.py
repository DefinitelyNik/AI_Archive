"""Tests for the relations extraction module."""

from unittest.mock import patch
from relations import extract_relations, _parse_llm_response


class TestParseLlmResponse:
    """Tests for the _parse_llm_response function."""

    def test_parse_valid_list(self):
        """Test parsing a valid Python list response."""
        response = ("[('Иван', 'место рождения', 'Москва'), "
                    "('Пётр', 'родитель', 'Иван')]")
        result = _parse_llm_response(response)
        assert len(result) == 2
        assert result[0] == ('Иван', 'место рождения', 'Москва')
        assert result[1] == ('Пётр', 'родитель', 'Иван')

    def test_parse_empty_list(self):
        """Test parsing an empty list response."""
        response = "[]"
        result = _parse_llm_response(response)
        assert result == []

    def test_parse_with_extra_text(self):
        """Test parsing when LLM adds explanation text."""
        response = ("Вот извлечённые отношения:"
                    "\n[('Иван', 'место рождения', 'Москва')]"
                    "\nБольше ничего не нашёл.")
        result = _parse_llm_response(response)
        assert len(result) == 1
        assert result[0] == ('Иван', 'место рождения', 'Москва')

    def test_parse_invalid_response(self):
        """Test parsing an invalid response."""
        response = "Я не нашёл никаких отношений в этом тексте."
        result = _parse_llm_response(response)
        assert result == []

    def test_parse_double_quotes(self):
        """Test parsing with double quotes."""
        response = '[("Иван", "место рождения", "Москва")]'
        result = _parse_llm_response(response)
        assert len(result) == 1
        assert result[0] == ('Иван', 'место рождения', 'Москва')

    def test_parse_tuple_fallback(self):
        """Test regex fallback for tuple extraction."""
        response = "('Иван', 'место рождения', 'Москва'), ('Пётр', 'родитель', 'Иван')"
        result = _parse_llm_response(response)
        assert len(result) == 2

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = _parse_llm_response("")
        assert result == []

    def test_parse_none(self):
        """Test parsing None."""
        result = _parse_llm_response(None)
        assert result == []


class TestExtractRelations:
    """Tests for the extract_relations function."""

    def test_empty_text(self):
        """Test with empty text."""
        result = extract_relations("")
        assert result == []

    def test_none_text(self):
        """Test with None text."""
        result = extract_relations(None)
        assert result == []

    def test_short_text(self):
        """Test with very short text."""
        result = extract_relations("Привет")
        assert result == []

    @patch('relations._load_model')
    @patch('relations._generator')
    @patch('relations._tokenizer')
    def test_extract_relations_mock(self, mock_tokenizer, mock_generator, mock_load):
        """Test extract_relations with mocked model."""
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"

        mock_generator.return_value = [{
            "generated_text": "formatted prompt[('Иван', 'место рождения', 'Москва')]"
        }]

        result = extract_relations("Иван родился в Москве.")
        assert len(result) == 1
        assert result[0] == ('Иван', 'место рождения', 'Москва')

    @patch('relations._load_model')
    @patch('relations._generator')
    @patch('relations._tokenizer')
    def test_extract_relations_error_handling(self,
                                              mock_tokenizer,
                                              mock_generator,
                                              mock_load):
        """Test that errors are handled gracefully."""
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.apply_chat_template.side_effect = Exception("Model error")

        result = extract_relations("Иван родился в Москве.")
        assert result == []
