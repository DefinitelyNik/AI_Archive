from unittest.mock import patch

from text_cleanup import clean_text


def test_clean_text_empty_text_does_not_load_model():
    with patch("text_cleanup._load_generator") as mock_load:
        assert clean_text("") == ""
        mock_load.assert_not_called()


@patch("text_cleanup._generator")
@patch("text_cleanup._tokenizer")
@patch("text_cleanup._load_generator")
def test_clean_text_returns_generated_text(mock_load, mock_tokenizer, mock_generator):
    mock_tokenizer.eos_token_id = 0
    mock_generator.return_value = [{
        "generated_text": "promptCleaned archive text."
    }]

    with patch("text_cleanup._build_prompt", return_value="prompt"):
        result = clean_text("Raw archive text.")

    assert result == "Cleaned archive text."


@patch("text_cleanup._load_generator", side_effect=RuntimeError("model error"))
def test_clean_text_returns_original_text_on_error(mock_load):
    text = "original text"
    assert clean_text(text) == text
