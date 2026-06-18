import json
from unittest.mock import Mock, patch
from urllib.error import URLError

from text_cleanup import _build_payload, _parse_cleaned_text, clean_text


def test_cleanup_prompt_is_readable_russian_text():
    payload = _build_payload("Пушкин неоднократнс писал 0 своей родословной.")
    prompt = payload["messages"][1]["content"]

    assert payload["model"] == "qwen3.5:9b"
    assert payload["format"] == "json"
    assert "/no_think" in payload["messages"][0]["content"]
    assert "/no_think" in prompt
    assert "Исправь распознанный текст" in prompt
    assert "Пушкин неоднократнс писал 0 своей родословной." in prompt
    assert "Рў" not in prompt


def test_clean_text_empty_text_does_not_call_ollama():
    with patch("text_cleanup._call_ollama") as mock_call:
        assert clean_text("") == ""
        mock_call.assert_not_called()


@patch("text_cleanup.urlopen")
def test_clean_text_returns_ollama_response(mock_urlopen):
    response = Mock()
    response.read.return_value = json.dumps({
        "message": {
            "content": json.dumps({
                "cleaned_text": "Пушкин неоднократно писал о своей родословной."
            })
        }
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = response

    result = clean_text("Пушкин неоднократнс писал 0 своей родословной.")

    assert result == "Пушкин неоднократно писал о своей родословной."
    request = mock_urlopen.call_args.args[0]
    request_body = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:11434/api/chat"
    assert request_body["model"] == "qwen3.5:9b"
    assert request_body["format"] == "json"
    assert request_body["stream"] is False


@patch("text_cleanup.urlopen")
def test_clean_text_removes_thinking_block(mock_urlopen):
    response = Mock()
    response.read.return_value = json.dumps({
        "message": {
            "content": (
                "<think>Нужно исправить OCR.</think>\n"
                "{\"cleaned_text\": \"Пушкин неоднократно писал о своей родословной.\"}"
            )
        }
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = response

    result = clean_text("Пушкин неоднократнс писал 0 своей родословной.")

    assert result == "Пушкин неоднократно писал о своей родословной."


def test_parse_cleaned_text_rejects_reasoning():
    result = _parse_cleaned_text(
        "1. **Analyze Errors in Input Text:**\n"
        "* Input Text provided: \"raw text\"\n"
        "* Wait, the prompt says to return clean text."
    )

    assert result == ""


@patch("text_cleanup.urlopen", side_effect=URLError("ollama is unavailable"))
def test_clean_text_returns_original_text_on_error(mock_urlopen):
    text = "original text"
    assert clean_text(text) == text
