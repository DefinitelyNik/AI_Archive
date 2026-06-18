"""LLM-based cleanup for OCR and HTR text through Ollama."""

import json
import logging
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))

SYSTEM_PROMPT = (
    "/no_think\n"
    "Ты исправляешь ошибки OCR/HTR в русском архивном тексте. "
    "Верни JSON без рассуждений, заголовков и комментариев."
)

USER_PROMPT = """/no_think

Исправь распознанный текст.

Правила:
- Исправляй только явные ошибки распознавания, пробелы, переносы строк и пунктуацию.
- Не добавляй новые факты, имена, даты или места.
- Не сокращай, не пересказывай и не объясняй текст.
- Сохраняй исходный смысл и порядок фрагментов.
- Исправляй типичные OCR-замены: 0/о, |/л/П, ПО/по, лишние подчёркивания,
  перепутанные русские и латинские буквы, слитые годы и пробелы.
- Не используй блоки <think> и не показывай ход рассуждений.
- Верни только JSON: {{"cleaned_text": "..."}}.

Пример:
Исходный текст: Пушкин неоднократнс писал 0 своей родословной.
Ответ: {{"cleaned_text": "Пушкин неоднократно писал о своей родословной."}}

Исходный текст:
{text}

Ответ:"""


def _build_messages(text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(text=text.strip())},
    ]


def _build_payload(text: str):
    return {
        "model": OLLAMA_MODEL,
        "format": "json",
        "stream": False,
        "messages": _build_messages(text),
        "options": {
            "temperature": 0,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
    }


def _ollama_chat_url() -> str:
    return f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    return text


def _strip_answer_prefix(text: str) -> str:
    for prefix in ("Очищенный текст:", "Ответ:", "Исправленный текст:"):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _looks_like_reasoning(text: str) -> bool:
    markers = (
        "Analyze Errors",
        "Input Text provided",
        "Wait,",
        "the prompt says",
        "system instruction",
        "I should output",
        "Let's look",
    )
    return any(marker in text for marker in markers)


def _parse_cleaned_text(content: str) -> str:
    content = _strip_answer_prefix(_strip_thinking(content.strip()))

    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return str(payload.get("cleaned_text", "")).strip()
    except json.JSONDecodeError:
        pass

    match = re.search(
        r'"cleaned_text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"',
        content,
        flags=re.DOTALL,
    )
    if match:
        return json.loads(f'"{match.group("text")}"').strip()

    if _looks_like_reasoning(content):
        return ""

    return content


def _call_ollama(text: str) -> str:
    request_body = json.dumps(_build_payload(text)).encode("utf-8")
    request = Request(
        _ollama_chat_url(),
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        response_body = response.read().decode("utf-8")

    payload = json.loads(response_body)
    content = payload.get("message", {}).get("content", "")
    return _parse_cleaned_text(content)


def clean_text(text: str) -> str:
    """
    Clean OCR/HTR text with a local Ollama model.

    If Ollama is unavailable or generation fails, the original text is returned
    so the main OCR/HTR pipeline remains usable.
    """
    if not text or not text.strip():
        return text

    try:
        cleaned_text = _call_ollama(text)
        return cleaned_text or text
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.error("Ollama text cleanup failed: %s", exc)
        logger.exception("Full traceback:")
        return text
