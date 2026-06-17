"""LLM-based cleanup for OCR and HTR text."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9
REPETITION_PENALTY = 1.05

_tokenizer = None
_generator = None

CLEANUP_PROMPT = """Ты исправляешь ошибки OCR/HTR в русском архивном тексте.

Правила:
- Исправляй только явные ошибки распознавания, пробелы, переносы строк и
  пунктуацию.
- Не добавляй новые факты, имена, даты или места.
- Не сокращай, не пересказывай и не объясняй текст.
- Сохраняй исходный смысл и порядок фрагментов.
- Верни только очищенный текст без комментариев.

Текст:
{text}

Очищенный текст:"""


def _load_generator():
    """Load Qwen text-generation pipeline once and reuse it."""
    global _tokenizer, _generator

    if _generator is not None:
        return

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    if not torch.cuda.is_available():
        model.to("cpu")

    _generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        device=device,
    )


def _build_prompt(text: str) -> str:
    return CLEANUP_PROMPT.format(text=text.strip())


def _extract_generated_text(result, prompt: str) -> str:
    generated_text = result[0].get("generated_text", "")
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    return generated_text.strip()


def clean_text(text: str) -> str:
    """
    Clean OCR/HTR text with a local Qwen model.

    If the model cannot be loaded or generation fails, the original text is
    returned so the main OCR pipeline remains usable.
    """
    if not text or not text.strip():
        return text

    try:
        _load_generator()
        prompt = _build_prompt(text)
        result = _generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
        cleaned_text = _extract_generated_text(result, prompt)
        return cleaned_text or text
    except Exception as exc:
        logger.error("Text cleanup failed: %s", exc)
        logger.exception("Full traceback:")
        return text
