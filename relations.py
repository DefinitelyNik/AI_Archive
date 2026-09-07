"""
Relations Extraction Module

This module provides functions for extracting relations from text
using a local LLM model via the transformers library.
"""

import ast
import logging
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# FIX: Убран logging.basicConfig() — не переопределяем глобальную конфигурацию
logger = logging.getLogger(__name__)

# Global variables for model caching (loaded on first use)
_tokenizer = None
_generator = None

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
TOP_P = 0.9
REPETITION_PENALTY = 1.1

# Prompt template for relation extraction
RELATION_PROMPT = """
Ты — система извлечения семантических отношений из текста на русском языке.
Проанализируй текст и извлеки ВСЕ отношения между сущностями.

Типы отношений для поиска:
- родитель (отец, мать)
- ребёнок (сын, дочь)
- место рождения
- дата рождения
- крещение (кто крестил, кем крещён)
- брак (муж, жена, супруг, супруга)
- место жительства / проживания
- работа / служба (место работы)
- образование (где учился, окончил)
- смерть (дата смерти, место смерти)
- родство (брат, сестра, дедушка, бабушка, дядя, тётя и т.д.)
- звание / чин
- принадлежность к организации

Верни результат СТРОГО в виде списка кортежей Python:
[(сущность1, тип_отношения, сущность2), ...]

Правила:
- Если отношений нет — верни пустой список: []
- Используй полные имена, если они есть в тексте
- Тип отношения пиши кратко на русском языке
- Не добавляй пояснений, только список

Пример 1:
Текст: "Иван Петрович родился 15 марта 1890 года в Москве.
Его отец Пётр Сергеевич, а мать Анна Михайловна.
Крестила его Мария Сидорова в церкви села Коломенское."
Ответ: [('Иван Петрович', 'дата рождения', '15 марта 1890 года'),
('Иван Петрович', 'место рождения', 'Москва'),
('Пётр Сергеевич', 'родитель', 'Иван Петрович'),
('Анна Михайловна', 'родитель', 'Иван Петрович'),
('Мария Сидорова', 'крестил', 'Иван Петрович'),
('Иван Петрович', 'место крещения', 'церковь села Коломенское')]

Пример 2:
Текст: "Князь Алексей Дмитриевич Щербаков,
1845 года рождения,
скончался в Санкт-Петербурге в 1912 году.
Его жена — Екатерина Васильевна."
Ответ: [('Алексей Дмитриевич Щербаков', 'дата рождения', '1845'),
('Алексей Дмитриевич Щербаков', 'место смерти', 'Санкт-Петербург'),
('Алексей Дмитриевич Щербаков', 'дата смерти', '1912'),
('Екатерина Васильевна', 'супруг', 'Алексей Дмитриевич Щербаков')]

Текст для анализа:
{text}

Ответ:"""


def _load_model():
    """
    Load the LLM model and tokenizer. Cached after first call.
    Uses GPU if available, otherwise falls back to CPU.
    """
    global _tokenizer, _generator

    if _generator is not None:
        logger.debug("Model already loaded, skipping initialization")
        return

    logger.info("Loading relation extraction model: %s ...", MODEL_NAME)
    logger.debug("Device: %s", 'GPU' if torch.cuda.is_available() else 'CPU')

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.debug("Tokenizer loaded successfully")

    # FIX: Разделяем логику для GPU и CPU — избегаем конфликта device_map и model.to()
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        device = 0
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")
        device = -1

    logger.debug("Model loaded successfully")

    _generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        device=device,
    )

    logger.info("Model loaded successfully on %s", 'GPU' if torch.cuda.is_available() else 'CPU')


def _parse_llm_response(response: str) -> list:
    """
    Parse the LLM response to extract the list of relations.

    Args:
        response: Raw text response from the LLM.

    Returns:
        List of tuples (entity1, relation, entity2).
    """
    if not response:
        logger.debug("Empty response, returning empty list")
        return []

    logger.debug("Parsing LLM response (length: %d chars)", len(response))

    # Try to find a Python list in the response
    match = re.search(r'\[.*?\]', response, re.DOTALL)
    if match:
        list_str = match.group()
        logger.debug("Found list pattern: %s...", list_str[:150])
        try:
            relations = ast.literal_eval(list_str)
            if isinstance(relations, list):
                valid_relations = []
                for rel in relations:
                    if isinstance(rel, (list, tuple)) and len(rel) == 3:
                        entity1 = str(rel[0]).strip()
                        relation_type = str(rel[1]).strip()
                        entity2 = str(rel[2]).strip()
                        if entity1 and relation_type and entity2:
                            valid_relations.append((entity1, relation_type, entity2))
                            logger.debug("Extracted relation: (%s, %s, %s)",
                                         entity1, relation_type, entity2)
                        else:
                            logger.warning("Skipped empty relation: %s", rel)
                    else:
                        logger.warning("Skipped invalid relation format: %s", rel)

                logger.info("Successfully parsed %d relations", len(valid_relations))
                return valid_relations
        except (ValueError, SyntaxError) as e:
            logger.warning("Failed to parse list with ast.literal_eval: %s", e)

    # Fallback: try to extract tuples using regex
    logger.debug("Trying regex fallback for tuple extraction")
    tuple_pattern = (r'\(\s*["\'](.+?)["\']\s*,'
                     r'\s*["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\s*\)')
    matches = re.findall(tuple_pattern, response)
    if matches:
        result = [(m[0].strip(), m[1].strip(), m[2].strip()) for m in matches]
        logger.info("Regex fallback extracted %d relations", len(result))
        return result

    logger.warning("No relations found in response")
    return []


def extract_relations(text: str) -> list:
    """
    Extracts relationships from the input text using a local LLM.

    Args:
        text: Input text to analyze for relationships.

    Returns:
        List of tuples containing (entity1, relation, entity2).
    """
    if not text or not text.strip():
        logger.debug("Empty or None text provided, returning empty list")
        return []

    if len(text.strip()) < 10:
        logger.debug("Text too short (%d chars), returning empty list", len(text.strip()))
        return []

    logger.info("Starting relation extraction for text (length: %d chars)", len(text))

    try:
        _load_model()

        prompt = RELATION_PROMPT.format(text=text.strip())
        logger.debug("Prompt prepared (length: %d chars)", len(prompt))

        messages = [
            {"role": "system",
             "content": "Ты — система извлечения отношений из текста. "
                        "Отвечай только списком кортежей."},
            {"role": "user", "content": prompt},
        ]

        if hasattr(_tokenizer, 'apply_chat_template'):
            formatted_prompt = _tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.debug("Applied chat template to prompt")
        else:
            formatted_prompt = prompt
            logger.debug("Using raw prompt (no chat template available)")

        logger.debug("Generation parameters: max_new_tokens=%d, temperature=%.1f, top_p=%.1f",
                     MAX_NEW_TOKENS, TEMPERATURE, TOP_P)

        # FIX: Проверяем pad_token_id
        pad_token_id = _tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = getattr(_tokenizer, 'pad_token_id', 0)

        result = _generator(
            formatted_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=pad_token_id,
        )

        response = result[0]["generated_text"]
        logger.debug("Raw generated text length: %d chars", len(response))

        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):]
            logger.debug("Removed prompt prefix from response")

        logger.debug("Response to parse: %s...", response[:300])

        relations = _parse_llm_response(response)

        logger.info("Extraction complete. Found %d relations:", len(relations))
        for i, rel in enumerate(relations, 1):
            logger.info(" %d. %s → %s → %s", i, rel[0], rel[1], rel[2])

        return relations

    except Exception as e:
        logger.error("Error extracting relations: %s", e)
        logger.exception("Full traceback:")
        return []
    finally:
        # FIX: Очистка GPU памяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()