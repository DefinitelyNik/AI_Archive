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

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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

    logger.info(f"Loading relation extraction model: {MODEL_NAME}...")
    logger.debug(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.debug(f"Torch dtype: "
                 f"{'float16' if torch.cuda.is_available() else 'float32'}")

    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.debug("Tokenizer loaded successfully")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    logger.debug("Model loaded successfully")

    if not torch.cuda.is_available():
        model.to("cpu")
        logger.debug("Model moved to CPU")

    _generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        device=device,
    )

    logger.info(f"Model loaded successfully on "
                f"{'GPU' if torch.cuda.is_available() else 'CPU'}")
    logger.debug(f"Generator pipeline created with device={device}")


def _parse_llm_response(response: str) -> list:
    """
    Parse the LLM response to extract the list of relations.
    Handles various formats the model might return.

    Args:
        response (str): Raw text response from the LLM.

    Returns:
        list: List of tuples (entity1, relation, entity2).
    """
    if not response:
        logger.debug("Empty response, returning empty list")
        return []

    logger.debug(f"Parsing LLM response (length: {len(response)} chars)")
    logger.debug(f"Raw response preview: {response[:200]}...")

    # Try to find a Python list in the response
    # Look for patterns like [...] including multiline
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        list_str = match.group()
        logger.debug(f"Found list pattern: {list_str[:150]}...")
        try:
            relations = ast.literal_eval(list_str)
            if isinstance(relations, list):
                valid_relations = []
                for rel in relations:
                    if isinstance(rel, (list, tuple)) and len(rel) == 3:
                        # Ensure all elements are strings
                        entity1 = str(rel[0]).strip()
                        relation_type = str(rel[1]).strip()
                        entity2 = str(rel[2]).strip()
                        if entity1 and relation_type and entity2:
                            valid_relations.append((entity1, relation_type, entity2))
                            logger.debug(f"Extracted relation: "
                                         f"({entity1}, {relation_type}, {entity2})")
                        else:
                            logger.warning(f"Skipped empty relation: {rel}")
                    else:
                        logger.warning(f"Skipped invalid relation format: {rel}")

                logger.info(f"Successfully parsed {len(valid_relations)} relations")
                return valid_relations
        except (ValueError, SyntaxError) as e:
            logger.warning(f"Failed to parse list with ast.literal_eval: {e}")

    # Fallback: try to extract tuples using regex
    logger.debug("Trying regex fallback for tuple extraction")
    tuple_pattern = (r'\(\s*["\'](.+?)["\']\s*,'
                     r'\s*["\'](.+?)["\']\s*,\s*["\'](.+?)["\']\s*\)')
    matches = re.findall(tuple_pattern, response)
    if matches:
        result = [(m[0].strip(), m[1].strip(), m[2].strip()) for m in matches]
        logger.info(f"Regex fallback extracted {len(result)} relations")
        return result

    logger.warning("No relations found in response")
    return []


def extract_relations(text: str) -> list:
    """
    Extracts relationships from the input text using a local LLM.

    Uses Qwen2.5-3B-Instruct model to identify semantic relations such as
    parent-child, birth place, birth date, baptism, marriage, etc.

    Args:
        text (str): Input text to analyze for relationships.

    Returns:
        list: List of tuples containing (entity1, relation, entity2).

    Example:
        >>> relations = extract_relations('Иван родился в Москве в 1890 году.')
        >>> print(relations)
        [('Иван', 'место рождения', 'Москва'), ('Иван', 'дата рождения', '1890')]
    """
    if not text or not text.strip():
        logger.debug("Empty or None text provided, returning empty list")
        return []

    # Skip very short texts that are unlikely to contain relations
    if len(text.strip()) < 10:
        logger.debug(f"Text too short "
                     f"({len(text.strip())} chars), returning empty list")
        return []

    logger.info(f"Starting relation extraction for text (length: {len(text)} chars)")
    logger.debug(f"Input text preview: {text[:100]}...")

    try:
        _load_model()

        prompt = RELATION_PROMPT.format(text=text.strip())
        logger.debug(f"Prompt prepared (length: {len(prompt)} chars)")

        messages = [
            {"role": "system",
             "content": "Ты — система извлечения отношений из текста. "
                        "Отвечай только списком кортежей."},
            {"role": "user", "content": prompt},
        ]

        # Use chat template if available
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

        logger.debug(f"Generation parameters: max_new_tokens={MAX_NEW_TOKENS}, "
                    f"temperature={TEMPERATURE}, top_p={TOP_P}, "
                    f"repetition_penalty={REPETITION_PENALTY}")

        result = _generator(
            formatted_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

        response = result[0]["generated_text"]
        logger.debug(f"Raw generated text length: {len(response)} chars")

        if response.startswith(formatted_prompt):
            response = response[len(formatted_prompt):]
            logger.debug("Removed prompt prefix from response")

        logger.debug(f"Response to parse: {response[:300]}...")

        relations = _parse_llm_response(response)

        logger.info(f"Extraction complete. Found {len(relations)} relations:")
        for i, rel in enumerate(relations, 1):
            logger.info(f"  {i}. {rel[0]} → {rel[1]} → {rel[2]}")

        return relations

    except Exception as e:
        logger.error(f"Error extracting relations: {e}")
        logger.exception("Full traceback:")
        return []
