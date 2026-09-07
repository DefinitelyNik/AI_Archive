"""Named Entity Recognition and text preprocessing module."""

import html
import os
import re
import threading

from navec import Navec
from slovnet import NER

# Models are loaded only on first use and protected against concurrent loading.
_navec = None
_ner_model = None
_model_lock = threading.Lock()

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
NAVEC_PATH = os.environ.get(
    "NAVEC_PATH", os.path.join(_MODULE_DIR, "navec_news_v1_1B_250K_300d_100q.tar")
)
NER_MODEL_PATH = os.environ.get(
    "NER_MODEL_PATH", os.path.join(_MODULE_DIR, "slovnet_ner_news_v1.tar")
)


def get_ner_model():
    """Lazy-load and cache NER models safely for concurrent web requests."""
    global _navec, _ner_model

    with _model_lock:
        if _navec is None:
            if not os.path.exists(NAVEC_PATH):
                raise FileNotFoundError(f"Navec model not found: {NAVEC_PATH}")
            _navec = Navec.load(NAVEC_PATH)

        if _ner_model is None:
            if not os.path.exists(NER_MODEL_PATH):
                raise FileNotFoundError(f"NER model not found: {NER_MODEL_PATH}")
            _ner_model = NER.load(NER_MODEL_PATH)
            _ner_model.navec(_navec)

        return _ner_model


def translate_text(text: str) -> str:
    """Normalize pre-revolutionary Russian spelling to contemporary spelling."""
    if not text:
        return ""

    replacements = {
        "ѣ": "е",
        "Ѣ": "Е",
        "i": "и",
        "I": "И",
        "І": "И",
        "і": "и",
        "ѵ": "и",
        "Ѵ": "И",
        "ѳ": "ф",
        "Ѳ": "Ф",
        "ћ": "e",
        "y": "ы",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'ъ(\s|[.,;:!?—–\n"\'\]])', r"\1", text)
    text = re.sub(r'(\s|[.,;:!?—–\n"\'\[(])ъ', r"\1", text)
    text = text.replace("ъ", "").replace("Ъ", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def find_dates(text: str) -> list:
    """Find dates in common numerical and Russian-language formats."""
    if not text:
        return []

    patterns = (
        r"\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{4})\s*(?:г\.?|год|года|году|годах)\b",
        r"\b(?:в\s+)?(\d{3}0)-[хxs]\s*(?:годах|годов|году|год|гг\.?)?\b",
    )

    dates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            dates.append((match.start(), match.end(), "date"))
    return sorted(dates, key=lambda item: item[0])


def annotate_text(markup) -> str:
    """Return escaped source text with controlled HTML NER/date annotations."""
    ner_spans = [
        (span.start, span.stop, span.type.lower()) for span in markup.spans
    ]
    all_spans = sorted(ner_spans + find_dates(markup.text), key=lambda item: item[0])

    filtered = []
    last_end = -1
    for start, stop, label in all_spans:
        if start >= last_end:
            filtered.append((start, stop, label))
            last_end = stop

    tokens = []
    last = 0
    allowed_labels = {"per", "loc", "org", "date"}
    for start, stop, label in filtered:
        if last < start:
            tokens.append(html.escape(markup.text[last:start]))

        safe_label = label if label in allowed_labels else "org"
        entity_text = html.escape(markup.text[start:stop])
        tokens.append(f'<mark class="ner-{safe_label}">{entity_text}</mark>')
        last = stop

    if last < len(markup.text):
        tokens.append(html.escape(markup.text[last:]))
    return "".join(tokens)


def perform_ner(text: str) -> str:
    """Run NER and return escaped, annotated HTML for the user interface."""
    if not text or not text.strip():
        return ""

    model = get_ner_model()
    return annotate_text(model(text))
