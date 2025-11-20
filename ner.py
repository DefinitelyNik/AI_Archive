from navec import Navec
from slovnet import NER
import re

navec_path = 'navec_news_v1_1B_250K_300d_100q.tar'
navec = Navec.load(navec_path)

ner_model = NER.load('slovnet_ner_news_v1.tar')
ner_model.navec(navec)

def find_dates(text):
    date_pattern = r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\b'
    iso_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
    year_pattern = r'\b(\d{4})\s*(?:г\.?|год|года)\b'
    year_decade_pattern = r'\b(?:в\s+)?(\d{3}0)-[хxs]\s*(?:годах|годов|году|год|гг\.?)?\b'

    dates = []
    for match in re.finditer(date_pattern, text):
        dates.append((match.start(), match.end(), 'date'))
    for match in re.finditer(iso_pattern, text):
        dates.append((match.start(), match.end(), 'date'))
    for match in re.finditer(year_pattern, text):
        dates.append((match.start(), match.end(), 'date'))
    for match in re.finditer(year_decade_pattern, text):
        dates.append((match.start(), match.end(), 'date'))

    dates.sort(key=lambda x: x[0])
    return dates

def annotate_text(markup):
    ner_spans = [(span.start, span.stop, span.type.lower()) for span in markup.spans]

    dates = find_dates(markup.text)

    all_spans = ner_spans + dates
    all_spans.sort(key=lambda x: x[0])

    tokens = []
    last = 0
    for start, stop, label in all_spans:
        if last < start:
            tokens.append(markup.text[last:start])
        entity_text = markup.text[start:stop]
        tokens.append(f'<mark class="ner-{label}">{entity_text}</mark>')
        last = stop
    if last < len(markup.text):
        tokens.append(markup.text[last:])
    return ''.join(tokens)

def perform_ner(text):
    markup = ner_model(text)
    return annotate_text(markup)