# Architecture

AI Archive is built as a Flask application with a simple request-response
pipeline. The application accepts images or text, extracts text, annotates named
entities, detects dates, and prepares relation data for display or API output.

## Processing Pipeline

```text
Image upload
  -> OCR or HTR
  -> optional pre-revolutionary spelling translation
  -> NER and date detection
  -> relation extraction
  -> HTML results page or JSON API response
```

## Main Modules

### `app.py`

`app.py` is the Flask entry point. It defines web routes, API routes, upload
handling, Swagger configuration, and rendering of result pages.

Important routes:

- `/`: image upload and processing form.
- `/results`: HTML page with the processed image and annotated text.
- `/ner_check`: text-only page for checking NER output.
- `/api/process`: JSON API for image processing.

### `ocr.py`

`ocr.py` wraps EasyOCR. It creates an EasyOCR reader and exposes
`perform_ocr(image_path)`, which returns text extracted from a printed-text
image.

### `tesseract_ocr.py`

`tesseract_ocr.py` wraps Tesseract OCR. It is used as an alternative OCR model
for printed text. The module configures a Tesseract executable path on Windows
when a known installation path is found.

### `htr.py`

`htr.py` handles handwritten text recognition. It uses EasyOCR to detect text
regions, groups them into lines, and then applies the TrOCR model
`kazars24/trocr-base-handwritten-ru` to recognize handwritten Russian text.

### `ner.py`

`ner.py` loads Navec embeddings and a Slovnet NER model. It also contains:

- `translate_text`: normalizes selected pre-revolutionary Russian characters.
- `find_dates`: detects dates and years with regular expressions.
- `perform_ner`: returns HTML with NER and date markup.

### `relations.py`

`relations.py` provides relation extraction output. In the current `master`
branch it returns demonstration relation tuples. A separate feature branch adds
LLM-based relation extraction.

### `templates/`

HTML templates define the web interface:

- `base.html`: common layout.
- `index.html`: upload form.
- `results.html`: image and text processing results.
- `ner_check.html`: text-only NER check page.

### `tests/`

Tests cover Flask routes, OCR/HTR helper behavior, NER/date logic, translation,
and API responses. Heavy model calls are mocked where needed so tests can run in
CI.

## Data Flow

The web pipeline stores uploaded images in `static/uploads`. Extracted text is
passed to NER and relation extraction. The web interface renders annotated HTML,
while `/api/process` returns a JSON response with raw text, annotated text, and
relations.

## Model Usage

The project uses pretrained models rather than training custom models:

- EasyOCR for printed text OCR.
- Tesseract OCR as an alternative OCR engine.
- TrOCR for handwritten Russian text.
- Slovnet and Navec for named entity recognition.

Feature branches may add additional LLM-based processing, such as relation
extraction or text cleanup.

