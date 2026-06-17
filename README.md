# AI Archive

AI Archive is a Flask web application for working with archival images and text. It combines OCR, HTR, NER, date detection, and relation extraction to help turn scanned material into structured, searchable data.

## Features

- OCR for printed text with EasyOCR or Tesseract.
- HTR for handwritten text with TrOCR.
- Named entity recognition for persons, locations, organizations, and dates.
- Date detection with regular expressions.
- Relation extraction pipeline for archival text.
- Swagger API documentation.
- Docker support and GitHub Actions CI.

## Technology Stack

- Python 3.10+
- Flask
- EasyOCR
- Tesseract OCR
- Transformers
- Slovnet and Navec
- Pytest
- Flake8
- Docker
- GitHub Actions

## Repository Structure

```text
app.py                    Flask routes and API endpoints
ocr.py                    EasyOCR integration
tesseract_ocr.py          Tesseract OCR integration
htr.py                    Handwritten text recognition
ner.py                    Named entity recognition
relations.py              Relation extraction
text_cleanup.py           LLM cleanup feature branch module
templates/                HTML templates
static/                   Static files and uploads
tests/                    Unit tests and fixtures
Dockerfile                Docker image definition
.github/workflows/ci.yml  CI pipeline
```

## Project Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Testing and Quality](docs/TESTING.md)

## Local Run

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python app.py
```

5. Open:

```text
http://127.0.0.1:5000
```

Swagger is available at:

```text
http://127.0.0.1:5000/apidocs
```

## Docker

Build the image:

```bash
docker build -t ai-archive .
```

Run the container:

```bash
docker run --rm -p 5000:5000 ai-archive
```

Open:

```text
http://127.0.0.1:5000
```

- **OCR Processing**: Extract text from uploaded images using EasyOCR.
- **HTR Processing**(soon): Extract text from uploaded images using trocr-base-handwritten-ru.
- **Named Entity Recognition (NER)**: Identify and classify entities (PER, LOC, ORG) using `slovnet` with `navec` embeddings.
- **Date Recognition**: Detect dates in various formats using regular expressions.
- **Visual Highlighting**: Entities are highlighted with distinct colors.
- **LLM Text Cleanup**: Optionally clean OCR/HTR output with a local Qwen model before NER and relation extraction.

### POST /api/process

Processes an uploaded image and returns extracted text, annotated text, and relations.

Form fields:

- `image`: image file.
- `text_type`: `ocr` or `htr`.
- `ocr_model`: optional, `easyocr` or `tesseract`.
- `translate`: optional.

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/process \
  -F "image=@sample.jpg" \
  -F "text_type=ocr" \
  -F "ocr_model=easyocr"
```

## Tests and Quality

Run tests:

```bash
pytest tests/
```

Run linting:

```bash
flake8 .
```

## 📞 Support
For questions and support, please open an issue in the GitHub repository.

## LLM Text Cleanup

The application can clean OCR/HTR output with the local
`Qwen/Qwen2.5-3B-Instruct` model. Enable **Clean OCR/HTR text with LLM** in the
upload form to show both the raw extracted text and the cleaned version. When
cleanup is enabled, NER and relation extraction run on the cleaned text.

The model is loaded on the first cleanup request and can require significant
RAM/VRAM and startup time, especially in Docker or CPU-only environments. If
model loading or generation fails, the application keeps the original text so
the main OCR/HTR pipeline remains usable.

Clean already extracted text through the API:

```bash
curl -X POST http://localhost:5000/api/clean-text \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"raw OCR text\"}"
```

Image processing also supports the optional `clean_text` form field:

```bash
curl -X POST http://localhost:5000/api/process \
  -F "image=@sample.jpg" \
  -F "text_type=ocr" \
  -F "clean_text=1"
```
