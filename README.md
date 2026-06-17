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

## API

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

The repository CI runs tests and linting on pushes and pull requests to `master` or `main`.

## Continuous Integration and Docker Hub

The workflow in `.github/workflows/ci.yml` performs:

- dependency installation;
- unit tests with Pytest;
- PEP8 style checks with Flake8;
- Docker image build;
- Docker image publishing to Docker Hub on pushes to the main branch.

Docker Hub publishing requires these GitHub repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Development Workflow

- Work on separate feature branches.
- Open pull requests into `master`.
- Request code review before merging.
- Wait for CI checks before merging.
- Keep code style PEP8-compliant.

## Course Notes

This project satisfies the course workflow through separate branches, pull requests, code review, tests, linting, Docker configuration, and CI automation.

## Feature Branch Notes

Additional features are developed in separate branches before merge. For
example, the LLM cleanup feature branch adds optional text cleanup and the
`/api/clean-text` endpoint.
