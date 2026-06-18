# 📝 AI Archive app
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🗒️ Description

This Flask web application processes images to extract text in Russian using OCR and HTR and identifies named entities (NER) such as persons, locations, organizations, and dates. Extracted text and ner-tags would be used to create graph database with people and connections between them.

---

## 🚀 Features

- **OCR Processing**: Extract text from uploaded images using EasyOCR.
- **HTR Processing**(soon): Extract text from uploaded images using trocr-base-handwritten-ru.
- **Named Entity Recognition (NER)**: Identify and classify entities (PER, LOC, ORG) using `slovnet` with `navec` embeddings.
- **Date Recognition**: Detect dates in various formats using regular expressions.
- **Visual Highlighting**: Entities are highlighted with distinct colors.
- **LLM Text Cleanup**: Optionally clean OCR/HTR output with a local Qwen model before NER and relation extraction.

---

## 🛠️ Technologies Used

- **Backend**: Flask
- **OCR**: EasyOCR
- **NER**: Slovnet, Navec
- **Frontend**: HTML, CSS, JavaScript
- **Models**: `EasyOCR`, `kazars24/trocr-base-handwritten-ru`, `slovnet_ner`

---

## 🎌 Supported languages

- **Russian**
- **English** - coming soon

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup
Install using `pip`:
```bash
pip install git+https://github.com/DefinitelyNik/AI_Archive.git
pip install -r requirements.txt
```

## 📞 Support
For questions and support, please open an issue in the GitHub repository.

## LLM Text Cleanup

The application can clean OCR/HTR output with a local Ollama model. Enable
**Clean OCR/HTR text with LLM** in the upload form to show both the raw
extracted text and the cleaned version. When cleanup is enabled, NER and
relation extraction run on the cleaned text.

By default cleanup uses `qwen3.5:9b` through the Ollama HTTP API at
`http://localhost:11434`. Install Ollama, pull the model, and keep Ollama
running while the Flask app is running:

```bash
ollama pull qwen3.5:9b
ollama serve
```

The model requires several GB of disk space and can require significant
RAM/VRAM at runtime. If Ollama is unavailable or generation fails, the
application keeps the original text so the main OCR/HTR pipeline remains
usable.

Optional environment variables:

```bash
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_TIMEOUT=180
```

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
