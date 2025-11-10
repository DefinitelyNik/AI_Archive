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
