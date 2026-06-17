# API Documentation

This document describes the application routes and API endpoints available in
the current `master` branch.

## Web Routes

### `GET /`

Displays the image upload form.

### `POST /`

Processes an uploaded image through the web interface.

Form fields:

- `image`: required image file.
- `text_type`: required, either `ocr` or `htr`.
- `ocr_model`: optional, either `easyocr` or `tesseract`.
- `translate`: optional checkbox flag.

Behavior:

1. Saves the uploaded image to `static/uploads`.
2. Runs OCR or HTR depending on `text_type`.
3. Applies optional spelling translation.
4. Runs NER and relation extraction.
5. Redirects to `/results`.

### `GET /results`

Displays the processed image, annotated text, and extracted relations.

Query parameters:

- `image_path`
- `extracted_text`
- `text_type`
- `ocr_model`
- `translate`
- `relations`

If required result data is missing, the route redirects back to `/`.

### `GET /ner_check`

Displays a text input form for checking NER output without uploading an image.

### `POST /ner_check`

Runs NER and relation extraction on submitted text.

Form fields:

- `text`: text to analyze.
- `translate`: optional checkbox flag.

## JSON API

### `POST /api/process`

Processes an uploaded image and returns JSON.

Consumes:

```text
multipart/form-data
```

Form fields:

- `image`: required image file.
- `text_type`: required, either `ocr` or `htr`.
- `ocr_model`: optional, either `easyocr` or `tesseract`.
- `translate`: optional flag.

Success response:

```json
{
  "text": "extracted text",
  "annotated_text": "HTML with NER tags",
  "relations": [
    ["entity1", "relation", "entity2"]
  ]
}
```

Error responses:

- `400 {"error": "Missing required fields"}`
- `400 {"error": "No file selected"}`
- `400 {"error": "Invalid text_type"}`

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/process \
  -F "image=@sample.jpg" \
  -F "text_type=ocr" \
  -F "ocr_model=easyocr"
```

## Swagger

Swagger UI is available at:

```text
http://127.0.0.1:5000/apidocs
```

## Feature Branch APIs

The LLM cleanup branch adds `/api/clean-text`. It is intentionally documented as
a feature-branch addition until that branch is merged into `master`.

