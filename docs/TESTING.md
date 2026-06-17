# Testing and Quality

The project uses Pytest for tests and Flake8 for style checks.

## Run Tests

```bash
pytest tests/
```

Expected result for a healthy branch:

```text
all tests passed
```

## Run Flake8

```bash
flake8 .
```

The Flake8 configuration is stored in `.flake8`.

## Test Areas

The test suite covers:

- Flask route availability and redirects.
- OCR helper behavior.
- HTR line grouping logic.
- NER/date detection behavior.
- Pre-revolutionary spelling translation.
- API response behavior.

## Model Handling in Tests

Some project dependencies load large pretrained models. Tests should mock heavy
model calls when possible so CI can run reliably and quickly.

Examples of functions that are commonly mocked in route tests:

- `perform_ocr`
- `perform_htr`
- `perform_ner`
- `extract_relations`

## Continuous Integration

GitHub Actions runs checks on pushes and pull requests to `master` or `main`.
The workflow:

1. Installs dependencies.
2. Runs Pytest.
3. Runs Flake8.
4. Builds a Docker image after successful checks.
5. Pushes the Docker image to Docker Hub on the main branch.

Docker Hub publishing requires:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

