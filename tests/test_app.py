from unittest.mock import patch


def test_index_get(client):
    """Тест GET запроса к главной странице"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Upload an image' in response.data


def test_ner_check_get(client):
    """Тест GET запроса к странице NER Check"""
    response = client.get('/ner_check')
    assert response.status_code == 200
    assert b'Enter text to analyze' in response.data


def test_results_get_without_params(client):
    """Тест GET запроса к /results без параметров"""
    response = client.get('/results')
    assert response.status_code == 302  # Редирект на /


def test_index_post_no_file(client):
    """Тест POST запроса к главной странице без файла"""
    response = client.post('/', data={'text_type': 'ocr'})
    assert response.status_code == 302  # Редирект на /


def test_index_post_invalid_file(client):
    """Тест POST запроса с неправильным файлом"""
    response = client.post(
        '/',
        data={
            'image': (b'test', 'test.txt'),
            'text_type': 'ocr'
        },
        content_type='multipart/form-data'
    )
    assert response.status_code == 302


def test_ner_check_post_empty_text(client):
    """Тест POST запроса к NER Check с пустым текстом"""
    response = client.post('/ner_check', data={'text': ''})
    assert response.status_code == 200
    assert b'NER Analysis Results' not in response.data


def test_test_page_get(client):
    """Тест GET запроса к тестовой странице"""
    response = client.get('/test_page')
    assert response.status_code == 200
    assert b'Тестовая страница' in response.data


@patch("app.perform_ner", return_value="mock ner result")
def test_ner_check_post_with_text(mock_ner, client):
    """Тест POST запроса к NER Check с текстом"""
    response = client.post('/ner_check', data={'text': 'Привет мир'})

    mock_ner.assert_called_once()
    assert response.status_code == 200
    assert b'NER Analysis Results' in response.data


@patch("app.perform_ocr", return_value="mock ocr text")
def test_index_post_with_image_ocr(mock_ocr, client):
    """Тест POST запроса с изображением и OCR"""
    with open('tests/fixtures/test_image_ner.jpg', 'rb') as img:
        response = client.post(
            '/',
            data={
                'image': (img, 'test_image.jpg'),
                'text_type': 'ocr'
            },
            content_type='multipart/form-data'
        )

    mock_ocr.assert_called_once()
    assert response.status_code == 302  # Редирект на /results


@patch("app.perform_htr", return_value=([], "mock htr text"))
def test_index_post_with_image_htr(mock_htr, client):
    """Тест POST запроса с изображением и HTR"""
    with open('tests/fixtures/test_image_htr.jpg', 'rb') as img:
        response = client.post(
            '/',
            data={
                'image': (img, 'test_image.jpg'),
                'text_type': 'htr'
            },
            content_type='multipart/form-data'
        )

    mock_htr.assert_called_once()
    assert response.status_code == 302  # Редирект на /results


@patch("app.perform_ocr", return_value="mock ocr text")
def test_index_post_with_translate(mock_ocr, client):
    """Тест POST запроса с флагом перевода"""
    with open('tests/fixtures/test_image_ner.jpg', 'rb') as img:
        response = client.post(
            '/',
            data={
                'image': (img, 'test_image.jpg'),
                'text_type': 'ocr',
                'translate': '1'
            },
            content_type='multipart/form-data'
        )

    mock_ocr.assert_called_once()
    assert response.status_code == 302  # Редирект на /results


@patch("app.perform_ocr", return_value="mock ocr text")
def test_index_post_with_translate(mock_ocr, client):
    """Тест POST запроса с флагом перевода"""
    with open('tests/fixtures/test_image_ner.jpg', 'rb') as img:
        response = client.post(
            '/',
            data={
                'image': (img, 'test_image.jpg'),
                'text_type': 'ocr',
                'translate': '1'
            },
            content_type='multipart/form-data'
        )

    mock_ocr.assert_called_once()
    assert response.status_code == 302  # Редирект на /results
