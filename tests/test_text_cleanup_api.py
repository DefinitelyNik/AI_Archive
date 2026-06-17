from unittest.mock import patch


@patch("app.clean_text", return_value="cleaned text")
def test_api_clean_text(mock_clean, client):
    response = client.post('/api/clean-text', json={'text': 'raw text'})

    assert response.status_code == 200
    assert response.get_json() == {'cleaned_text': 'cleaned text'}
    mock_clean.assert_called_once_with('raw text')


def test_api_clean_text_empty_text(client):
    response = client.post('/api/clean-text', json={'text': '   '})

    assert response.status_code == 400
    assert response.get_json() == {'error': 'Text is required'}


def test_api_clean_text_requires_json_object(client):
    response = client.post('/api/clean-text', json=['raw text'])

    assert response.status_code == 400
    assert response.get_json() == {'error': 'JSON object is required'}


@patch("app.clean_text", return_value="cleaned ocr text")
@patch("app.extract_relations", return_value=[])
@patch("app.perform_ner", return_value="mock ner result")
@patch("app.perform_ocr", return_value="mock ocr text")
def test_index_post_with_clean_text(mock_ocr,
                                    mock_ner,
                                    mock_relations,
                                    mock_clean,
                                    client):
    with open('tests/fixtures/test_image_ner.jpg', 'rb') as img:
        response = client.post(
            '/',
            data={
                'image': (img, 'test_image.jpg'),
                'text_type': 'ocr',
                'clean_text': '1'
            },
            content_type='multipart/form-data'
        )

    assert response.status_code == 200
    assert b'Raw extracted text' in response.data
    assert b'Cleaned text' in response.data
    mock_ocr.assert_called_once()
    mock_clean.assert_called_once_with("mock ocr text")
    mock_ner.assert_called_once_with("cleaned ocr text")
    mock_relations.assert_called_once_with("cleaned ocr text")


@patch("app.clean_text", return_value="cleaned api text")
@patch("app.extract_relations", return_value=[])
@patch("app.perform_ner", return_value="mock ner result")
@patch("app.perform_ocr", return_value="mock ocr text")
def test_api_process_with_clean_text(mock_ocr,
                                     mock_ner,
                                     mock_relations,
                                     mock_clean,
                                     client):
    with open('tests/fixtures/test_image_ner.jpg', 'rb') as img:
        response = client.post(
            '/api/process',
            data={
                'image': (img, 'test_image.jpg'),
                'text_type': 'ocr',
                'clean_text': '1'
            },
            content_type='multipart/form-data'
        )

    data = response.get_json()
    assert response.status_code == 200
    assert data['raw_text'] == 'mock ocr text'
    assert data['text'] == 'cleaned api text'
    assert data['annotated_text'] == 'mock ner result'
    mock_clean.assert_called_once_with('mock ocr text')
    mock_ner.assert_called_once_with('cleaned api text')
    mock_relations.assert_called_once_with('cleaned api text')


@patch("app.extract_relations", return_value=[])
@patch("app.perform_ner", return_value="mock ner result")
@patch("app.perform_ocr", return_value="mock ocr text")
def test_index_post_with_non_ascii_filename(mock_ocr,
                                            mock_ner,
                                            mock_relations,
                                            client):
    with open('tests/fixtures/test_image_ner.jpg', 'rb') as img:
        response = client.post(
            '/',
            data={
                'image': (img, 'Скриншот.jpg'),
                'text_type': 'ocr'
            },
            content_type='multipart/form-data'
        )

    assert response.status_code == 200
    mock_ocr.assert_called_once()
    saved_path = mock_ocr.call_args.args[0]
    assert saved_path.endswith('.jpg')
    assert 'Скриншот' not in saved_path
    mock_ner.assert_called_once_with('mock ocr text')
    mock_relations.assert_called_once_with('mock ocr text')
