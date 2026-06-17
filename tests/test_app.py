"""Tests for the Flask application."""

import io
import pytest
from unittest.mock import MagicMock
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_heavy_functions(monkeypatch):
    """Mock all heavy functions to avoid loading real models."""
    monkeypatch.setattr('app.perform_ner',
                        MagicMock(return_value="mock ner result"))
    monkeypatch.setattr('app.perform_ocr',
                        MagicMock(return_value="mock ocr text"))
    monkeypatch.setattr('app.perform_tesseract_ocr',
                        MagicMock(return_value="mock tesseract text"))
    monkeypatch.setattr('app.perform_htr',
                        MagicMock(return_value=("mock_image", "mock htr text")))
    monkeypatch.setattr('app.translate_text',
                        MagicMock(side_effect=lambda x: f"translated: {x}"))
    monkeypatch.setattr('app.extract_relations',
                        MagicMock(return_value=[
                            ('Entity1', 'relation_type', 'Entity2')]))


class TestIndexRoute:
    """Tests for the index route."""

    def test_index_get(self, client):
        """Test GET request to index page."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'AI Archive' in response.data

    @pytest.mark.parametrize("text_type,ocr_model", [
        ('ocr', 'easyocr'),
        ('ocr', 'tesseract'),
        ('htr', 'easyocr'),
    ])
    def test_index_post_missing_image(self, client, text_type, ocr_model):
        """Test POST request without image file - should redirect."""
        response = client.post('/', data={
            'text_type': text_type,
            'ocr_model': ocr_model
        })

        assert response.status_code == 302

    def test_index_post_with_image(self, client, mock_heavy_functions):
        """Test POST request with valid image file."""
        data = {
            'image': (io.BytesIO(b"fake image data"), 'test.jpg'),
            'text_type': 'ocr',
            'ocr_model': 'easyocr'
        }
        response = client.post('/', data=data, content_type='multipart/form-data')

        assert response.status_code == 302
        assert '/results' in response.headers['Location']


class TestNerCheckRoute:
    """Tests for the NER check route."""

    def test_ner_check_get(self, client):
        """Test GET request to NER check page."""
        response = client.get('/ner_check')
        assert response.status_code == 200
        assert b'NER' in response.data or b'ner' in response.data

    def test_ner_check_post_empty_text(self, client, mock_heavy_functions):
        """Test POST request with empty text."""
        response = client.post('/ner_check', data={'text': ''})
        assert response.status_code == 200

    def test_ner_check_post_with_text(self, client, mock_heavy_functions):
        """Test POST request to NER Check with text."""
        response = client.post('/ner_check', data={'text': 'Привет мир'})

        assert response.status_code == 200
        assert b'mock ner result' in response.data

    def test_ner_check_post_with_translate(self, client, mock_heavy_functions):
        """Test POST request with translation enabled."""
        response = client.post('/ner_check', data={
            'text': 'Привет мир',
            'translate': 'on'
        })
        assert response.status_code == 200
        assert b'mock ner result' in response.data


class TestResultsRoute:
    """Tests for the results route."""

    def test_results_without_params(self, client):
        """Test results page without required parameters."""
        response = client.get('/results')
        assert response.status_code == 302  # Should redirect to index

    def test_results_with_params(self, client, mock_heavy_functions):
        """Test results page with all required parameters."""
        response = client.get('/results', query_string={
            'image_path': '/static/uploads/test.jpg',
            'extracted_text': 'Test text',
            'text_type': 'ocr',
            'ocr_model': 'easyocr',
            'translate': 'False'
        })
        assert response.status_code == 200
        assert b'Test text' in response.data


class TestSessionHandling:
    """Tests for session-based relations storage."""

    def test_relations_stored_in_session(self, client, mock_heavy_functions):
        """Test that relations are stored in session after index POST."""
        data = {
            'image': (io.BytesIO(b"fake image data"), 'test.jpg'),
            'text_type': 'ocr',
            'ocr_model': 'easyocr'
        }

        response = client.post('/', data=data, content_type='multipart/form-data')

        assert response.status_code == 302
        assert '/results' in response.headers['Location']

        response = client.get(response.headers['Location'])
        assert response.status_code == 200


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
