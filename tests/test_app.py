"""Tests for the Flask application."""

import io
import pytest
from unittest.mock import MagicMock
from app import app, db


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SECRET_KEY'] = 'test-secret-key'

    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.drop_all()


@pytest.fixture
def mock_heavy_functions(monkeypatch):
    """Mock all heavy functions."""
    monkeypatch.setattr('app.perform_ner', MagicMock(return_value="mock ner result"))
    monkeypatch.setattr('app.perform_ocr', MagicMock(return_value="mock ocr text"))
    monkeypatch.setattr('app.perform_tesseract_ocr', MagicMock(return_value="mock tesseract text"))
    monkeypatch.setattr('app.perform_htr', MagicMock(return_value=("mock_image", "mock htr text")))
    monkeypatch.setattr('app.translate_text', MagicMock(side_effect=lambda x: f"translated: {x}"))
    monkeypatch.setattr('app.extract_relations', MagicMock(return_value=[
        ('Entity1', 'relation_type', 'Entity2')
    ]))


@pytest.fixture
def authenticated_client(client):
    """Create an authenticated test client."""
    from models import User

    # Create test user
    user = User(username='testuser')
    user.set_password('testpass')
    db.session.add(user)
    db.session.commit()

    # Login
    client.post('/login', data={
        'username': 'testuser',
        'password': 'testpass'
    })

    return client


class TestAuthentication:
    """Tests for authentication routes."""

    def test_login_page(self, client):
        response = client.get('/login')
        assert response.status_code == 200

    def test_register_page(self, client):
        response = client.get('/register')
        assert response.status_code == 200

    def test_login_success(self, client):
        from models import User
        user = User(username='testuser')
        user.set_password('testpass')
        db.session.add(user)
        db.session.commit()

        response = client.post('/login', data={
            'username': 'testuser',
            'password': 'testpass'
        }, follow_redirects=True)
        assert response.status_code == 200

    def test_logout(self, authenticated_client):
        response = authenticated_client.get('/logout', follow_redirects=True)
        assert response.status_code == 200


class TestProtectedRoutes:
    """Tests for routes that require authentication."""

    def test_index_requires_login(self, client):
        response = client.get('/')
        assert response.status_code == 302  # Redirect to login

    def test_index_authenticated(self, authenticated_client):
        response = authenticated_client.get('/')
        assert response.status_code == 200

    def test_my_results_requires_login(self, client):
        response = client.get('/my_results')
        assert response.status_code == 302

    def test_my_results_authenticated(self, authenticated_client):
        response = authenticated_client.get('/my_results')
        assert response.status_code == 200


class TestProcessing:
    """Tests for image processing."""

    def test_process_requires_login(self, client):
        response = client.post('/process')
        assert response.status_code == 302

    def test_process_with_image(self, authenticated_client, mock_heavy_functions):
        data = {
            'image': (io.BytesIO(b"fake image data"), 'test.jpg'),
            'text_type': 'ocr',
            'ocr_model': 'easyocr'
        }
        response = authenticated_client.post('/process',
                                            data=data,
                                            content_type='multipart/form-data')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'result_id' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
