import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_test_page_loads(client):
    """Проверка доступности тестовой страницы."""
    response = client.get('/test_page')
    assert response.status_code == 200
    assert b'Тестовая страница' in response.data
    assert b'Проверки новых функций' in response.data

def test_index_page_loads(client):
    """Проверка доступности главной страницы."""
    response = client.get('/')
    assert response.status_code == 200
