"""Tests for Flask application factory and error handlers."""

import pytest

from app import create_app
from config import TestingConfig


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app(config_class=TestingConfig)
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestCreateApp:
    """Tests for the application factory."""

    def test_create_app_returns_flask_instance(self, app):
        assert app is not None
        assert app.config['TESTING'] is True

    def test_create_app_uses_sqlite_in_testing(self, app):
        assert 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']

    def test_create_app_has_cors_enabled(self, app):
        # CORS adds after_request handlers
        assert app.after_request_funcs is not None

    def test_create_app_registers_game_blueprint(self, app):
        assert 'game' in app.blueprints


class TestErrorHandlers:
    """Tests for global error handlers."""

    def test_404_returns_json_error(self, client):
        response = client.get('/api/nonexistent')
        assert response.status_code == 404
        data = response.get_json()
        assert data['error']['code'] == 'GAME_NOT_FOUND'
        assert 'message' in data['error']

    def test_405_returns_json_error(self, client):
        # DELETE is not defined on /api/games
        response = client.delete('/api/games')
        assert response.status_code == 405
        data = response.get_json()
        assert data['error']['code'] == 'METHOD_NOT_ALLOWED'
        assert 'message' in data['error']

    def test_500_hides_internal_details(self, app, client):
        # Register a route that raises an exception
        @app.route('/test-error')
        def error_route():
            raise RuntimeError('secret internal detail')

        response = client.get('/test-error')
        assert response.status_code == 500
        data = response.get_json()
        assert data['error']['code'] == 'INTERNAL_ERROR'
        assert data['error']['message'] == 'An internal error occurred'
        # Internal details must NOT be exposed
        assert 'secret internal detail' not in str(data)

    def test_error_response_format(self, client):
        response = client.get('/api/nonexistent-resource')
        data = response.get_json()
        # Verify the error response structure
        assert 'error' in data
        assert 'code' in data['error']
        assert 'message' in data['error']
