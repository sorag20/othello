"""Flask application factory for the Othello AI application."""

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS

from config import Config, DevelopmentConfig, ProductionConfig, TestingConfig

logger = logging.getLogger(__name__)

# Map environment names to config classes
config_map = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
}


def create_app(config_class=None):
    """Create and configure the Flask application.

    Args:
        config_class: Configuration class to use. If None, selects based
            on the FLASK_ENV environment variable (defaults to DevelopmentConfig).

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Determine configuration
    if config_class is None:
        env = os.environ.get('FLASK_ENV', 'development')
        config_class = config_map.get(env, DevelopmentConfig)

    app.config.from_object(config_class)

    # Enable CORS for frontend communication
    CORS(app)

    # Initialize database
    from models.database import init_db
    init_db(app)

    # Register blueprints
    from routes.game_routes import game_bp
    app.register_blueprint(game_bp, url_prefix='/api')

    # Register global error handlers
    _register_error_handlers(app)

    return app


def _register_error_handlers(app):
    """Register global error handlers to prevent internal details from leaking."""

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            'error': {
                'code': 'INVALID_PARAMETERS',
                'message': str(e.description) if hasattr(e, 'description') else 'Bad request'
            }
        }), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            'error': {
                'code': 'GAME_NOT_FOUND',
                'message': 'The requested resource was not found'
            }
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({
            'error': {
                'code': 'METHOD_NOT_ALLOWED',
                'message': 'The HTTP method is not allowed for this endpoint'
            }
        }), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.error('Internal server error: %s', e)
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'An internal error occurred'
            }
        }), 500

    @app.errorhandler(Exception)
    def unhandled_exception(e):
        logger.exception('Unhandled exception: %s', e)
        return jsonify({
            'error': {
                'code': 'INTERNAL_ERROR',
                'message': 'An internal error occurred'
            }
        }), 500


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
