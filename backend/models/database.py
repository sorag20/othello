"""Database initialization.

Provides the SQLAlchemy database instance used by all models.
The db instance is initialized with the Flask app via db.init_app().
"""

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def init_db(app):
    """Initialize the database with the Flask application.

    Binds the SQLAlchemy instance to the app and creates all tables
    if they do not already exist.

    Args:
        app: The Flask application instance.
    """
    db.init_app(app)
    with app.app_context():
        # Import models to ensure they are registered before create_all
        from models.game_session import GameSession  # noqa: F401
        db.create_all()
