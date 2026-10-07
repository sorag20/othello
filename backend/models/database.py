"""Database initialization.

Provides the SQLAlchemy database instance used by all models.
The db instance is initialized with the Flask app via db.init_app().
"""

import logging

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text

logger = logging.getLogger(__name__)

db = SQLAlchemy()


def init_db(app):
    """Initialize the database with the Flask application.

    Binds the SQLAlchemy instance to the app and creates all tables
    if they do not already exist, then applies any missing column migrations.

    Args:
        app: The Flask application instance.
    """
    db.init_app(app)
    with app.app_context():
        from models.game_session import GameSession  # noqa: F401
        db.create_all()
        _migrate_columns()


def _migrate_columns():
    """Add any columns that exist in the model but not yet in the DB table.

    This is a lightweight alternative to Alembic for simple additive migrations.
    Only handles adding new nullable/default columns; does not drop or alter existing ones.
    """
    # Columns to ensure exist: (column_name, DDL_definition)
    _REQUIRED_COLUMNS = [
        ('opponent_type', "VARCHAR(32) NOT NULL DEFAULT 'internal'"),
        ('edax_level',    "INT NULL"),
    ]

    inspector = inspect(db.engine)
    existing = {col['name'] for col in inspector.get_columns('game_sessions')}

    with db.engine.begin() as conn:
        for col_name, col_def in _REQUIRED_COLUMNS:
            if col_name not in existing:
                conn.execute(text(
                    f'ALTER TABLE game_sessions ADD COLUMN {col_name} {col_def}'
                ))
                logger.info("Migration: added column '%s' to game_sessions", col_name)
