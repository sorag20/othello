"""Application configuration.

Configuration is loaded from environment variables with sensible defaults
for development. Production deployments should set all environment variables.
"""

import os


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'mysql+pymysql://othello:othello@localhost:3306/othello'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # AI Engine settings
    AI_MODEL_PATH = os.environ.get('AI_MODEL_PATH', 'models/weights/model.pt')
    AI_SIMULATIONS = int(os.environ.get('AI_SIMULATIONS', '800'))
    AI_TIMEOUT = float(os.environ.get('AI_TIMEOUT', '5.0'))

    # Edax engine settings
    EDAX_PATH = os.environ.get('EDAX_PATH', '')
    EDAX_EVAL_PATH = os.environ.get('EDAX_EVAL_PATH', '')
    EDAX_BOOK_PATH = os.environ.get('EDAX_BOOK_PATH', '')

    # Minimax engine settings
    MINIMAX_TIME_LIMIT = float(os.environ.get('MINIMAX_TIME_LIMIT', '3.0'))
    MINIMAX_ENDGAME_EMPTY_THRESHOLD = int(os.environ.get('MINIMAX_ENDGAME_EMPTY_THRESHOLD', '10'))

    # Game session settings
    SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', '300'))


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
