"""SQLAlchemy model for game sessions.

Provides the GameSession model representing an Othello game,
including board state, move history, result, and timestamps.
"""

import uuid

from sqlalchemy import Column, String, Enum, JSON, DateTime, Index, Integer
from sqlalchemy.sql import func

from models.database import db


class GameSession(db.Model):
    """Represents a single Othello game session stored in the database."""

    __tablename__ = 'game_sessions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    player_id = Column(String(255), nullable=True)
    result = Column(Enum('black_win', 'white_win', 'draw', name='game_result'), nullable=True)
    move_sequence = Column(JSON, nullable=False, default=list)
    status = Column(
        Enum('in_progress', 'finished', name='game_status'),
        nullable=False,
        default='in_progress'
    )
    board_state = Column(JSON, nullable=False)
    current_turn = Column(
        Enum('black', 'white', name='turn_color'),
        nullable=False,
        default='black'
    )
    opponent_type = Column(String(32), nullable=False, default='internal')
    edax_level = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=False, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index('idx_player_ended', 'player_id', ended_at.desc()),
        Index('idx_status', 'status'),
    )

    def to_dict(self):
        """Serialize the game session to a dictionary.

        Returns:
            dict: A dictionary representation of the game session.
        """
        return {
            'id': self.id,
            'player_id': self.player_id,
            'result': self.result,
            'move_sequence': self.move_sequence,
            'status': self.status,
            'board_state': self.board_state,
            'current_turn': self.current_turn,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return f'<GameSession {self.id} status={self.status}>'
