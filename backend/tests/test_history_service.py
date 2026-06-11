"""Tests for the HistoryService class."""

import uuid
from datetime import datetime, timedelta

import pytest

from app import create_app
from config import TestingConfig
from models.database import db
from models.game_session import GameSession
from services.history_service import HistoryService


@pytest.fixture
def app():
    """Create a test Flask application."""
    app = create_app(TestingConfig)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def service(app):
    """Create a HistoryService instance within app context."""
    with app.app_context():
        yield HistoryService()


def _create_finished_game(result="black_win", move_sequence=None, ended_offset_hours=0):
    """Helper to create a finished game session."""
    if move_sequence is None:
        move_sequence = [{"row": 2, "col": 3}, {"row": 2, "col": 2}]
    game = GameSession(
        id=str(uuid.uuid4()),
        status="finished",
        result=result,
        move_sequence=move_sequence,
        board_state=[[0] * 8 for _ in range(8)],
        current_turn="black",
        ended_at=datetime.utcnow() - timedelta(hours=ended_offset_hours),
    )
    db.session.add(game)
    db.session.commit()
    return game


class TestGetHistory:
    """Tests for HistoryService.get_history."""

    def test_empty_history(self, app, service):
        """Return empty list with total=0 when no games exist (Req 5.6)."""
        with app.app_context():
            result = service.get_history()
            assert result["games"] == []
            assert result["total"] == 0
            assert result["page"] == 1
            assert result["per_page"] == 20

    def test_returns_finished_games_only(self, app, service):
        """Only finished games are returned."""
        with app.app_context():
            _create_finished_game(result="black_win")
            # Create an in-progress game - should not appear
            in_progress = GameSession(
                id=str(uuid.uuid4()),
                status="in_progress",
                move_sequence=[],
                board_state=[[0] * 8 for _ in range(8)],
                current_turn="black",
            )
            db.session.add(in_progress)
            db.session.commit()

            result = service.get_history()
            assert result["total"] == 1
            assert len(result["games"]) == 1

    def test_newest_first_ordering(self, app, service):
        """Games are ordered by ended_at descending (Req 5.2)."""
        with app.app_context():
            older = _create_finished_game(result="white_win", ended_offset_hours=2)
            newer = _create_finished_game(result="black_win", ended_offset_hours=0)

            result = service.get_history()
            assert result["games"][0]["id"] == newer.id
            assert result["games"][1]["id"] == older.id

    def test_pagination(self, app, service):
        """Pagination works correctly with page and per_page."""
        with app.app_context():
            for i in range(5):
                _create_finished_game(ended_offset_hours=i)

            result = service.get_history(page=1, per_page=2)
            assert len(result["games"]) == 2
            assert result["total"] == 5
            assert result["page"] == 1
            assert result["per_page"] == 2

            result2 = service.get_history(page=3, per_page=2)
            assert len(result2["games"]) == 1

    def test_per_page_capped_at_50(self, app, service):
        """per_page is capped at MAX_PER_PAGE (50)."""
        with app.app_context():
            result = service.get_history(per_page=100)
            assert result["per_page"] == 50

    def test_game_fields(self, app, service):
        """Each game entry has correct fields."""
        with app.app_context():
            moves = [{"row": 2, "col": 3}, {"row": 4, "col": 5}, {"row": 1, "col": 0}]
            game = _create_finished_game(result="draw", move_sequence=moves)

            result = service.get_history()
            entry = result["games"][0]
            assert entry["id"] == game.id
            assert entry["result"] == "draw"
            assert entry["ended_at"] is not None
            assert entry["move_count"] == 3


class TestGetReplay:
    """Tests for HistoryService.get_replay."""

    def test_game_not_found_raises_valueerror(self, app, service):
        """Raises ValueError when game_id doesn't exist."""
        with app.app_context():
            with pytest.raises(ValueError, match="Game not found"):
                service.get_replay("non-existent-id")

    def test_replay_with_valid_game(self, app, service):
        """Returns correct replay data for a valid game."""
        with app.app_context():
            # Moves that are legal from initial position
            # Initial: black at (3,4),(4,3), white at (3,3),(4,4)
            # Black's legal moves: (2,3), (3,2), (4,5), (5,4)
            moves = [{"row": 2, "col": 3}]  # Black plays (2,3)
            game = _create_finished_game(result="black_win", move_sequence=moves)

            result = service.get_replay(game.id)
            assert result["id"] == game.id
            assert result["result"] == "black_win"
            assert result["move_sequence"] == moves
            assert result["total_moves"] == 1
            # Initial board + 1 move = 2 board states
            assert len(result["board_states"]) == 2

    def test_replay_board_reconstruction(self, app, service):
        """Board states are correctly reconstructed (Req 5.4)."""
        with app.app_context():
            # Black plays (2,3) - flips (3,3) from white to black
            moves = [{"row": 2, "col": 3}]
            game = _create_finished_game(result="black_win", move_sequence=moves)

            result = service.get_replay(game.id)
            initial_state = result["board_states"][0]
            after_move_state = result["board_states"][1]

            # Initial state check
            assert initial_state[3][3] == 2  # white
            assert initial_state[3][4] == 1  # black
            assert initial_state[4][3] == 1  # black
            assert initial_state[4][4] == 2  # white

            # After black plays (2,3): (3,3) should be flipped to black
            assert after_move_state[2][3] == 1  # placed black
            assert after_move_state[3][3] == 1  # flipped to black
            assert after_move_state[3][4] == 1  # unchanged black
            assert after_move_state[4][3] == 1  # unchanged black
            assert after_move_state[4][4] == 2  # unchanged white

    def test_replay_empty_move_sequence(self, app, service):
        """Replay with no moves returns only initial board state."""
        with app.app_context():
            game = _create_finished_game(result="draw", move_sequence=[])

            result = service.get_replay(game.id)
            assert result["total_moves"] == 0
            assert len(result["board_states"]) == 1

    def test_replay_multiple_moves(self, app, service):
        """Replay with multiple moves returns correct number of states."""
        with app.app_context():
            # Black plays (2,3), White plays (2,2)
            moves = [{"row": 2, "col": 3}, {"row": 2, "col": 2}]
            game = _create_finished_game(result="black_win", move_sequence=moves)

            result = service.get_replay(game.id)
            assert result["total_moves"] == 2
            # Initial + 2 moves = 3 states
            assert len(result["board_states"]) == 3
