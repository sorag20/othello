"""Tests for the GameService class."""

import pytest
from unittest.mock import MagicMock, patch

from game.board import Board
from game.constants import BLACK, WHITE


@pytest.fixture
def app():
    """Create a test Flask app with in-memory SQLite."""
    from app import create_app
    from config import TestingConfig
    app = create_app(TestingConfig)
    yield app


@pytest.fixture
def app_context(app):
    """Provide app context for DB operations."""
    with app.app_context():
        yield


@pytest.fixture
def game_service(app_context):
    """Create a GameService instance without AI engine."""
    from services.game_service import GameService
    return GameService(ai_engine=None)


class TestCreateGame:
    """Tests for GameService.create_game()"""

    def test_create_game_returns_initial_state(self, game_service):
        """New game should have initial board, black turn, in_progress status."""
        result = game_service.create_game()

        assert result["current_turn"] == "black"
        assert result["status"] == "in_progress"
        assert result["result"] is None
        assert result["passed"] is False
        assert result["id"] is not None

    def test_create_game_initial_board(self, game_service):
        """Initial board should have 2 black and 2 white stones in center."""
        result = game_service.create_game()
        board = result["board"]

        # Standard Othello initial position
        assert board[3][3] == WHITE
        assert board[3][4] == BLACK
        assert board[4][3] == BLACK
        assert board[4][4] == WHITE

    def test_create_game_stone_count(self, game_service):
        """Initial stone count should be 2 black, 2 white."""
        result = game_service.create_game()

        assert result["stone_count"] == {"black": 2, "white": 2}

    def test_create_game_has_legal_moves(self, game_service):
        """Initial position should have legal moves for black."""
        result = game_service.create_game()

        assert len(result["legal_moves"]) > 0
        # Standard Othello has 4 legal moves for black at start
        assert len(result["legal_moves"]) == 4


class TestApplyMove:
    """Tests for GameService.apply_move()"""

    def test_apply_valid_move(self, game_service):
        """A valid move should update the board and flip stones."""
        game = game_service.create_game()
        game_id = game["id"]

        # Pick first legal move
        legal = game["legal_moves"]
        row, col = legal[0]

        result = game_service.apply_move(game_id, row, col)

        assert result["move"] is not None
        assert result["move"]["row"] == row
        assert result["move"]["col"] == col
        assert len(result["flipped"]) > 0

    def test_apply_move_game_not_found(self, game_service):
        """Applying move to non-existent game should raise GameNotFoundError."""
        from services.game_service import GameNotFoundError

        with pytest.raises(GameNotFoundError):
            game_service.apply_move("nonexistent-id", 2, 3)

    def test_apply_invalid_move(self, game_service):
        """Applying an illegal move should raise InvalidMoveError."""
        from services.game_service import InvalidMoveError

        game = game_service.create_game()
        game_id = game["id"]

        # (0, 0) is not a legal move at start
        with pytest.raises(InvalidMoveError):
            game_service.apply_move(game_id, 0, 0)

    def test_apply_move_to_finished_game(self, game_service):
        """Applying move to finished game should raise GameFinishedError."""
        from services.game_service import GameFinishedError
        from models.game_session import GameSession

        game = game_service.create_game()
        game_id = game["id"]

        # Manually set game to finished
        session = GameSession.query.get(game_id)
        session.status = 'finished'
        from models.database import db
        db.session.commit()

        with pytest.raises(GameFinishedError):
            game_service.apply_move(game_id, 2, 3)

    def test_apply_move_updates_stone_count(self, game_service):
        """After a move, stone counts should be updated."""
        game = game_service.create_game()
        game_id = game["id"]

        legal = game["legal_moves"]
        row, col = legal[0]

        result = game_service.apply_move(game_id, row, col)

        total_stones = result["stone_count"]["black"] + result["stone_count"]["white"]
        # At least 5 stones after first move (2+2 initial + 1 placed)
        assert total_stones >= 5


class TestGetGameState:
    """Tests for GameService.get_game_state()"""

    def test_get_game_state(self, game_service):
        """Should return current game state."""
        game = game_service.create_game()
        game_id = game["id"]

        state = game_service.get_game_state(game_id)

        assert state["id"] == game_id
        assert state["current_turn"] == "black"
        assert state["status"] == "in_progress"
        assert len(state["legal_moves"]) == 4

    def test_get_game_state_not_found(self, game_service):
        """Should raise GameNotFoundError for non-existent game."""
        from services.game_service import GameNotFoundError

        with pytest.raises(GameNotFoundError):
            game_service.get_game_state("nonexistent-id")


class TestAIMove:
    """Tests for AI move handling."""

    def test_ai_fallback_without_engine(self, game_service):
        """Without AI engine, should use first legal move as fallback."""
        game = game_service.create_game()
        game_id = game["id"]

        legal = game["legal_moves"]
        row, col = legal[0]

        # Apply player move - AI should auto-play using fallback
        result = game_service.apply_move(game_id, row, col)

        # After player's move, AI should have played (since no engine, uses first legal move)
        # Turn should be back to black (player)
        assert result["current_turn"] == "black"

    def test_ai_timeout_passes_turn(self, app_context):
        """When AI times out, its turn should be passed."""
        from services.game_service import GameService
        from ai.engine import AITimeoutError

        mock_engine = MagicMock()
        mock_engine.compute_move.side_effect = AITimeoutError("timeout")

        service = GameService(ai_engine=mock_engine)
        game = service.create_game()
        game_id = game["id"]

        legal = game["legal_moves"]
        row, col = legal[0]

        result = service.apply_move(game_id, row, col)

        # AI timed out, turn passes back to player
        assert result["current_turn"] == "black"
        assert result["passed"] is True


class TestDBRetry:
    """Tests for DB save retry logic."""

    def test_save_retries_on_failure(self, app_context):
        """Should retry DB save up to 3 times."""
        from services.game_service import GameService
        from sqlalchemy.exc import SQLAlchemyError

        service = GameService(ai_engine=None)

        with patch('services.game_service.time.sleep'):
            with patch('services.game_service.db.session') as mock_session:
                # First two commits fail, third succeeds
                mock_session.commit.side_effect = [
                    SQLAlchemyError("fail1"),
                    SQLAlchemyError("fail2"),
                    None,
                ]
                mock_session.add = MagicMock()
                mock_session.rollback = MagicMock()

                from models.game_session import GameSession
                game_session = GameSession(
                    board_state=Board().get_state(),
                    current_turn='black',
                    status='in_progress',
                    move_sequence=[],
                )

                # Should not raise - succeeds on 3rd try
                service._save_with_retry(game_session)
                assert mock_session.commit.call_count == 3

    def test_save_raises_after_max_retries(self, app_context):
        """Should raise after MAX_RETRIES failures."""
        from services.game_service import GameService
        from sqlalchemy.exc import SQLAlchemyError

        service = GameService(ai_engine=None)

        with patch('services.game_service.time.sleep'):
            with patch('services.game_service.db.session') as mock_session:
                mock_session.commit.side_effect = SQLAlchemyError("persistent failure")
                mock_session.add = MagicMock()
                mock_session.rollback = MagicMock()

                from models.game_session import GameSession
                game_session = GameSession(
                    board_state=Board().get_state(),
                    current_turn='black',
                    status='in_progress',
                    move_sequence=[],
                )

                with pytest.raises(SQLAlchemyError):
                    service._save_with_retry(game_session)

                assert mock_session.commit.call_count == 3
