"""Game orchestration service.

Provides the GameService class responsible for coordinating game creation,
move application, AI turn handling, pass detection, and game state persistence.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from sqlalchemy.exc import SQLAlchemyError

from ai.engine import AIEngine, AITimeoutError
from game.board import Board
from game.constants import BLACK, WHITE
from game.rules import OthelloRules
from models.database import db
from models.game_session import GameSession

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


class GameService:
    """ゲームオーケストレーションサービス

    ゲームの作成、手の適用、AI手番処理、パス検出、
    ゲーム終了判定、DB永続化を統括する。
    """

    def __init__(self, ai_engine: Optional[AIEngine] = None):
        """Initialize with optional AI engine.

        Args:
            ai_engine: AIEngine instance. If None, AI moves will be skipped.
        """
        self._ai_engine = ai_engine

    def create_game(self) -> dict:
        """Create new game with player as black, AI as white.

        Creates a new GameSession with the initial board state,
        saves to DB, and returns game data including legal moves.

        Returns:
            dict: Game state response.
        """
        board = Board()
        board_state = board.get_state()
        legal_moves = board.get_legal_moves(BLACK)
        stone_count = board.count_stones()

        game_session = GameSession(
            board_state=board_state,
            current_turn='black',
            status='in_progress',
            move_sequence=[],
        )

        self._save_with_retry(game_session)

        return {
            "id": game_session.id,
            "board": board_state,
            "current_turn": "black",
            "legal_moves": legal_moves,
            "stone_count": stone_count,
            "status": "in_progress",
            "result": None,
            "move": None,
            "flipped": [],
            "passed": False,
        }

    def apply_move(self, game_id: str, row: int, col: int) -> dict:
        """Apply player's move, then trigger AI if it's AI's turn.

        Args:
            game_id: The game session ID.
            row: Row coordinate (0-7).
            col: Column coordinate (0-7).

        Returns:
            dict: Updated game state response.

        Raises:
            GameNotFoundError: If game_id doesn't exist.
            GameFinishedError: If game is already finished.
            InvalidMoveError: If move is not legal.
        """
        game_session = GameSession.query.get(game_id)
        if game_session is None:
            raise GameNotFoundError(f"Game {game_id} not found")

        if game_session.status == 'finished':
            raise GameFinishedError("Game is already finished")

        board = Board(state=game_session.board_state)
        current_color = BLACK if game_session.current_turn == 'black' else WHITE

        # Validate move is legal
        legal_moves = board.get_legal_moves(current_color)
        if (row, col) not in legal_moves:
            raise InvalidMoveError(
                f"Move ({row}, {col}) is not a legal move"
            )

        # Apply the move
        flipped = board.place_stone(row, col, current_color)

        # Update move sequence
        move_sequence = list(game_session.move_sequence or [])
        move_sequence.append({"row": row, "col": col})
        game_session.move_sequence = move_sequence

        # Update board state and stone counts
        game_session.board_state = board.get_state()
        stone_count = board.count_stones()

        passed = False

        # Check game end
        if self._check_game_end(board, current_color):
            self._finalize_game(game_session, board)
            self._save_with_retry(game_session)
            next_legal_moves = []
            return self._build_response(
                game_session, board, stone_count, next_legal_moves,
                move={"row": row, "col": col},
                flipped=flipped,
                passed=False,
            )

        # Determine next turn
        opponent_color = WHITE if current_color == BLACK else BLACK

        # Check if opponent must pass
        if not board.get_legal_moves(opponent_color):
            # Opponent has no legal moves - pass back to current player
            passed = True
            # current_turn stays the same color
        else:
            # Switch turn to opponent
            game_session.current_turn = 'white' if current_color == BLACK else 'black'

        # Calculate legal moves for next player
        next_color = BLACK if game_session.current_turn == 'black' else WHITE
        next_legal_moves = board.get_legal_moves(next_color)

        # Save after player move
        self._save_with_retry(game_session)

        # Build player move response
        response = self._build_response(
            game_session, board, stone_count, next_legal_moves,
            move={"row": row, "col": col},
            flipped=flipped,
            passed=passed,
        )

        # If it's now AI's turn and game is not over, trigger AI
        if game_session.current_turn == 'white' and game_session.status == 'in_progress':
            response = self._apply_ai_move(game_session)

        return response

    def get_game_state(self, game_id: str) -> dict:
        """Get current game state.

        Args:
            game_id: The game session ID.

        Returns:
            dict: Current game state.

        Raises:
            GameNotFoundError: If game_id doesn't exist.
        """
        game_session = GameSession.query.get(game_id)
        if game_session is None:
            raise GameNotFoundError(f"Game {game_id} not found")

        board = Board(state=game_session.board_state)
        current_color = BLACK if game_session.current_turn == 'black' else WHITE
        stone_count = board.count_stones()

        if game_session.status == 'finished':
            legal_moves = []
        else:
            legal_moves = board.get_legal_moves(current_color)

        return self._build_response(
            game_session, board, stone_count, legal_moves,
            move=None,
            flipped=[],
            passed=False,
        )

    def _apply_ai_move(self, game_session: GameSession) -> dict:
        """Apply AI's move.

        Uses the AI engine to compute the best move, applies it to the board,
        and handles game end / pass detection afterward.

        Args:
            game_session: The current game session.

        Returns:
            dict: Updated game state response.
        """
        board = Board(state=game_session.board_state)
        ai_color = WHITE  # AI is always white

        # Check if AI has legal moves
        legal_moves = board.get_legal_moves(ai_color)
        if not legal_moves:
            # AI must pass - switch back to player
            game_session.current_turn = 'black'
            next_legal_moves = board.get_legal_moves(BLACK)
            stone_count = board.count_stones()
            self._save_with_retry(game_session)
            return self._build_response(
                game_session, board, stone_count, next_legal_moves,
                move=None,
                flipped=[],
                passed=True,
            )

        # Compute AI move
        ai_move = None
        try:
            if self._ai_engine is not None:
                ai_move = self._ai_engine.compute_move(board, ai_color)
            else:
                # Fallback: pick first legal move if no AI engine
                ai_move = legal_moves[0]
        except AITimeoutError:
            logger.warning("AI computation timed out, passing AI's turn")
            # AI timeout → pass AI's turn
            game_session.current_turn = 'black'
            next_legal_moves = board.get_legal_moves(BLACK)
            stone_count = board.count_stones()
            self._save_with_retry(game_session)
            return self._build_response(
                game_session, board, stone_count, next_legal_moves,
                move=None,
                flipped=[],
                passed=True,
            )

        # Apply AI's move
        flipped = board.place_stone(ai_move[0], ai_move[1], ai_color)

        # Update move sequence
        move_sequence = list(game_session.move_sequence or [])
        move_sequence.append({"row": ai_move[0], "col": ai_move[1]})
        game_session.move_sequence = move_sequence

        # Update board state
        game_session.board_state = board.get_state()
        stone_count = board.count_stones()

        passed = False

        # Check game end after AI move
        if self._check_game_end(board, ai_color):
            self._finalize_game(game_session, board)
            self._save_with_retry(game_session)
            return self._build_response(
                game_session, board, stone_count, [],
                move={"row": ai_move[0], "col": ai_move[1]},
                flipped=flipped,
                passed=False,
            )

        # Determine next turn after AI move
        player_color = BLACK
        if not board.get_legal_moves(player_color):
            # Player must pass - AI keeps turn
            passed = True
            game_session.current_turn = 'white'
            # AI needs to move again
            self._save_with_retry(game_session)
            # Recursively apply AI move
            ai_response = self._apply_ai_move(game_session)
            # Merge pass information
            ai_response["passed"] = True
            return ai_response
        else:
            # Switch to player's turn
            game_session.current_turn = 'black'

        next_legal_moves = board.get_legal_moves(BLACK)
        self._save_with_retry(game_session)

        return self._build_response(
            game_session, board, stone_count, next_legal_moves,
            move={"row": ai_move[0], "col": ai_move[1]},
            flipped=flipped,
            passed=passed,
        )

    def _check_game_end(self, board: Board, last_move_color: int) -> bool:
        """Check if game is over.

        The game ends when the board is full or neither player has legal moves.

        Args:
            board: Current board state.
            last_move_color: Color of the player who just moved.

        Returns:
            True if the game is over.
        """
        return OthelloRules.is_game_over(board, last_move_color)

    def _finalize_game(self, game_session: GameSession, board: Board) -> None:
        """Finalize a game by setting result and status.

        Args:
            game_session: The game session to finalize.
            board: The final board state.
        """
        game_session.status = 'finished'
        game_session.ended_at = datetime.utcnow()

        winner = OthelloRules.determine_winner(board)
        if winner == BLACK:
            game_session.result = 'black_win'
        elif winner == WHITE:
            game_session.result = 'white_win'
        else:
            game_session.result = 'draw'

    def _save_with_retry(self, game_session: GameSession) -> None:
        """Save game session to DB with retry logic.

        Attempts up to MAX_RETRIES times with RETRY_DELAY between retries.

        Args:
            game_session: The game session to save.

        Raises:
            SQLAlchemyError: If all retry attempts fail.
        """
        for attempt in range(MAX_RETRIES):
            try:
                db.session.add(game_session)
                db.session.commit()
                return
            except SQLAlchemyError as e:
                db.session.rollback()
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"DB save failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(
                        f"DB save failed after {MAX_RETRIES} attempts: {e}"
                    )
                    raise

    def _build_response(
        self,
        game_session: GameSession,
        board: Board,
        stone_count: dict,
        legal_moves: list,
        move: Optional[dict],
        flipped: list,
        passed: bool,
    ) -> dict:
        """Build standardized response dictionary.

        Args:
            game_session: Current game session.
            board: Current board state.
            stone_count: Stone counts dict.
            legal_moves: List of legal move tuples.
            move: Last move made (or None).
            flipped: List of flipped stone coordinates.
            passed: Whether a pass occurred.

        Returns:
            dict: Standardized response.
        """
        # Convert flipped tuples to list format for JSON serialization
        flipped_list = [(r, c) for r, c in flipped] if flipped else []

        return {
            "id": game_session.id,
            "board": board.get_state(),
            "current_turn": game_session.current_turn,
            "legal_moves": legal_moves,
            "stone_count": stone_count,
            "status": game_session.status,
            "result": game_session.result,
            "move": move,
            "flipped": flipped_list,
            "passed": passed,
        }


class GameNotFoundError(Exception):
    """Raised when a game session is not found."""
    pass


class GameFinishedError(Exception):
    """Raised when attempting to modify a finished game."""
    pass


class InvalidMoveError(Exception):
    """Raised when an invalid move is attempted."""
    pass
