"""History retrieval service.

Provides the HistoryService class for querying game history,
paginated listing, and replay data reconstruction.
"""

from typing import List, Optional

from game.board import Board
from game.constants import BLACK, WHITE
from models.database import db
from models.game_session import GameSession


class HistoryService:
    """Service for retrieving game history and replay data."""

    MAX_PER_PAGE = 50

    def get_history(self, page: int = 1, per_page: int = 20) -> dict:
        """Get game history list, newest first, with pagination.

        Args:
            page: Page number (1-indexed). Defaults to 1.
            per_page: Number of items per page. Defaults to 20, capped at 50.

        Returns:
            Dictionary containing games list, total count, page, and per_page.
        """
        # Validate and cap parameters
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 1
        if per_page > self.MAX_PER_PAGE:
            per_page = self.MAX_PER_PAGE

        # Query finished games ordered by ended_at descending
        query = GameSession.query.filter_by(status='finished').order_by(
            GameSession.ended_at.desc()
        )

        total = query.count()

        # Paginate with offset/limit
        offset = (page - 1) * per_page
        games = query.offset(offset).limit(per_page).all()

        return {
            "games": [
                {
                    "id": game.id,
                    "result": game.result,
                    "ended_at": game.ended_at.isoformat() if game.ended_at else None,
                    "move_count": len(game.move_sequence) if game.move_sequence else 0,
                }
                for game in games
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
        }

    def get_replay(self, game_id: str) -> dict:
        """Get replay data for a specific game.

        Retrieves the game session and reconstructs board states by replaying
        the move sequence from the initial position.

        Args:
            game_id: The UUID of the game session.

        Returns:
            Dictionary containing game id, result, move_sequence,
            board_states, and total_moves.

        Raises:
            ValueError: If the game_id does not correspond to an existing game.
        """
        game = db.session.get(GameSession, game_id)
        if game is None:
            raise ValueError(f"Game not found: {game_id}")

        move_sequence = game.move_sequence if game.move_sequence else []
        board_states = self._reconstruct_board_states(move_sequence)

        return {
            "id": game.id,
            "result": game.result,
            "move_sequence": move_sequence,
            "board_states": board_states,
            "total_moves": len(move_sequence),
        }

    def _reconstruct_board_states(self, move_sequence: List[dict]) -> List[List[List[int]]]:
        """Reconstruct board states from initial position by replaying moves.

        Starts from the initial board and applies each move in sequence,
        alternating between BLACK and WHITE, handling passes when a player
        has no legal moves.

        Args:
            move_sequence: List of move dictionaries with 'row' and 'col' keys.

        Returns:
            List of board states (each is an 8x8 2D list), starting with
            the initial board state followed by states after each move.
        """
        board = Board()
        board_states: List[List[List[int]]] = [board.get_state()]

        current_color = BLACK

        for move in move_sequence:
            # Check if current player has legal moves
            legal_moves = board.get_legal_moves(current_color)
            if not legal_moves:
                # Current player must pass, switch to opponent
                current_color = WHITE if current_color == BLACK else BLACK

            row = move["row"]
            col = move["col"]
            board.place_stone(row, col, current_color)
            board_states.append(board.get_state())

            # Alternate color for next move
            current_color = WHITE if current_color == BLACK else BLACK

        return board_states
