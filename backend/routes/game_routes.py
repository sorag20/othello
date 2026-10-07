"""API endpoints for game operations.

Provides REST API endpoints for game creation, move submission,
game state retrieval, history listing, and replay data.
"""

import logging
import os

from flask import Blueprint, current_app, jsonify, request

from services.game_service import (
    GameNotFoundError,
    GameFinishedError,
    GameService,
    InvalidMoveError,
)
from services.history_service import HistoryService

logger = logging.getLogger(__name__)

game_bp = Blueprint('game', __name__)

# Lazy-initialized service instances
_game_service = None
_history_service = None

# Whether edax binary is available
_edax_available = None


def _check_edax_available() -> bool:
    """Check once if the edax binary exists and return the result."""
    global _edax_available
    if _edax_available is None:
        edax_path = current_app.config.get('EDAX_PATH', '')
        _edax_available = bool(edax_path) and os.path.isfile(edax_path)
        if _edax_available:
            logger.info("Edax engine found at %s", edax_path)
        else:
            logger.info("Edax engine not found at %r; edax opponent unavailable", edax_path)
    return _edax_available


def _make_edax_factory():
    """Return a callable (level) -> EdaxEngine, or None if edax is unavailable."""
    if not _check_edax_available():
        return None

    edax_path = current_app.config['EDAX_PATH']
    eval_path = current_app.config.get('EDAX_EVAL_PATH', '')
    book_path = current_app.config.get('EDAX_BOOK_PATH', '')

    def factory(level: int):
        from ai.edax_engine import EdaxEngine
        return EdaxEngine(
            edax_path=edax_path,
            eval_path=eval_path,
            book_path=book_path,
            level=level,
        )

    return factory


def _get_game_service() -> GameService:
    """Get or create the GameService instance (lazy initialization)."""
    global _game_service
    if _game_service is None:
        ai_engine = None
        model_path = current_app.config.get('AI_MODEL_PATH', 'models/weights/model.pt')
        simulations = current_app.config.get('AI_SIMULATIONS', 800)
        timeout = current_app.config.get('AI_TIMEOUT', 5.0)

        if os.path.isfile(model_path):
            try:
                from ai.engine import AIEngine
                ai_engine = AIEngine(
                    model_path=model_path,
                    simulations=simulations,
                    timeout=timeout,
                )
                logger.info("AI engine loaded successfully from %s", model_path)
            except Exception as e:
                logger.warning(
                    "Failed to load AI engine from %s: %s. "
                    "Running without AI (fallback to first legal move).",
                    model_path, e,
                )
        else:
            logger.info(
                "AI model file not found at %s. "
                "Running without AI (fallback to first legal move). "
                "Train a model via self-play to enable AI.",
                model_path,
            )

        from ai.minimax_engine import MinimaxEngine
        minimax_engine = MinimaxEngine(
            time_limit=current_app.config.get('MINIMAX_TIME_LIMIT', 3.0),
            endgame_empty_threshold=current_app.config.get('MINIMAX_ENDGAME_EMPTY_THRESHOLD', 10),
        )

        _game_service = GameService(
            ai_engine=ai_engine,
            edax_engine=_make_edax_factory(),
            minimax_engine=minimax_engine,
        )
    return _game_service


def _get_history_service() -> HistoryService:
    """Get or create the HistoryService instance (lazy initialization)."""
    global _history_service
    if _history_service is None:
        _history_service = HistoryService()
    return _history_service


def _error_response(code: str, message: str, status: int):
    """Build a standardized error response.

    Args:
        code: Error code string (e.g. INVALID_MOVE).
        message: Human-readable error message.
        status: HTTP status code.

    Returns:
        Tuple of (response, status_code).
    """
    return jsonify({"error": {"code": code, "message": message}}), status


@game_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker container monitoring.

    Returns:
        200: Service is healthy.
    """
    return jsonify({"status": "healthy"}), 200


@game_bp.route('/edax/status', methods=['GET'])
def edax_status():
    """Return whether the edax engine is available.

    Returns:
        200: {"available": bool}
    """
    return jsonify({"available": _check_edax_available()}), 200


@game_bp.route('/games', methods=['POST'])
def create_game():
    """Create a new game.

    Optional request body:
        opponent_type: 'internal' (default), 'edax', or 'minimax'.
        edax_level: integer 1-60 (default 1), used when opponent_type='edax'.

    Returns:
        201: GameResponse with initial board state.
        400: Invalid parameters.
        500: Internal error.
    """
    try:
        data = request.get_json(silent=True) or {}
        opponent_type = data.get('opponent_type', 'internal')
        edax_level = data.get('edax_level', 1)

        if opponent_type not in ('internal', 'edax', 'minimax'):
            return _error_response(
                "INVALID_PARAMETERS",
                "opponent_type must be 'internal', 'edax', or 'minimax'",
                400,
            )

        if opponent_type == 'edax':
            if not _check_edax_available():
                return _error_response(
                    "EDAX_UNAVAILABLE",
                    "Edax engine is not available on this server",
                    503,
                )
            if not isinstance(edax_level, int) or not (1 <= edax_level <= 60):
                return _error_response(
                    "INVALID_PARAMETERS",
                    "edax_level must be an integer between 1 and 60",
                    400,
                )

        service = _get_game_service()
        result = service.create_game(
            opponent_type=opponent_type,
            edax_level=edax_level if opponent_type == 'edax' else None,
        )
        return jsonify(result), 201
    except Exception as e:
        logger.exception("Unexpected error creating game")
        return _error_response(
            "INTERNAL_ERROR",
            "An internal error occurred",
            500,
        )


@game_bp.route('/games/<game_id>', methods=['GET'])
def get_game(game_id: str):
    """Get current game state.

    Args:
        game_id: UUID of the game session.

    Returns:
        200: GameResponse with current state.
        404: Game not found.
        500: Internal error.
    """
    try:
        service = _get_game_service()
        result = service.get_game_state(game_id)
        return jsonify(result), 200
    except GameNotFoundError:
        return _error_response(
            "GAME_NOT_FOUND",
            "The specified game was not found",
            404,
        )
    except Exception as e:
        logger.exception("Unexpected error getting game state")
        return _error_response(
            "INTERNAL_ERROR",
            "An internal error occurred",
            500,
        )


@game_bp.route('/games/<game_id>/moves', methods=['POST'])
def make_move(game_id: str):
    """Submit a move for the specified game.

    Request body must contain row and col as integers in range 0-7.

    Args:
        game_id: UUID of the game session.

    Returns:
        200: MoveResponse with updated state.
        400: Invalid move or parameters.
        404: Game not found.
        500: Internal error.
    """
    try:
        # Parse and validate request body
        data = request.get_json(silent=True)
        if data is None:
            return _error_response(
                "INVALID_PARAMETERS",
                "Request body must be valid JSON with row and col fields",
                400,
            )

        row = data.get("row")
        col = data.get("col")

        # Validate row and col are present and are integers
        if row is None or col is None:
            return _error_response(
                "INVALID_PARAMETERS",
                "Both row and col fields are required",
                400,
            )

        if not isinstance(row, int) or not isinstance(col, int):
            return _error_response(
                "INVALID_PARAMETERS",
                "row and col must be integers",
                400,
            )

        # Validate range 0-7
        if not (0 <= row <= 7) or not (0 <= col <= 7):
            return _error_response(
                "INVALID_PARAMETERS",
                "row and col must be between 0 and 7",
                400,
            )

        service = _get_game_service()
        result = service.apply_move(game_id, row, col)
        return jsonify(result), 200

    except GameNotFoundError:
        return _error_response(
            "GAME_NOT_FOUND",
            "The specified game was not found",
            404,
        )
    except GameFinishedError:
        return _error_response(
            "INVALID_MOVE",
            "The game is already finished",
            400,
        )
    except InvalidMoveError:
        return _error_response(
            "INVALID_MOVE",
            "The specified cell is not a legal move position",
            400,
        )
    except Exception as e:
        logger.exception("Unexpected error applying move")
        return _error_response(
            "INTERNAL_ERROR",
            "An internal error occurred",
            500,
        )


@game_bp.route('/games', methods=['GET'])
def get_history():
    """Get game history list with pagination.

    Query parameters:
        page: Page number (1-indexed, default 1).
        per_page: Items per page (default 20, max 50).

    Returns:
        200: HistoryResponse with paginated games list.
        400: Invalid pagination parameters.
        500: Internal error.
    """
    try:
        # Parse pagination parameters
        page_str = request.args.get("page", "1")
        per_page_str = request.args.get("per_page", "20")

        try:
            page = int(page_str)
            per_page = int(per_page_str)
        except (ValueError, TypeError):
            return _error_response(
                "INVALID_PARAMETERS",
                "page and per_page must be integers",
                400,
            )

        if page < 1:
            return _error_response(
                "INVALID_PARAMETERS",
                "page must be greater than or equal to 1",
                400,
            )

        if per_page < 1:
            return _error_response(
                "INVALID_PARAMETERS",
                "per_page must be greater than or equal to 1",
                400,
            )

        service = _get_history_service()
        result = service.get_history(page=page, per_page=per_page)
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Unexpected error getting history")
        return _error_response(
            "INTERNAL_ERROR",
            "An internal error occurred",
            500,
        )


@game_bp.route('/games/<game_id>/replay', methods=['GET'])
def get_replay(game_id: str):
    """Get replay data for a specific game.

    Reconstructs board states from the move sequence.

    Args:
        game_id: UUID of the game session.

    Returns:
        200: ReplayResponse with move sequence and board states.
        404: Game not found.
        500: Internal error.
    """
    try:
        service = _get_history_service()
        result = service.get_replay(game_id)
        return jsonify(result), 200
    except ValueError:
        return _error_response(
            "GAME_NOT_FOUND",
            "The specified game was not found",
            404,
        )
    except Exception as e:
        logger.exception("Unexpected error getting replay")
        return _error_response(
            "INTERNAL_ERROR",
            "An internal error occurred",
            500,
        )
