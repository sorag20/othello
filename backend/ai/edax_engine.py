"""Edax reversi engine integration.

Communicates with the edax-reversi binary via the NBoard protocol over stdin/stdout.
Edax is started fresh for each move request to avoid managing persistent process state.
"""

import logging
import re
import subprocess
from typing import Optional

from game.board import Board
from game.constants import BLACK, WHITE

logger = logging.getLogger(__name__)

# Edax NBoard protocol response pattern: "=== <move> <score> <time>"
_MOVE_RE = re.compile(r'^===\s+([a-h][1-8]|pa)\s', re.IGNORECASE)


def _to_edax_coord(row: int, col: int) -> str:
    """Convert (row, col) to edax coordinate string (e.g., (4,5) -> 'f5')."""
    return chr(ord('a') + col) + str(row + 1)


def _from_edax_coord(coord: str) -> tuple[int, int]:
    """Convert edax coordinate string to (row, col) (e.g., 'f5' -> (4,5))."""
    col = ord(coord[0].lower()) - ord('a')
    row = int(coord[1]) - 1
    return row, col


class EdaxEngine:
    """Wrapper around the edax-reversi binary.

    Each call to compute_move starts a fresh edax subprocess, replays the
    full move history to restore game state, asks edax to move, then exits.
    """

    def __init__(self, edax_path: str, eval_path: str, book_path: str, level: int = 1):
        self._edax_path = edax_path
        self._eval_path = eval_path
        self._book_path = book_path
        self._level = max(1, min(60, level))

    def compute_move(
        self,
        board: Board,
        ai_color: int,
        move_sequence: list[dict],
    ) -> Optional[tuple[int, int]]:
        """Ask edax for the best move given the current game state.

        Replays move_sequence from the initial board position to reconstruct
        the game state for edax, then requests its move.

        Args:
            board: Current board (used only to validate legal moves).
            ai_color: Color edax is playing (WHITE = 2).
            move_sequence: Full game move history as [{"row": r, "col": c}, ...].

        Returns:
            (row, col) tuple of edax's chosen move, or None on failure.
        """
        legal_moves = board.get_legal_moves(ai_color)
        if not legal_moves:
            return None

        commands = self._build_commands(move_sequence)
        try:
            response = self._run_edax(commands)
        except Exception as e:
            logger.error("Edax subprocess error: %s", e)
            return None

        move_coord = self._parse_move(response)
        if move_coord is None:
            logger.error("Edax returned unparseable output: %r", response)
            return None

        row, col = _from_edax_coord(move_coord)
        if (row, col) not in legal_moves:
            logger.error("Edax returned illegal move %s; legal=%s", move_coord, legal_moves)
            return None

        return row, col

    def _build_commands(self, move_sequence: list[dict]) -> str:
        """Build the NBoard protocol command string to replay game state."""
        lines = ['nboard 1']

        # Replay moves using a scratch Board to detect passes
        replay_board = Board()
        current_color = BLACK

        for entry in move_sequence:
            # If current player has no moves, insert a pass before the next move
            if not replay_board.get_legal_moves(current_color):
                lines.append('move pa')
                current_color = WHITE if current_color == BLACK else BLACK

            row, col = entry['row'], entry['col']
            lines.append(f'move {_to_edax_coord(row, col)}')
            replay_board.place_stone(row, col, current_color)
            current_color = WHITE if current_color == BLACK else BLACK

        lines.append('go')
        lines.append('quit')
        return '\n'.join(lines) + '\n'

    def _run_edax(self, commands: str) -> str:
        """Run edax subprocess and return stdout output.

        Writes all commands up to 'go', then reads stdout line by line until
        the '===' move response appears (or timeout is reached), then sends
        'quit'. This avoids the fixed-sleep race condition where edax is still
        thinking when we terminate it.
        """
        import threading

        proc = subprocess.Popen(
            [
                self._edax_path,
                '-nboard',
                '-eval-file', self._eval_path,
                '-book-file', self._book_path,
                '-l', str(self._level),
                '-q',
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Write all commands except quit
        go_commands = '\n'.join(
            line for line in commands.splitlines() if line.strip() != 'quit'
        ) + '\n'
        proc.stdin.write(go_commands)
        proc.stdin.flush()

        # Read stdout in a background thread until we see the move line.
        collected: list[str] = []
        move_found = threading.Event()

        def _reader():
            for line in proc.stdout:
                collected.append(line)
                if _MOVE_RE.match(line.strip()):
                    move_found.set()
                    break  # stop reading; quit will be sent below

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()

        # Wait up to 60 s for edax to respond (level 21 endgame can take tens of seconds)
        move_found.wait(timeout=60.0)

        proc.stdin.write('quit\n')
        proc.stdin.flush()

        reader.join(timeout=5.0)
        try:
            proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

        return ''.join(collected)

    def _parse_move(self, output: str) -> Optional[str]:
        """Parse the edax move from NBoard output.

        Looks for a line matching '=== <coord> <score> <time>'.
        """
        for line in output.splitlines():
            m = _MOVE_RE.match(line.strip())
            if m:
                return m.group(1).lower()
        return None
