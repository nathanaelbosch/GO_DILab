from typing import Tuple
from src.model.Game import InvalidMove_Error

# TODO
# use these to map location strings to coordinates in HumanConsolePlayer instead of passing string-location to game


def str2index(loc: str, board_size) -> Tuple[int, int]:
    col = _chr2ord(loc[0], board_size)
    row = _chr2ord(loc[1], board_size)
    return row, col


def _chr2ord(c: str, board_size) -> int:
    idx = ord(c) - ord('a')
    if idx < 0 or idx >= board_size:
        raise InvalidMove_Error(
            c + '=' + str(idx) +
            ' is an invalid row/column index, board size is ' +
            str(board_size))
    return idx
