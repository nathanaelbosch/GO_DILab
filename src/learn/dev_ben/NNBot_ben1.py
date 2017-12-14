from os.path import abspath
import numpy as np

from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Board import EMPTY, BLACK, WHITE
from src.play.model.Move import Move

EMPTY_val = 0.45
BLACK_val = -1.35
WHITE_val = 1.05


class NNBot_ben1(BaseNNBot):

    def __init__(self):
        super().__init__()

    def get_path_to_self(self):
        return abspath(__file__)

    def customize_color_values(self, flat_board):
        flat_board[flat_board == BLACK] = BLACK_val
        flat_board[flat_board == EMPTY] = EMPTY_val
        flat_board[flat_board == WHITE] = WHITE_val
        return flat_board

    def _genmove(self, color, game, flat_board):
        flat_board = flat_board.reshape(1, len(flat_board))
        predict = self.model.predict(flat_board)[0]
        max_idx = np.argmax(predict)
        if max_idx == 82:
            return Move(is_pass=True)
        else:
            board = predict[:-1]  # strip away the pass-slot at pos 82
            # set all invalid locations to 0 to avoid them being chosen
            for move in game.get_invalid_locations(color):
                flat_idx = move.to_flat_idx(game.size)
                board[flat_idx] = 0
            max_idx = np.argmax(board)

            # If this move is invalid pass!
            if board[max_idx] == 0:
                return Move(is_pass=True)

            return Move.from_flat_idx(max_idx)
