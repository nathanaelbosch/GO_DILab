from os.path import abspath
import numpy as np

from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Board import EMPTY, BLACK, WHITE
from src.play.model.Move import Move

EMPTY_val = 0.45
BLACK_val = -1.35
WHITE_val = 1.05


class MovePredictionBot(BaseNNBot):

    def __init__(self):
        super().__init__()

    def get_path_to_self(self):
        return abspath(__file__)

    @staticmethod
    def replace_entry(entry):
        if entry is EMPTY:
            return EMPTY_val
        if entry is BLACK:
            return BLACK_val
        if entry is WHITE:
            return WHITE_val

    def _genmove(self, color, game, flat_board):
        flat_board = flat_board.reshape(1, len(flat_board))
        input_board = flat_board.tolist()
        input_board = [self.replace_entry(entry) for row in input_board for entry in row]
        if color == BLACK:
            input_board.append(1)
        else:
            input_board.append(-1)
        pred = self.model.predict(np.array([input_board]).reshape(1,-1))
        max_idx = np.argmax(pred)
        if max_idx is 81:
            return Move(is_pass=True)
        else:
            board = pred[0][0:81] # strip away the pass class at pos 82
            # set all invalid locations to -1 to avoid them being chosen
            for move in game.get_invalid_locations(color):
                flat_idx = move.to_flat_idx(game.size)
                board[flat_idx] = -1
            max_idx = np.argmax(board)

            # if all moves are invalid, play pass
            if board[max_idx] == -1:
                return Move(is_pass=True)

            return Move.from_flat_idx(max_idx)
