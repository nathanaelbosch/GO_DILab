from os.path import abspath
import numpy as np

from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Move import Move


class NNBot(BaseNNBot):

    def __init__(self):
        super().__init__()

    def get_path_to_self(self):
        return abspath(__file__)

    def _genmove(self, color, game, flat_board):
        predict = self.model.predict(np.array([flat_board]))
        max_idx = np.argmax(predict)
        if max_idx is 0:
            return Move(is_pass=True)
        else:
            board = predict[0][1:]  # strip away the pass-slot at pos zero
            # set all invalid locations to 0 to avoid them being chosen
            for move in game.get_invalid_locations(color):
                flat_idx = move.to_flat_idx(game.size)
                board[flat_idx] = 0
            max_idx = np.argmax(board)
            row, col = self.deflatten_move(max_idx)
            return Move(col=col, row=row)
