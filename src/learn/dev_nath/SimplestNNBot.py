from os.path import abspath
import numpy as np

from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Board import EMPTY, BLACK, WHITE
from src.play.model.Move import Move


class SimplestNNBot(BaseNNBot):

    def __init__(self):
        super().__init__()

    def get_path_to_self(self):
        return abspath(__file__)

    @staticmethod
    def board_to_input(flat_board):
        X = np.concatenate(
            ((flat_board==WHITE)*3 - 1,
             (flat_board==BLACK)*3 - 1,
             (flat_board==EMPTY)*3 - 1),
            axis=1)
        X = X / np.sqrt(2)
        return X

    def _genmove(self, color, game, flat_board):
        flat_board = flat_board.reshape(1, len(flat_board))

        X = self.board_to_input(flat_board)
        predict = self.model.predict(X)[0]

        # Set invalid moves to 0
        for move in game.get_invalid_locations(color):
            flat_idx = move.to_flat_idx()
            predict[flat_idx] = 0

        max_idx = np.argmax(predict)
        if max_idx == 81 or predict[max_idx] == 0:
            return Move(is_pass=True)
        else:
            return Move.from_flat_idx(max_idx)
