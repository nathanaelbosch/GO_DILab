import os
from os.path import dirname, abspath
import numpy as np
import math

from src import Utils
from src.play.model.Board import EMPTY, BLACK, WHITE
from src.play.model.Move import Move

BLACK_VAL = -1.35
WHITE_VAL = 1.25
EMPTY_VAL = 0.25


class MovePredictionBot:
    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
        Utils.set_keras_backend('theano')
        import keras
        model_path = os.path.join(project_dir, 'src/learn/dev_yu/model.h5')
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def replace_entry(entry):
        if entry is EMPTY:
            return EMPTY_VAL
        if entry is BLACK:
            return BLACK_VAL
        if entry is WHITE:
            return WHITE_VAL

    def flatten_matrix(self, m, color):
        ls = m.tolist()
        if color == BLACK:
            ls = [self.replace_entry(entry) for row in ls for entry in row]
            ls.append(BLACK_VAL)
        else:
            ls = [self.replace_entry(entry) for row in ls for entry in row]
            ls.append(WHITE_VAL)
        return ls

    def genmove(self, color, game) -> Move:
        input_board = self.flatten_matrix(game.board, color)
        pred = self.model.predict(np.array([input_board]).reshape(1,-1))
        max_idx = np.argmax(pred)
        if max_idx is 81:
            return Move(is_pass=True)
        else:
            board = pred[0][0:81]
            # set all invalid locations to -1 to avoid them being chosen
            # if all moves are invalid, play pass
            for move in game.get_invalid_locations(color):
                flat_idx = move.to_flat_idx(game.size)
                board[flat_idx] = -1
            max_idx = np.argmax(board)
            row = int(math.floor(max_idx / game.size))
            col = int(max_idx % game.size)
            return Move(col=col, row=row)
