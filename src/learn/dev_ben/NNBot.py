import os
from os.path import dirname, abspath
import math
import numpy as np

from src import Utils
from src.play.model.Move import Move
from src.learn.dev_ben.generate_training_data import EMPTY_val, BLACK_val, WHITE_val
from src.play.model.Board import EMPTY, BLACK, WHITE


class NNBot:

    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
        Utils.set_keras_backend('theano')
        import keras
        model_path = os.path.join(project_dir, 'src/learn/dev_ben/model.h5')
        self.model = keras.models.load_model(model_path)

    @staticmethod
    def replace_entry(entry):
        if entry is EMPTY:
            return EMPTY_val
        if entry is BLACK:
            return BLACK_val
        if entry is WHITE:
            return WHITE_val

    def serialize_matrix(self, m):
        ls = m.tolist()
        ls = [self.replace_entry(entry) for row in ls for entry in row]
        return ls

    def genmove(self, color, game) -> Move:
        nn_input_board = self.serialize_matrix(game.board)
        predict = self.model.predict(np.array([nn_input_board]))
        max_idx = np.argmax(predict)
        if max_idx is 0:
            return Move(is_pass=True)
        else:
            max_idx -= 1  # to compensate for the added pass at pos zero
            row = int(math.floor(max_idx / 9))
            col = int(max_idx % 9)
            return Move(col=col, row=row)
