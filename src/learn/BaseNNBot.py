import os
import math
import numpy as np
from os.path import dirname
from abc import ABC, abstractmethod

from src import Utils
from src.play.model.Move import Move


class BaseNNBot(ABC):

    def __init__(self):
        Utils.set_keras_backend('theano')
        from keras.models import model_from_json
        model_architecture_path = os.path.join(dirname(self.get_path_to_self()), 'model_architecture.json')
        if not os.path.isfile(model_architecture_path):
            print('model architecture not found: ' + model_architecture_path)
            exit(1)
        model_weights_path = os.path.join(dirname(self.get_path_to_self()), 'model_weights.h5')
        if not os.path.isfile(model_weights_path):
            print('model weights not found: ' + model_weights_path)
            exit(1)
        json_file = open(model_architecture_path, 'r')
        self.model = model_from_json(json_file.read())
        json_file.close()
        self.model.load_weights(model_weights_path)

    @abstractmethod
    def get_path_to_self(self):
        pass

    @staticmethod
    def flatten_matrix(matrix):
        return np.array([val for row in matrix.tolist() for val in row])

    # can be overwritten by extending classes
    def customize_color_values(self, flat_board):
        return flat_board

    @abstractmethod
    def _genmove(self, color, game, flat_board):
        pass

    @staticmethod
    def deflatten_move(flat_move):
        row = int(math.floor(flat_move / 9))
        col = int(flat_move % 9)
        return row, col

    def genmove(self, color, game) -> Move:
        flat_board = self.customize_color_values(self.flatten_matrix(game.board))
        return self._genmove(color, game, flat_board)
