import os
import sys
import numpy as np
from os.path import dirname
from abc import ABC, abstractmethod

from src import Utils
from src.play.model.Move import Move


class BaseNNBot(ABC):

    def __init__(self):
        Utils.set_keras_backend('theano')
        from keras.models import model_from_json
        model_files_dir = dirname(self.get_path_to_self())  # sys._MEIPASS, for use with pyinstaller
        model_architecture_path = os.path.join(model_files_dir, 'model_architecture.json')
        if not os.path.isfile(model_architecture_path):
            print('model architecture not found: ' + model_architecture_path)
            sys.exit(1)
        model_weights_path = os.path.join(model_files_dir, 'model_weights.h5')
        if not os.path.isfile(model_weights_path):
            print('model weights not found: ' + model_weights_path)
            sys.exit(1)
        json_file = open(model_architecture_path, 'r')
        self.model = model_from_json(json_file.read())
        json_file.close()
        self.model.load_weights(model_weights_path)

    @abstractmethod
    def get_path_to_self(self):
        pass

    # can be overwritten by extending classes
    def customize_color_values(self, flat_board):
        return flat_board

    @abstractmethod
    def _genmove(self, color, game, flat_board):
        pass

    def genmove(self, color, game) -> Move:
        flat_board = np.array(game.board).flatten()
        flat_board = self.customize_color_values(flat_board)
        return self._genmove(color, game, flat_board)
