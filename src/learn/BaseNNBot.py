import os
import sys
import numpy as np
from os.path import dirname
from abc import ABC, abstractmethod

from src import Utils
from src.play.model.Move import Move


class BaseNNBot(ABC):

    def __init__(self):
        if Utils.in_pyinstaller_mode():
            # path to root dir, expects data to be added like this when calling pyinstaller:
            # --add-data ".\src\learn\[...]\model.json;."
            # the dot means: place it in root dir
            model_files_dir = sys._MEIPASS
        else:  # if not in pyinstaller mode, use theano, because it works with pygame, tensorflow doesn't
            Utils.set_keras_backend('theano')
            model_files_dir = dirname(self.get_path_to_self())
        from keras.models import model_from_json
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
