import os
import numpy as np

from src.learn.bots.PolicyBot import PolicyBot
from src.play.model.Game import BLACK, WHITE


class Bot_12(PolicyBot):

    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def generate_nn_input(flat_board, color):
        out = np.append(
            flat_board, [[BLACK if color == 'b' else WHITE]], axis=1)
        return out
