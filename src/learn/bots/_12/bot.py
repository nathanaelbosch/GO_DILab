import os
import numpy as np

from src.learn.bots.PolicyBot import PolicyBot


class Bot_12(PolicyBot):

    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def generate_nn_input(flat_board, color):
        out = np.append(
            flat_board, [[color]], axis=1)
        return out

    def __str__(self):
        return 'PolicyBot1'
