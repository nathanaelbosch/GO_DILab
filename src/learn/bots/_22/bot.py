import os
import numpy as np
from src.learn.bots.PolicyBot import PolicyBot
import src.learn.bots.utils as utils


class Bot_22(PolicyBot):

    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def board_to_input(flat_board):
        X = utils.encode_board(flat_board)
        return X
