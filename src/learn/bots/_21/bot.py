import os
from src.learn.bots.ValueBot import ValueBot
import src.learn.bots.utils as utils


class Bot(ValueBot):
    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def board_to_input(flat_board):
        X = utils.encode_board(flat_board)
        return X
