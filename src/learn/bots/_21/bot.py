import os
from src.learn.bots.ValueBot import ValueBot
import src.learn.bots.utils as utils
from src.play.model.Game import WHITE, BLACK


class Bot_21(ValueBot):
    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def generate_nn_input(flat_board, color):
        color = WHITE if color == 'w' else BLACK
        X = utils.encode_board(flat_board, color)
        return X
