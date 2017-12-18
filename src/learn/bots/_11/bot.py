import os
from src.learn.bots.ValueBot import ValueBot


class Bot_11(ValueBot):
    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def generate_nn_input(flat_board, color):
        return flat_board
