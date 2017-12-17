import os
from src.learn.bots.PolicyBot import PolicyBot


class Bot(PolicyBot):

    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def board_to_input(flat_board):
        return flat_board
