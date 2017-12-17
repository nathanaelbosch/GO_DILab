from src.learn.bots.ValueBot import ValueBot
import src.learn.bots.utils as utils


class Bot(ValueBot):
    def __init__(self):
        super().__init__()

    @staticmethod
    def board_to_input(flat_board):
        X = utils.encode_board(flat_board)
        return X
