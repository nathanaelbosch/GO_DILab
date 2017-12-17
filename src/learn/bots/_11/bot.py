from src.learn.bots.ValueBot import ValueBot


class Bot(ValueBot):
    def __init__(self):
        super().__init__()

    @staticmethod
    def board_to_input(flat_board):
        return flat_board
