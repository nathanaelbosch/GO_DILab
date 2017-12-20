import os
import numpy as np
from src.learn.bots.PolicyBot import PolicyBot
import src.learn.bots.utils as utils
from src.play.model.Game import WHITE, BLACK


class Bot_32(PolicyBot):
    def get_path_to_self(self):
        return os.path.abspath(__file__)

    @staticmethod
    def generate_nn_input(flat_board, color):
        color = WHITE if color == 'w' else BLACK
        encoded_boards = utils.encode_board(flat_board, color)
        player_liberties = utils.get_liberties(flat_board, color)
        opponent_liberties = utils.get_liberties(flat_board, -color)
        # print(encoded_boards.shape)
        # print(player_liberties.shape)
        # print(opponent_liberties.shape)
        X = np.concatenate(
            (encoded_boards, player_liberties, opponent_liberties), axis=1)
        return X
