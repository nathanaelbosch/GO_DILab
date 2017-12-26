"""Basically what I did in WinPredictionBot

Value-Network with 3*81-vector as input containing the encoded board
"""
import os
import numpy as np
from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, data):
        boards = data[data.columns[3:-2]].as_matrix()

        y = utils.value_output(data['result'], data['color'])

        boards, _other = self.get_symmetries(
            boards, other_data=[y, data['color']])
        y, colors = _other
        colors = colors.reshape(len(colors), 1)

        encoded_boards = utils.encode_board(boards, colors)
        player_liberties = utils.get_liberties_vectorized(
            boards, colors)
        opponent_liberties = utils.get_liberties_vectorized(
            boards, -colors)
        X = np.concatenate(
            (encoded_boards, player_liberties, opponent_liberties), axis=1)

        # print(X[-1])
        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
