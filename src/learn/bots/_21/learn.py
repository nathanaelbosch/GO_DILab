"""Basically what I did in WinPredictionBot

Value-Network with 3*81-vector as input containing the encoded board
"""
import os
from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, training_data):
        data = utils.separate_data(training_data)

        X, _other = self.get_symmetries(
            data['boards'], other_data=[data['results'], data['colors']])
        results, colors = _other[0], _other[1]

        X = utils.encode_board(X, colors)

        y = utils.value_output(results, colors)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
