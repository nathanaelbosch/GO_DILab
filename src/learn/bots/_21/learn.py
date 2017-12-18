"""Basically what I did in WinPredictionBot

Value-Network with 3*81-vector as input containing the encoded board
"""
import os
from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, training_data):
        data = utils.separate_data(training_data)

        boards, training_data = self.get_symmetries(
            data['boards'], other_data=training_data)
        data = utils.separate_data(training_data)

        y = utils.value_output(data['results'])

        X = utils.encode_board(boards, data['colors'])

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
