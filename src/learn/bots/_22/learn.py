"""Policy Network with naive board encoding"""
import os
import numpy as np

from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, training_data):
        data = utils.separate_data(training_data)

        boards, training_data = self.get_symmetries(
            data['boards'], other_data=training_data)
        data = utils.separate_data(training_data)

        y = utils.policy_output(data['moves'])

        X = utils.encode_board(boards, data['colors'])

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
