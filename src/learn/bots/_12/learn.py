"""Policy Network with naive board encoding


This approach is inherently flawed. It only serves as a comparison for the
more refined approaches we have.

Input here is the naive board encoding but also the current players color.
"""
import os
import numpy as np

from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, training_data):
        data = utils.separate_data(training_data)

        y = utils.policy_output(data['moves'])
        X, y, colors = self.get_symmetries(
            data['boards'], moves=y, other_data=data['colors'])
        colors = colors[:, None]

        X = np.concatenate((X, colors), axis=1)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)
        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
