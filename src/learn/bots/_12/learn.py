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
    def handle_data(self, data):
        boards = data[data.columns[3:-2]].as_matrix()

        y = utils.policy_output(data['move'])
        X, _other = self.get_symmetries(
            boards, other_data=[y, data['color']])
        y, colors = _other

        X = np.append(X, colors.reshape((len(colors), 1)), axis=1)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)
        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
