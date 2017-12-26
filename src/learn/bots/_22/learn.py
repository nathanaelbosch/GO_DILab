"""Policy Network with naive board encoding"""
import os

from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, data):
        boards = data[data.columns[3:-2]].as_matrix()

        y = utils.policy_output(data['move'])

        boards, _other = self.get_symmetries(
            boards, other_data=[y, data['color']])
        y, colors = _other

        X = utils.encode_board(boards, colors)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


if __name__ == '__main__':
    Learn().run()
