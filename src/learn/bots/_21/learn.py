"""Basically what I did in WinPredictionBot

Value-Network with 3*81-vector as input containing the encoded board
"""
import os
from src.learn.bots.CommonLearn import CommonLearn
import src.learn.bots.utils as utils


class Learn(CommonLearn):
    def handle_data(self, data):
        boards = data[data.columns[3:-2]].as_matrix()

        y = utils.value_output(data['result'], data['color'])

        # Get symmetries and duplicate others accordingly
        X, _other = self.get_symmetries(
            boards, other_data=[y, data['color']])
        y, colors = _other[0], _other[1]
        X = utils.encode_board(X, colors)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def get_path_to_self(self):
        return os.path.abspath(__file__)


def main():
    Learn().run()


if __name__ == '__main__':
    main()
