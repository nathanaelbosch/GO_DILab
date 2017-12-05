from os.path import abspath
import math
import numpy as np

from keras import Sequential
from keras.layers import Dense

from src.learn.BaseLearn import BaseLearn
from src.play.model.Board import EMPTY, BLACK, WHITE

EMPTY_val = 0.45
BLACK_val = -1.35
WHITE_val = 1.05

CENTER = np.array([4, -4])
identity_transf = np.array([[1, 0], [0, 1]])
rot90_transf = np.array([[0, -1], [1, 0]])
rot180_transf = np.array([[-1, 0], [0, -1]])
rot270_transf = np.array([[0, 1], [-1, 0]])
hflip_transf = np.array([[-1, 0], [0, 1]])
hflip_rot90_transf = np.array([[0, -1], [-1, 0]])
hflip_rot180_transf = np.array([[1, 0], [0, -1]])
hflip_rot270_transf = np.array([[0, 1], [1, 0]])


class Learn(BaseLearn):

    def __init__(self):
        super().__init__()
        self.numb_games_to_learn_from = 10

    @staticmethod
    def apply_transf_and_flatten(flat_move, transf_matrix):
        row = int(math.floor(flat_move / 9))
        col = int(flat_move % 9)
        coord = col, -row
        # this matrix multiplication approach is from Yushan
        coord_transf = np.dot(transf_matrix, coord - CENTER) + CENTER
        flat_move_transf = coord_transf[1] * 9 + coord_transf[0]
        return flat_move_transf

    def append_symmetry(self, X, Y, board, flat_move, transf_matrix):
        flat_board = board.flatten()
        y = np.array([0 for _i in range(82)])
        if flat_move == -1:  # PASS
            y[0] = 1
        else:
            flat_move_transf = self.apply_transf_and_flatten(flat_move, transf_matrix)
            y[flat_move_transf + 1] = 1
        return self.append_to_numpy_array(X, flat_board), self.append_to_numpy_array(Y, y)

    @staticmethod
    def replace_value(value):
        if value == EMPTY:
            return EMPTY_val
        if value == BLACK:
            return BLACK_val
        if value == WHITE:
            return WHITE_val

    def customize_color_values(self, flat_board):
        return np.array([self.replace_value(entry) for entry in flat_board])

    def handle_row(self, X, Y, game_id, color, flat_move, flat_board):
        board = flat_board.reshape(9, 9)
        hflip_board = np.fliplr(board)
        X, Y = self.append_symmetry(X, Y, board, flat_move, identity_transf)
        X, Y = self.append_symmetry(X, Y, np.rot90(board, 1), flat_move, rot90_transf)
        X, Y = self.append_symmetry(X, Y, np.rot90(board, 2), flat_move, rot180_transf)
        X, Y = self.append_symmetry(X, Y, np.rot90(board, 3), flat_move, rot270_transf)
        X, Y = self.append_symmetry(X, Y, hflip_board, flat_move, identity_transf)
        X, Y = self.append_symmetry(X, Y, np.rot90(hflip_board, 1), flat_move, hflip_rot90_transf)
        X, Y = self.append_symmetry(X, Y, np.rot90(hflip_board, 2), flat_move, hflip_rot180_transf)
        X, Y = self.append_symmetry(X, Y, np.rot90(hflip_board, 3), flat_move, hflip_rot270_transf)

        # TODO
        # what to do about flipping colors? would double the 8 symmetries to 16
        # would require komi-math... but how

        return X, Y

    def setup_and_compile_model(self):
        model = Sequential()
        model.add(Dense(162, input_dim=81, activation='relu'))  # first parameter of Dense is number of neurons
        model.add(Dense(82, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=1)

    def get_path_to_self(self):
        return abspath(__file__)


if __name__ == '__main__':
    Learn().run()
