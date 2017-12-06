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
        self.training_size = 50

    @staticmethod
    def apply_transf(flat_move, transf_matrix):
        if flat_move == 0:
            return 0
        row = int(math.floor(flat_move / 9))
        col = int(flat_move % 9)
        coord = col, -row
        # this matrix multiplication approach is from Yushan
        coord_transf = np.dot(transf_matrix, coord - CENTER) + CENTER
        flat_move_transf = coord_transf[1] * 9 + coord_transf[0]
        return flat_move_transf

    def handle_data(self, training_data):
        # ids = training_data[:, 0]
        # colors = training_data[:, 1]
        moves = training_data[:, 2]
        moves += 1
        boards = training_data[:, 3:]

        boards[boards == EMPTY] = EMPTY_val
        boards[boards == BLACK] = BLACK_val
        boards[boards == WHITE] = WHITE_val

        n = boards.shape[0] * 8
        X = np.ndarray(shape=(n, 81))
        Y = np.zeros(shape=(n, 82))

        i = 0
        for k, board in enumerate(boards):
            move = moves[k]
            matrix = board.reshape(9, 9)
            hflip_matrix = np.fliplr(matrix)
            X[i] = board
            Y[i][move] = 1
            i += 1
            X[i] = np.rot90(matrix, 1).reshape(81)
            Y[i][self.apply_transf(move, rot90_transf)] = 1
            i += 1
            X[i] = np.rot90(matrix, 2).reshape(81)
            Y[i][self.apply_transf(move, rot180_transf)] = 1
            i += 1
            X[i] = np.rot90(matrix, 3).reshape(81)
            Y[i][self.apply_transf(move, rot270_transf)] = 1
            i += 1
            X[i] = hflip_matrix.reshape(81)
            Y[i][self.apply_transf(move, hflip_transf)] = 1
            i += 1
            X[i] = np.rot90(hflip_matrix, 1).reshape(81)
            Y[i][self.apply_transf(move, hflip_rot90_transf)] = 1
            i += 1
            X[i] = np.rot90(hflip_matrix, 2).reshape(81)
            Y[i][self.apply_transf(move, hflip_rot180_transf)] = 1
            i += 1
            X[i] = np.rot90(hflip_matrix, 3).reshape(81)
            Y[i][self.apply_transf(move, hflip_rot270_transf)] = 1

        print('X.shape:', X.shape)
        print('Y.shape:', Y.shape)

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
