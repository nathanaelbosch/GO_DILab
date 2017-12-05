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

    def handle_data(self, training_data):
        ids = training_data[:, 0]
        colors = training_data[:, 1]
        moves = training_data[:, 2]
        moves += 1
        boards = training_data[:, 3:]

        # Generate y
        y = np.zeros((moves.shape[0], 82))
        y[np.arange(len(y)), moves] = 1
        assert (y.sum(axis=1) == 1).all()

        # Generate X
        X = boards

        # Replace values as you like to do
        X[X==BLACK] = BLACK_val
        X[X==EMPTY] = EMPTY_val
        X[X==WHITE] = WHITE_val

        print('X.shape:', X.shape)
        print('y.shape:', y.shape)

        return X, y

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
