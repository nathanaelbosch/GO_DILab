from os.path import abspath
import math
import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

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
        self.training_size = 50000

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

    def get_symmetries(self, boards, moves):
        """Given arramovess containing boards and moves recreate all smovesmmetries

        Also checks if moves are given using one-hot encoding alreadmoves
        """
        # print(boards.shape)
        boards = boards.reshape((boards.shape[0], 9, 9))

        passes, moves = moves[:, 81], moves[:, :81]
        moves = moves.reshape((moves.shape[0], 9, 9))

        boards_90 = np.rot90(boards, axes=(1, 2))
        moves_90 = np.rot90(moves, axes=(1, 2))
        boards_180 = np.rot90(boards, k=2, axes=(1, 2))
        moves_180 = np.rot90(moves, k=2, axes=(1, 2))
        boards_270 = np.rot90(boards, k=3, axes=(1, 2))
        moves_270 = np.rot90(moves, k=3, axes=(1, 2))
        boards_flipped = np.fliplr(boards)
        moves_flipped = np.fliplr(moves)
        boards_flipped_90 = np.rot90(np.fliplr(boards), axes=(1, 2))
        moves_flipped_90 = np.rot90(np.fliplr(moves), axes=(1, 2))
        boards_flipped_180 = np.rot90(np.fliplr(boards), k=2, axes=(1, 2))
        moves_flipped_180 = np.rot90(np.fliplr(moves), k=2, axes=(1, 2))
        boards_flipped_270 = np.rot90(np.fliplr(boards), k=3, axes=(1, 2))
        moves_flipped_270 = np.rot90(np.fliplr(moves), k=3, axes=(1, 2))

        boards = np.concatenate((
            boards,
            boards_90,
            boards_180,
            boards_270,
            boards_flipped,
            boards_flipped_90,
            boards_flipped_180,
            boards_flipped_270))
        boards = boards.reshape((boards.shape[0], 81))

        moves = np.concatenate((
            moves,
            moves_90,
            moves_180,
            moves_270,
            moves_flipped,
            moves_flipped_90,
            moves_flipped_180,
            moves_flipped_270))
        moves = moves.reshape((moves.shape[0], 81))
        passes = np.concatenate(
            (passes, passes, passes, passes, passes, passes, passes, passes))
        print(passes[:, None].shape)
        moves = np.concatenate((moves, passes[:, None]), axis=1)

        print('boards.shape:', boards.shape)
        print('moves.shape:', moves.shape)
        return boards, moves


    def handle_data(self, training_data):
        ids = training_data[:, 0]
        colors = training_data[:, 1]
        moves = training_data[:, 2]
        boards = training_data[:, 3:]

        # Generate y
        moves[moves==-1] = 81
        y = np.zeros((moves.shape[0], 82))
        y = to_categorical(moves)
        assert (y.sum(axis=1) == 1).all()

        # Generate X
        X = boards.astype(np.float64)

        # Replace values as you like to do
        X[X==BLACK] = BLACK_val
        X[X==EMPTY] = EMPTY_val
        X[X==WHITE] = WHITE_val
        assert X[0, 0] in [BLACK_val, EMPTY_val, WHITE_val]

        # Generate symmetries:
        X, y = self.get_symmetries(X, y)

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
