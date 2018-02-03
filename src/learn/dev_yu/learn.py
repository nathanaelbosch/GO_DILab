from os.path import abspath
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras import initializers, optimizers

from src.learn.BaseLearn import BaseLearn
from src.play.model.Board import EMPTY, BLACK, WHITE

EMPTY_val = 0.45
BLACK_val = -1.35
WHITE_val = 1.05


class Learn(BaseLearn):

    def __init__(self):
        super().__init__()
        self.training_size = 1000000
        self.data_retrieval_command = '''SELECT games.*
                                         FROM games, meta
                                         WHERE games.id == meta.id
                                         AND meta.all_moves_imported!=0
                                         ORDER BY RANDOM()
                                         LIMIT ?'''

    def handle_data(self, training_data):
        colors = training_data[:, 1]
        moves = training_data[:, 2].astype(int)
        boards = training_data[:, 3:].astype(np.float64)

        # Output Y as next moves
        # pass as class 82
        moves[moves == -1] = 81
        Y = to_categorical(moves, num_classes=82)
        assert Y.shape[1] == 82
        assert (Y.sum(axis=1) == 1).all()

        # Generate board symmetries, 8 for each board
        boards, Y = self.get_symmetries(boards, Y)
        colors = [i for i in colors for r in range(8)]

        # Input X as board and current player
        X = np.column_stack((boards, colors))

        # Replace values as you like to do
        X[X == BLACK] = BLACK_val
        X[X == EMPTY] = EMPTY_val
        X[X == WHITE] = WHITE_val
        assert X[0, 0] in [BLACK_val, EMPTY_val, WHITE_val]

        print('X.shape:', X.shape)
        print('Y.shape:', Y.shape)

        return X, Y

    def setup_and_compile_model(self):
        model = Sequential()
        model.add( Dense(200, kernel_initializer=initializers.RandomNormal(stddev=1 / math.sqrt(82)), bias_initializer='normal', input_dim=82, activation='relu'))
        model.add( Dense(400, kernel_initializer=initializers.RandomNormal(stddev=1 / math.sqrt(200)), bias_initializer='normal', activation='relu'))
        model.add( Dense(200, kernel_initializer=initializers.RandomNormal(stddev=1 / math.sqrt(400)), bias_initializer='normal', activation='relu'))
        model.add(Dense(82, kernel_initializer=initializers.RandomNormal(stddev=1/math.sqrt(200)), bias_initializer='normal', activation='softmax'))
        # softmax guarantees a probability distribution over the 81 locations and pass
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=200, batch_size=1000)


    def get_path_to_self(self):
        return abspath(__file__)


if __name__ == '__main__':
    Learn().run()
