from os.path import abspath
import numpy as np
import sqlite3

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

from src.learn.BaseLearn import BaseLearn
from src.play.model.Board import EMPTY, BLACK, WHITE


class Learn(BaseLearn):

    def __init__(self):
        super().__init__()
        self.training_size = 100000
        self.data_retrieval_command = '''SELECT games.*, meta.result
                                         FROM games, meta
                                         WHERE games.id == meta.id
                                         AND meta.all_moves_imported!=0
                                         ORDER BY RANDOM()
                                         LIMIT ?'''

    def handle_data(self, training_data):
        # results_array = training_data[:, -1]
        results_array = training_data.result
        # ids = training_data[:, 0]
        # colors = training_data[:, 1]
        # moves = training_data[:, 2].astype(int)
        # moves = training_data.move
        boards = training_data[training_data.columns[3:-1]].as_matrix()

        # Moves as categorical data
        # moves[moves==-1] = 81
        # moves_categorical = to_categorical(moves, 82)
        # assert moves_categorical.shape[1] == 82
        # assert (moves_categorical.sum(axis=1) == 1).all()

        # Generate symmetries:
        boards, results_array = self.get_symmetries(
            boards, other_data=results_array)

        # Input: Board
        X = np.concatenate(
            ((boards==WHITE)*3 - 1,
             (boards==BLACK)*3 - 1,
             (boards==EMPTY)*3 - 1),
            axis=1)
        X = X / np.sqrt(2)
        print('X.mean():', X.mean())
        print('X.var():', X.var())

        # Output: Result
        results = np.chararray(results_array.shape)
        results[:] = results_array[:]
        black_wins = results.lower().startswith(b'b')[:, None]
        white_wins = results.lower().startswith(b'w')[:, None]
        # draws = results.lower().startswith('D')
        y = np.concatenate((black_wins, white_wins), axis=1)

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def setup_and_compile_model(self):
        model = Sequential()
        model.add(Dense(200, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(400, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(200, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        return model

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=8, batch_size=10000)

    def get_path_to_self(self):
        return abspath(__file__)


if __name__ == '__main__':
    Learn().run()
