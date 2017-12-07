from os.path import abspath
import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

from src.learn.BaseLearn import BaseLearn
from src.play.model.Board import EMPTY, BLACK, WHITE


class Learn(BaseLearn):

    def __init__(self):
        super().__init__()
        self.training_size = 5000000

    def handle_data(self, training_data):
        moves = training_data[:, 2].astype(int)
        boards = training_data[:, 3:].astype(np.float64)

        # Moves as categorical data
        moves[moves==-1] = 81
        moves_categorical = to_categorical(moves)
        assert moves_categorical.shape[1] == 82
        assert (moves_categorical.sum(axis=1) == 1).all()

        # Generate symmetries:
        boards, moves_categorical = self.get_symmetries(
            boards, moves_categorical)

        # Create input and output data
        X = np.concatenate(
            ((boards==WHITE)*3 - 1,
             (boards==BLACK)*3 - 1,
             (boards==EMPTY)*3 - 1),
            axis=1)
        X = X / np.sqrt(2)
        print('X.mean():', X.mean())
        print('X.var():', X.var())
        y = moves_categorical

        print('X.shape:', X.shape)
        print('Y.shape:', y.shape)

        return X, y

    def setup_and_compile_model(self):
        model = Sequential()
        model.add(Dense(200, input_dim=self.input_dim, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        return model

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=1)

    def get_path_to_self(self):
        return abspath(__file__)


if __name__ == '__main__':
    Learn().run()
