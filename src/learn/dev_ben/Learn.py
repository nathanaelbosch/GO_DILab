from os.path import abspath, dirname

from keras import Sequential
from keras.layers import Dense

from src.learn.BaseLearn import BaseLearn


class Learn(BaseLearn):

    def __init__(self):
        super().__init__()

    def handle_row(self, X, Y, game_id, color, flat_move, board):
        pass

    def setup_and_compile_model(self):
        model = Sequential()
        model.add(Dense(162, input_dim=81, activation='relu'))  # first parameter of Dense is number of neurons
        model.add(Dense(82, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=20)

    def get_model_store_dir(self):
        return dirname(abspath(__file__))


if __name__ == '__main__':
    Learn().run()
