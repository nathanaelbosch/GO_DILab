"""Intermediary class to define the common setup and add some funcitons

The multiple bots here are just all combinations of input and output type
presented in the markdown. It therefore makes sense to define the
generation of those data formats here, as they will be shared accross
multiple bots.
"""
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from src.learn.BaseLearn import BaseLearn


class CommonLearn(BaseLearn):

    def __init__(self):
        super().__init__()
        np.random.seed(1234)
        # Training size is here not the number of rows, but of games!
        # Max 150000
        self.training_size = 400000

        # self.training_size = 1000
        # self.data_retrieval_command = '''
        #     WITH relevant_games as (
        #         SELECT id,
        #             CASE WHEN elo_black <= elo_white THEN elo_black
        #             WHEN elo_white <= elo_black THEN elo_white
        #             END AS min_elo
        #         FROM meta
        #         WHERE elo_white != ""
        #         AND elo_black != ""
        #         AND turns > 30
        #         AND result != 'Draw'
        #         AND result != 'Time'
        #         AND all_moves_imported!=0
        #         ORDER BY min_elo DESC
        #         LIMIT ?)
        #     SELECT games.*, meta.result, relevant_games.min_elo
        #     FROM relevant_games, games, meta
        #     WHERE relevant_games.id == games.id
        #     AND games.id == meta.id'''

        self.data_retrieval_command = '''
            SELECT *
            FROM elo_ordered_games
            ORDER BY RANDOM()
            LIMIT ?'''

        # self.data_retrieval_command = '''SELECT games.*,
        #                                     meta.result,
        #                                     meta.elo_black
        #                                  FROM games, meta
        #                                  WHERE games.id == meta.id
        #                                  AND meta.all_moves_imported!=0
        #                                  -- AND meta.elo_black > 1000
        #                                  -- AND meta.elo_white > 1000
        #                                  ORDER BY RANDOM()
        #                                  LIMIT ?'''

    def setup_and_compile_model(self):
        model = Sequential()
        DROPOUT = 0
        model.add(Dense(200, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(400, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(200, input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(DROPOUT))
        # model.add(Dense(100, input_dim=self.input_dim, activation='relu'))
        # model.add(Dropout(DROPOUT))
        model.add(Dense(self.output_dim, activation='softmax'))
        adam = keras.optimizers.Adam()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])
        return model

    def train(self, model, X, Y):
        model.fit(X, Y, epochs=30, batch_size=100000)
        # model.fit(X, Y, epochs=3, batch_size=1000)


if __name__ == '__main__':
    CommonLearn().run()
