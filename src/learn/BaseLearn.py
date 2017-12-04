import logging
import os
import sqlite3
import numpy as np
import time
from os.path import dirname, abspath
from abc import ABC, abstractmethod

from src import Utils
# Utils.set_keras_backend("tensorflow")

project_root_dir = dirname(dirname(dirname(abspath(__file__))))
log_dir = os.path.join(project_root_dir, 'logs')
db_path = os.path.join(project_root_dir, 'data', 'db.sqlite')
if not os.path.exists(db_path):
    print('no db found at: ' + db_path)
    exit(1)


class BaseLearn(ABC):

    def __init__(self):
        self.db = sqlite3.connect(db_path)
        cursor = self.db.cursor()
        self.logger = Utils.get_unique_file_logger(self, logging.INFO)
        self.numb_all_games = cursor.execute('SELECT COUNT(*) FROM meta').fetchone()[0]
        self.invalid_game_ids = [_id[0] for _id in
                                 cursor.execute('SELECT id FROM meta WHERE all_moves_imported=0').fetchall()]
        self.log('database contains ' + str(self.numb_all_games) + ' games, ' + str(len(self.invalid_game_ids))
                 + ' are invalid and won\'t be used for training')
        self.numb_games_to_learn_from = self.numb_all_games  # overwrite this in your extending Learn class as desired

    def log(self, msg):
        self.logger.info(msg)
        print(msg)

    # can be used by Learn classes extending this class that require more game-info during training
    def get_sgf(self, game_id):
        return self.db.cursor().execute('SELECT sgf_content FROM meta WHERE id=?', (game_id,)).fetchone()[0]

    @staticmethod
    def append_to_numpy_array(base, new):
        if base is None:
            base = np.array([new])
        else:
            base = np.vstack((base, new))
        return base

    @staticmethod
    def flatten_matrix(matrix):
        return np.array([val for row in matrix.tolist() for val in row])  # better command?

    @abstractmethod
    def handle_data(self, result):
        pass

    @abstractmethod
    def setup_and_compile_model(self):
        pass

    @abstractmethod
    def train(self, model, X, Y):
        pass

    @abstractmethod
    def get_path_to_self(self):
        pass

    # can be overwritten by extending classes, doesn't have to though
    def customize_color_values(self, flat_board):
        return flat_board

    def run(self):
        start_time = time.time()
        self.log('starting the training with moves from '
                 + ('all ' if self.numb_games_to_learn_from == self.numb_all_games else '')
                 + str(self.numb_games_to_learn_from) + ' games as input ' + self.get_path_to_self())

        # Get data from Database
        cursor = self.db.cursor()
        cursor.execute('SELECT * FROM games LIMIT ?',
                       [self.numb_games_to_learn_from])
        result = cursor.fetchall()

        X, y = self.handle_data(result)

        # SET UP AND STORE NETWORK TOPOLOGY
        model = self.setup_and_compile_model()
        architecture_path = os.path.join(dirname(self.get_path_to_self()), 'model_architecture.json')
        json_file = open(architecture_path, 'w')
        json_file.write(model.to_json())
        json_file.close()

        # TRAIN AND STORE WEIGHTS
        self.train(model, X, y)
        weights_path = os.path.join(dirname(self.get_path_to_self()), 'model_weights.h5')
        model.save_weights(weights_path)

        # EVALUATE
        scores = model.evaluate(X, y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # DONE
        elapsed_time = time.time() - start_time
        self.log('training ended after {:.0f}s'.format(elapsed_time))
        self.log('model trained on {} moves from {} games'.format(
                 len(X), self.numb_games_to_learn_from))
        self.log('model architecture saved to: ' + architecture_path)
        self.log('model weights saved to: ' + weights_path)
