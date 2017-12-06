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
        self.numb_all_games = cursor.execute(
            'SELECT COUNT(*) FROM meta').fetchone()[0]
        self.games_table_length = cursor.execute(
            'SELECT COUNT(*) FROM games').fetchone()[0]
        self.invalid_game_ids = cursor.execute(
            'SELECT id FROM meta WHERE all_moves_imported=0').fetchall()
        self.log('''database contains {} games,
            {} are invalid and won\'t be used for training'''.format(
            self.numb_all_games, len(self.invalid_game_ids)))
        self.training_size = self.games_table_length  # override this in your Learn class as desired

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

    @abstractmethod
    def handle_data(self, training_data):
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

    def run(self):
        start_time = time.time()
        # self.log('starting the training with moves from '
        #          + ('all ' if self.numb_games_to_learn_from == self.numb_all_games else '')
        #          + str(self.numb_games_to_learn_from) + ' games as input ' + self.get_path_to_self())

        # Get data from Database
        cursor = self.db.cursor()
        cursor.execute('''SELECT games.*
                          FROM games, meta
                          WHERE games.id == meta.id
                          AND meta.all_moves_imported!=0
                          LIMIT ?''',
                       [self.training_size])
        training_data = np.array(cursor.fetchall())  # this is a gigantic array, has millions of rows

        self.log('working with {} rows'.format(len(training_data)))
        X, Y = self.handle_data(training_data)

        # SET UP and STORE NETWORK TOPOLOGY as json
        model = self.setup_and_compile_model()
        architecture_path = os.path.join(dirname(self.get_path_to_self()), 'model_architecture.json')
        json_file = open(architecture_path, 'w')
        json_file.write(model.to_json())
        json_file.close()

        # TRAIN and then STORE WEIGHTS as hdf5
        self.train(model, X, Y)
        weights_path = os.path.join(dirname(self.get_path_to_self()), 'model_weights.h5')
        model.save_weights(weights_path)

        # EVALUATE
        scores = model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # DONE
        elapsed_time = time.time() - start_time
        self.log('training ended after {:.0f}s'.format(elapsed_time))
        self.log('model trained on {} moves from {} games'.format(
                 len(X), self.numb_all_games))
        self.log('model architecture saved to: ' + architecture_path)
        self.log('model weights saved to: ' + weights_path)
