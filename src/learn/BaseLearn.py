import logging
import os
import sqlite3
import numpy as np
import time
from os.path import dirname, abspath
from abc import ABC, abstractmethod

from src import Utils
Utils.set_keras_backend("tensorflow")

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
        numb_all_games = cursor.execute('SELECT COUNT(*) FROM meta').fetchone()[0]
        self.invalid_game_ids = [_id[0] for _id in
                                 cursor.execute('SELECT id FROM meta WHERE all_moves_imported=0').fetchall()]
        self.log('database contains ' + str(numb_all_games) + ' games, ' + str(len(self.invalid_game_ids))
                 + ' are invalid and won\'t be used for training')
        self.numb_games_to_learn_from = 5

    def log(self, msg):
        self.logger.info(msg)
        print(msg)

    def get_sgf(self, game_id):
        return self.db.cursor().execute('SELECT sgf_content FROM meta WHERE id=?', (game_id,)).fetchone()[0]

    @abstractmethod
    def handle_row(self, X, Y, game_id, color, flat_move, board):
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
        self.log('starting the training with moves from ' + str(self.numb_games_to_learn_from)
                 + ' as input: ' + self.get_path_to_self())
        X = []
        Y = []
        cursor = self.db.cursor()
        cursor.execute('SELECT * FROM games')
        game_ids_learned_from = []
        last_game_id = None
        for i, row in enumerate(cursor):
            game_id = row[0]
            if game_id in self.invalid_game_ids:
                continue
            if last_game_id != game_id:
                if len(game_ids_learned_from) == self.numb_games_to_learn_from:
                    break
                game_ids_learned_from.append(game_id)
                last_game_id = game_id
            color = row[1]
            flat_move = row[2]
            board = row[3:]
            self.handle_row(X, Y, game_id, color, flat_move, board)

        X = np.array(X)
        Y = np.array(Y)

        # SET UP AND STORE NETWORK TOPOLOGY
        model = self.setup_and_compile_model()
        architecture_path = os.path.join(dirname(self.get_path_to_self()), 'model_architecture.json')
        json_file = open(architecture_path, 'w')
        json_file.write(model.to_json())
        json_file.close()

        # TRAIN AND STORE WEIGHTS
        self.train(model, X, Y)
        weights_path = os.path.join(dirname(self.get_path_to_self()), 'model_weights.h5')
        model.save_weights(weights_path)

        # EVALUATE
        scores = model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # DONE
        elapsed_time = time.time() - start_time
        self.log('training ended after ' + '{0:.0f}'.format(elapsed_time) + 's')
        self.log('model trained on ' + str(len(X)) + ' moves from ' + str(self.numb_games_to_learn_from) + ' games')
        self.log('model architecture saved to: ' + architecture_path)
        self.log('model weights saved to: ' + weights_path)
