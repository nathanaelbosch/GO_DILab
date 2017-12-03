import os
import sqlite3
import numpy as np
from time import strftime
from os.path import dirname, abspath
from abc import ABC, abstractmethod

from src import Utils
Utils.set_keras_backend("tensorflow")

project_root_dir = dirname(dirname(dirname(abspath(__file__))))
data_dir = os.path.join(project_root_dir, 'data')
log_dir = os.path.join(project_root_dir, 'logs')
db_path = os.path.join(data_dir, 'db.sqlite')
if not os.path.exists(db_path):
    print('no db found at: ' + db_path)
    exit(1)


class BaseLearn(ABC):

    def __init__(self):
        self.db = sqlite3.connect(db_path)
        cursor = self.db.cursor()
        report_filename = 'learn_' + strftime('%d%m%Y-%H%M%S') + '.log'
        self.report = open(os.path.join(log_dir, report_filename), 'w')
        numb_all_games = cursor.execute('SELECT COUNT(*) FROM meta').fetchone()[0]
        self.invalid_game_ids = [_id[0] for _id in
                                 cursor.execute('SELECT id FROM meta WHERE all_moves_imported=0').fetchall()]
        self.log('database contains ' + str(numb_all_games) + ' games, ' + str(len(self.invalid_game_ids))
                 + ' are invalid and won\'t be used for training')
        self.numb_games_to_learn_from = 5

    def log(self, msg):
        self.report.write(msg + '\n')
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
    def get_model_store_dir(self):
        pass

    def run(self):
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
        architecture_path = os.path.join(self.get_model_store_dir(), 'model_architecture.json')
        json_file = open(architecture_path, 'w')
        json_file.write(model.to_json())
        json_file.close()

        # TRAIN AND STORE WEIGHTS
        self.train(model, X, Y)
        weights_path = os.path.join(self.get_model_store_dir(), 'model_weights.h5')
        model.save_weights(weights_path)

        # EVALUATE
        scores = model.evaluate(X, Y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        # DONE
        self.log('model trained on ' + str(len(X)) + ' moves from ' + str(self.numb_games_to_learn_from) + ' games')
        self.log('model architecture saved to: ' + architecture_path)
        self.log('model weights saved to: ' + weights_path)

        self.report.close()
