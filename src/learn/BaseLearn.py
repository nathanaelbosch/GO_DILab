import logging
import os
import sqlite3
import numpy as np
import time
from os.path import dirname, abspath
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

from src import Utils
# Utils.set_keras_backend("tensorflow")

project_root_dir = dirname(dirname(dirname(abspath(__file__))))
log_dir = os.path.join(project_root_dir, 'logs')
db_path = os.path.join(project_root_dir, 'data', 'db.sqlite')


class BaseLearn(ABC):

    def __init__(self):
        if not os.path.exists(db_path):
            print('no db found at: ' + db_path)
            exit(1)

        self.db = sqlite3.connect(db_path)
        cursor = self.db.cursor()
        self.logger = Utils.get_unique_file_logger(self, logging.INFO)
        self.numb_all_games = cursor.execute(
            'SELECT COUNT(*) FROM meta').fetchone()[0]
        self.games_table_length = cursor.execute(
            'SELECT COUNT(*) FROM games').fetchone()[0]
        numb_invalid_games = cursor.execute(
            'SELECT COUNT(*) FROM meta WHERE all_moves_imported=0').fetchone()[0]
        self.log('''database contains {} games,
            {} are invalid and won\'t be used for training'''.format(
            self.numb_all_games, numb_invalid_games))
        self.training_size = self.games_table_length  # override this in your Learn class as desired
        self.data_retrieval_command = '''SELECT games.*
                                          FROM games, meta
                                          WHERE games.id == meta.id
                                          AND meta.all_moves_imported!=0
                                          LIMIT ?'''

    def log(self, msg):
        self.logger.info(msg)
        print(msg)

    # can be used by Learn classes extending this class that require more game-info during training
    def get_sgf(self, game_id):
        return self.db.cursor().execute('SELECT sgf_content FROM meta WHERE id=?', (game_id,)).fetchone()[0]

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

    @staticmethod
    def get_symmetries(boards, moves=None, other_data=None):
        """Given array containing boards and moves recreate all symmetries

        Right now it assumes that moves are given as categorical data already.
        Output arrays will be 8 times as long as the input data, to help
        keeping this in line with the other data you want to use for your
        training you can pass an additional array to `other_data` which will
        be returned with the adjusted length. Lines will still match.
        """
        # print(boards.shape)
        boards = boards.reshape((boards.shape[0], 9, 9))
        boards_90 = np.rot90(boards, axes=(1, 2))
        boards_180 = np.rot90(boards, k=2, axes=(1, 2))
        boards_270 = np.rot90(boards, k=3, axes=(1, 2))
        boards_flipped = np.fliplr(boards)
        boards_flipped_90 = np.rot90(np.fliplr(boards), axes=(1, 2))
        boards_flipped_180 = np.rot90(np.fliplr(boards), k=2, axes=(1, 2))
        boards_flipped_270 = np.rot90(np.fliplr(boards), k=3, axes=(1, 2))

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

        if moves is not None:
            passes, moves = moves[:, 81], moves[:, :81]
            moves = moves.reshape((moves.shape[0], 9, 9))

            moves_90 = np.rot90(moves, axes=(1, 2))
            moves_180 = np.rot90(moves, k=2, axes=(1, 2))
            moves_270 = np.rot90(moves, k=3, axes=(1, 2))
            moves_flipped = np.fliplr(moves)
            moves_flipped_90 = np.rot90(np.fliplr(moves), axes=(1, 2))
            moves_flipped_180 = np.rot90(np.fliplr(moves), k=2, axes=(1, 2))
            moves_flipped_270 = np.rot90(np.fliplr(moves), k=3, axes=(1, 2))

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
            moves = np.concatenate((moves, passes[:, None]), axis=1)

        # print('boards.shape:', boards.shape)
        # print('moves.shape:', moves.shape)
        if other_data is not None:
            other_data = np.concatenate(
                (other_data, other_data, other_data, other_data,
                 other_data, other_data, other_data, other_data))

        if moves is not None and other_data is not None:
            return boards, moves, other_data
        elif moves is not None and other_data is None:
            return boards, moves
        elif moves is None and other_data is not None:
            return boards, other_data
        else:
            return boards

    def run(self):
        start_time = time.time()
        # self.log('starting the training with moves from '
        #          + ('all ' if self.numb_games_to_learn_from == self.numb_all_games else '')
        #          + str(self.numb_games_to_learn_from) + ' games as input ' + self.get_path_to_self())

        # Get data from Database
        cursor = self.db.cursor()
        cursor.execute(self.data_retrieval_command,
                       [self.training_size])
        training_data = np.array(cursor.fetchall())  # this is a gigantic array, has millions of rows

        self.log('working with {} rows'.format(len(training_data)))
        X, Y = self.handle_data(training_data)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1)

        # Save input and output dimensions for easier, more modular use
        # Implicit assumtion is that X, y are two-dimensional
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]

        # SET UP AND STORE NETWORK TOPOLOGY
        model = self.setup_and_compile_model()
        architecture_path = os.path.join(dirname(self.get_path_to_self()), 'model_architecture.json')
        json_file = open(architecture_path, 'w')
        json_file.write(model.to_json())
        json_file.close()

        # TRAIN AND STORE WEIGHTS
        self.train(model, X_train, Y_train)
        weights_path = os.path.join(dirname(self.get_path_to_self()), 'model_weights.h5')
        model.save_weights(weights_path)

        # EVALUATE
        print('\n\n')
        scores = model.evaluate(X_test, Y_test)
        print("Test accuracy: {:.2f}%".format(scores[1] * 100))

        # DONE
        elapsed_time = time.time() - start_time
        self.log('training ended after {:.0f}s'.format(elapsed_time))
        self.log('model trained on {} moves from {} games'.format(
                 len(X), self.numb_all_games))
        self.log('model architecture saved to: ' + architecture_path)
        self.log('model weights saved to: ' + weights_path)
