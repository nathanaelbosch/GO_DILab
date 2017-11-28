import numpy as np
from os.path import abspath, dirname
from src.play.model.Move import Move
from src.play.model.Board import WHITE, BLACK, EMPTY
import sys
import os
import logging
from os.path import abspath, dirname
import copy

from src.play.model.Move import Move


bot_logger = logging.getLogger(__name__)


class WinPredictionBot:

    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(
            abspath(__file__)))))
        import keras
        model_path = os.path.join(
            project_dir, 'src/learn/dev_nath_win_prediction/model.h5')
        self.model = keras.models.load_model(model_path)
        # self.model = keras.models.load_model('src/learn/dev_nath/model.h5')
        mean_var_path = os.path.join(
            project_dir, 'src/learn/dev_nath_win_prediction/mean_var.txt')
        # with open(mean_var_path, 'r') as f:
        # # with open('src/learn/dev_nath/mean_var.txt', 'r') as f:
        #     lines = f.readlines()
        # self.mean = float(lines[0])
        # self.std = float(lines[1])

    def __str__(self):
        return 'WinPredictionBot'

    def board_to_input(self, color, board):
        b = board.astype(np.float64)

        # b -= self.mean
        # b /= self.std

        black_board = (b == BLACK) * 2 - 1
        white_board = (b == WHITE) * 2 - 1

        black_board_vect = black_board.reshape(
            1, black_board.shape[0]*black_board.shape[1])
        white_board_vect = white_board.reshape(
            1, white_board.shape[0]*white_board.shape[1])
        a = np.append([black_board_vect, white_board_vect], [])
        a = a.reshape(
            (1, black_board_vect.shape[1]+white_board_vect.shape[1]))
        return a

    def softmax(self, x):
        # e_x = np.exp(x - np.max(x))
        # return e_x / e_x.sum()
        return x / np.sum(x, axis=1)

    def genmove(self, color, game) -> Move:
        my_index = 0 if color == 'b' else 1

        # We're still interested in the playable locations
        playable_locations = game.get_playable_locations(color)

        inp = self.board_to_input(color, game.board)
        current_pred = self.model.predict(inp)
        # print('Current outcome prediction:', current_pred)
        assert (self.softmax(current_pred) == current_pred).all()
        my_pred = current_pred[0, my_index]

        my_value = BLACK if color == 'b' else WHITE

        results = np.zeros(game.board.shape)

        for move in playable_locations:
            if move.is_pass:
                continue

            test_board = copy.deepcopy(game.board)
            test_board[move.to_matrix_location()] = my_value
            inp = self.board_to_input(color, test_board)
            pred_result = self.model.predict(inp)
            pred_result = self.softmax(pred_result)

            results[move.to_matrix_location()] = pred_result[0, my_index]

        # print(results>0)
        # print(my_pred)
        results -= my_pred
        # print(results>0)

        """ `results` now contains our prediction of our win probabilities
        for each move, adjusted by our current win probability. We can now
        easily check if a move is worth playing by checking the
        sign; If it is negative, our probability to win gets worse. In general
        the higher the number in `results` the better the move."""

        row, col = np.unravel_index(
            results.argmax(),
            results.shape)

        move = Move(col=col, row=row)
        if (results[move.to_matrix_location()] <= 0):
            move = Move(is_pass=True)

        # print('Returned move:', move.to_gtp(9))

        return move
