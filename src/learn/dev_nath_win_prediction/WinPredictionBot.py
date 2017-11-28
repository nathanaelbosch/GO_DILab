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
        with open(mean_var_path, 'r') as f:
        # with open('src/learn/dev_nath/mean_var.txt', 'r') as f:
            lines = f.readlines()
        self.mean = float(lines[0])
        self.std = float(lines[1])

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
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def genmove(self, color, game) -> Move:
        # We're still interested in the playable locations
        playable_locations = game.get_playable_locations(color)

        inp = self.board_to_input(color, game.board)
        current_black_win = self.model.predict(inp)

        my_value = BLACK if color == 'b' else WHITE
        results = np.zeros(game.board.shape)
        if color == 'w':
            results = 1 - results

        for move in playable_locations:
            if move.is_pass:
                continue

            test_board = copy.deepcopy(game.board)
            test_board[move.to_matrix_location()] = my_value
            inp = self.board_to_input(color, test_board)
            pred_result = self.model.predict(inp)

            results[move.to_matrix_location()] = pred_result

        results -= current_black_win

        """ `results` now contains our prediction of blacks win probabilities
        for each move, adjusted by blacks current win probability. We can now
        easily check if a move is worth playing (as black) by checking the
        sign; If it is negative, our probability to win gets wors. In general
        the higher the number in `results` the better the move if we are black
        If we are white we swap the sign and we get the same behaviour"""
        if color == 'w':
            results = -results

        row, col = np.unravel_index(
            results.argmax(),
            results.shape)

        move = Move(col=col, row=row)
        if (results[move.to_matrix_location()] <= 0):
            move = Move(is_pass=True)

        # print('Returned move:', move.to_gtp(9))

        return move
