import numpy as np
from os.path import abspath, dirname
from src.play.model.Move import Move
from src.play.model.Board import WHITE, BLACK, EMPTY
import sys
import os
import logging
from os.path import abspath, dirname

from src.play.model.Move import Move


bot_logger = logging.getLogger(__name__)


class SimplestNNBot:

    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(
            abspath(__file__)))))
        import keras
        model_path = os.path.join(
            project_dir, 'src/learn/dev_nath/model.h5')
        self.model = keras.models.load_model(model_path)
        # self.model = keras.models.load_model('src/learn/dev_nath/model.h5')
        mean_var_path = os.path.join(
            project_dir, 'src/learn/dev_nath/mean_var.txt')
        with open(mean_var_path, 'r') as f:
        # with open('src/learn/dev_nath/mean_var.txt', 'r') as f:
            lines = f.readlines()
        self.mean = float(lines[0])
        self.std = float(lines[1])

    def board_to_input(self, color, board):
        b = board.astype(np.float64)

        b -= self.mean
        b /= self.std
        if color == 'b':
            me = BLACK
            other = WHITE
        else:
            me = WHITE
            other = BLACK
        my_board = (b == me) * 2 - 1
        other_board = (b == other) * 2 - 1

        my_board_vect = my_board.reshape(
            1, my_board.shape[0]*my_board.shape[1])
        other_board_vect = other_board.reshape(
            1, other_board.shape[0]*other_board.shape[1])
        a = np.append([my_board_vect, other_board_vect], [])
        a = a.reshape(
            (1, my_board_vect.shape[1]+other_board_vect.shape[1]))
        return a

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def genmove(self, color, game) -> Move:
        # We're still interested in the playable locations
        playable_locations = game.get_playable_locations(color)

        # Format the board and make predictions
        inp = self.board_to_input(color, game.board)
        bot_logger.debug('Input shape:', inp.shape)
        bot_logger.debug('Input:', inp)
        pred_moves = self.model.predict(inp)
        # pred_moves = self.model.predict(np.zeros((1, 162)))
        bot_logger.debug('This worked')
        bot_logger.debug('Predicted moves:', pred_moves)

        pred_moves = pred_moves.reshape(9, 9)
        # print(pred_moves)
        # print(playable_locations)
        dummy_value = -10
        potential_moves = np.array([[dummy_value]*9]*9, dtype=float)
        for move in playable_locations:
            # print(move)
            if move.is_pass:
                continue
            loc = move.to_matrix_location()
            potential_moves[loc[0]][loc[1]] = pred_moves[loc[0]][loc[1]]

        # print([i for row in potential_moves for i in row])

        potential_moves = self.softmax(potential_moves)
        bot_logger.debug('Potential moves:', potential_moves)

        row, col = np.unravel_index(
            potential_moves.argmax(),
            potential_moves.shape)

        move = Move(col=col, row=row)
        if (potential_moves[move.to_matrix_location()] == dummy_value or
                potential_moves[move.to_matrix_location()] < (1/81+0.0001)):
            move = Move(is_pass=True)

        return move
