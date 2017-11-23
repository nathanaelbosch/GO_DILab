import numpy as np
from os.path import abspath, dirname
from src.play.model.Move import Move
from src.play.model.Board import WHITE, BLACK, EMPTY
import sys
import os
from os.path import abspath, dirname

from src.play.model.Move import Move


class SimplestNNBot:

    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(dirname(
            abspath(__file__))))))
        # try to make this path relative instead of absolute to project_dir,
        # otherwise it might make troubles when packing it into
        # an executable TODO
        # filepath = os.path.join(
            # project_dir, 'src/learn/dev_nath/model.h5')
        # self.model = keras.models.load_model(filepath)
        import keras
        self.model = keras.models.load_model('src/learn/dev_nath/model.h5')
        with open('src/learn/dev_nath/mean_var.txt', 'r') as f:
            lines = f.readlines()
        self.mean = float(lines[0])
        self.std = float(lines[1])

    def board_to_input(self, color, board):
        b = board.astype(np.float64)

        print(b.dtype)
        print(type(self.mean))
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
        print(my_board_vect.shape)
        print(other_board_vect.shape)
        a = np.append([my_board_vect, other_board_vect], [])
        a = a.reshape(
            (1, my_board_vect.shape[1]+other_board_vect.shape[1]))
        print(a.shape)
        print(type(a))
        return a

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def genmove(self, color, game) -> Move:
        # We're still interested in the playable locations
        playable_locations = game.get_playable_locations(color)

        # Format the board and make predictions
        inp = self.board_to_input(color, game.board)
        pred_moves = self.model.predict(inp)
        # print(pred_moves)

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
        print(potential_moves)

        row, col = np.unravel_index(
            potential_moves.argmax(),
            potential_moves.shape)

        move = Move(col=col, row=row)
        if (potential_moves[move.to_matrix_location()] == dummy_value or
                potential_moves[move.to_matrix_location()] < (1/81+0.001)):
            move = Move(is_pass=True)

        return move
