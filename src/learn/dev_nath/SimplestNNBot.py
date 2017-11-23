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
        filepath = os.path.join(
            project_dir, 'src\learn\simplest_move_prediction\model.h5')
        self.model = keras.models.load_model(filepath)

    def board_to_input(self):
        b = self.game.board
        if self.color == 'b':
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
        return np.append(my_board_vect, other_board_vect)

    def genmove(self, color, game) -> Move:
        # We're still interested in the playable locations
        playable_locations = self.game.get_playable_locations(self.color)

        # Format the board and make predictions
        inp = self.board_to_input()
        pred_moves = self.model.predict(inp)
        # print(pred_moves)

        pred_moves = pred_moves.reshape(9, 9)
        # print(pred_moves)
        # print(playable_locations)
        dummy_value = -10 if self.color == 'b' else 10
        potential_moves = np.array([[dummy_value]*9]*9, dtype=float)
        for move in playable_locations:
            # print(move)
            if move.is_pass:
                continue
            loc = move.to_matrix_location()
            # print(loc)
            potential_moves[loc[0]][loc[1]] = pred_moves[loc[0]][loc[1]]
        # print([i for row in potential_moves for i in row])
        # print(self.color)
        if self.color == 'b':
            row, col = np.unravel_index(
                potential_moves.argmax(),
                potential_moves.shape)
        elif self.color == 'w':
            row, col = np.unravel_index(
                potential_moves.argmin(),
                potential_moves.shape)
        # random_choice = rn.choice(playable_locations)
        move = Move(col=col, row=row)

        if potential_moves[move.to_matrix_location()] == dummy_value:
            move = Move(is_pass=True)

        return move
