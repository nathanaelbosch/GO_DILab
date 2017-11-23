import keras
import numpy as np
from os.path import abspath, dirname

import sys

import os

from src.play.model.Move import Move


class SimplestNNBot:

    def __init__(self):
        project_dir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))
        # try to make this path relative instead of absolute to project_dir,
        # otherwise it might make troubles when packing it into an executable TODO
        filepath = os.path.join(project_dir, 'src\learn\simplest_move_prediction\model.h5')
        self.model = keras.models.load_model(filepath)

    def genmove(self, color, game) -> Move:
        # We're still interested in the playable locations
        playable_locations = game.get_playable_locations(color)

        # Format the board and make predictions
        board = game.board.tolist()
        board = np.array([[entry for row in board for entry in row]])
        pred_moves = self.model.predict(board)
        # print(pred_moves)

        pred_moves = pred_moves.reshape(9, 9)
        # print(pred_moves)
        # print(playable_locations)
        dummy_value = -10 if color == 'b' else 10
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
        if color == 'b':
            row, col = np.unravel_index(
                potential_moves.argmax(),
                potential_moves.shape)
        elif color == 'w':
            row, col = np.unravel_index(
                potential_moves.argmin(),
                potential_moves.shape)
        # random_choice = rn.choice(playable_locations)
        move = Move(col=col, row=row)

        if potential_moves[move.to_matrix_location()] == dummy_value:
            move = Move(is_pass=True)

        return move
