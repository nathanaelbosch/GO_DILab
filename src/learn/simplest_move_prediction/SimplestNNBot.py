import numpy as np
import os
from os.path import abspath, dirname

from src.play.model.Move import Move


class SimplestNNBot:

    def __init__(self):
        self.model = None

    def genmove(self, color, game) -> Move:
        if self.model is None:
            # run.py and GTPengine.py import this bot, this calls __init__ even so the bot might not be used.
            # Importing keras and loading the model creates log-entries in the console that we don't need to see
            # when this bot is not used. Therefore initialize upon first genmove request, not in __init__
            import keras
            self.model = keras.models.load_model(os.path.join(dirname(abspath(__file__)), 'model.h5'))
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
