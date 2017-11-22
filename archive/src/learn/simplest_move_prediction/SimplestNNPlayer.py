from src.play import Player
from src.play.utils.Move import Move
import random as rn
import numpy as np
import keras


class SimplestNNPlayer(Player):

    def __init__(self, name, color, game):
        self.model = keras.models.load_model(
            'src/learn/simplest_move_prediction/model.h5')
        Player.__init__(self, 'SimplestNN', color, game)

    def get_move(self):
        # We're still interested in the playable locations
        playable_locations = self.game.get_playable_locations(self.color)

        # Format the board and make predictions
        board = self.game.board.tolist()
        board = np.array([[entry for row in board for entry in row]])
        pred_moves = self.model.predict(board)
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
