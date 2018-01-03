"""Common wrapper for the ValueBot-logic

The real input generation still depends on the actual bot, but the game logic
on how to play, given the output of some value network, is the same
"""
import copy
import numpy as np

from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Board import BLACK, WHITE
from src.play.model.Move import Move


class ValueBot(BaseNNBot):
    def _genmove(self, color, game, flat_board):
        """Generate a move - ValueBot logic

        The logic of this bot is basically:
        1. Evaluate current probability of winning
        2. Evaluate the probabilities of winning for each move
        3. Make the best move if there is a valid move that raises the probs
        """
        color = WHITE if color == 'w' else BLACK
        flat_board = flat_board.reshape(1, len(flat_board))
        my_value = color

        # 1. Get current Win Probability
        inp = self.generate_nn_input(flat_board, color)
        current_prob = self.model.predict(inp)
        assert np.sum(current_prob) == 1, np.sum(current_prob)
        # print(current_prob)

        # 2. Evaluate all possible moves
        best_win_prob = current_prob[0, 0]
        best_move = Move(is_pass=True)

        playable_locations = game.get_playable_locations(color)
        for move in playable_locations:
            if move.is_pass:
                continue

            # Play the move and evaluate the resulting board
            test_board = copy.deepcopy(game.board)
            test_board.place_stone_and_capture_if_applicable_default_values(
                move.to_matrix_location(), my_value)
            inp = self.generate_nn_input(test_board.flatten(), color)
            pred_result = self.model.predict(inp)[0, 0]

            if pred_result > best_win_prob:
                best_move = move
                best_win_prob = pred_result

        return best_move
