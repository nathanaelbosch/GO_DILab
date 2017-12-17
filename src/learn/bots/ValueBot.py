"""Common wrapper for the ValueBot-logic

The real input generation still depends on the actual bot, but the game logic
on how to play, given the output of some value network, is the same
"""
import copy

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
        flat_board = flat_board.reshape(1, len(flat_board))
        my_index = 0 if color == 'b' else 1
        my_value = BLACK if color == 'b' else WHITE

        inp = self.board_to_input(flat_board)
        current_prob = self.model.predict(inp)

        # 1. Our win probability
        my_prob = current_prob[0, my_index]

        # 2. Evaluate all possible moves
        playable_locations = game.get_playable_locations(color)
        best_move = Move(is_pass=True)
        best_win_prob = my_prob
        for move in playable_locations:
            if move.is_pass:
                continue

            # Play the move and evaluate the resulting board
            test_board = copy.deepcopy(game.board)
            test_board.place_stone_and_capture_if_applicable_default_values(
                move.to_matrix_location(), my_value)
            inp = self.board_to_input(test_board.flatten())
            pred_result = self.model.predict(inp)[0, my_index]

            if pred_result > best_win_prob:
                best_move = move
                best_win_prob = pred_result

        return best_move
