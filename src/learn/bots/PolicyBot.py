"""Common wrapper for the ValueBot-logic

The real input generation still depends on the actual bot, but the game logic
on how to play, given the output of some value network, is the same
"""
from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Move import Move


class PolicyBot(BaseNNBot):
    def _genmove(self, color, game, flat_board):
        """Generate a move - PolicyBot logic

        The logic of this bot is basically:
        1. Directly generate a move
        2. Take the valid move with the highest score
        """
        flat_board = flat_board.reshape(1, len(flat_board))

        # 1. Generate move probabilities
        inp = self.generate_nn_input(flat_board, color)
        prediction = self.model.predict(inp)[0]
        print(prediction.shape)

        # 2. Look at each valid move and take the best one
        # Yes, this is looped, bad perf, but it is intuitively understandable
        # and it leaves little room for errors!
        playable_locations = game.get_playable_locations(color)
        best_move = Move(is_pass=True)
        best_move_prob = prediction[81]
        for move in playable_locations:
            if move.is_pass:
                continue

            if prediction[move.to_flat_idx()] > best_move_prob:
                best_move = move
                best_move_prob = prediction[move.to_flat_idx()]

        return best_move
