import random

import Move
from Board import WHITE, BLACK

GROUPING_PROBABILITY = 0.75


class RandomGroupingBot:
    """
    The RandomGroupingBot first retrieves all playable locations.
    From those it will select one move:
        - with a probability of grouping_prob it will be in touch with an existing group
        - with a probability of 1-grouping_prob it will be any of the playable locations
          (including the group-touching locations)
    """
    @staticmethod
    def neighbors_contain_my_color(color, move, board):
        color = WHITE if color == 'w' else BLACK
        for neighbor_coord in board.get_all_neighbor_coords(move.to_matrix_location()):
            if board[neighbor_coord] == color:
                return True
        return False

    def genmove(self, color, game) -> Move:
        playable_locations = game.get_playable_locations(color)

        if random.uniform(0, 1) <= GROUPING_PROBABILITY:
            playable_locations_touching_groups = []
            for move in playable_locations:
                if move.is_pass is False and self.neighbors_contain_my_color(color, move, game.board):
                    playable_locations_touching_groups.append(move)

            if len(playable_locations_touching_groups) > 0:
                return random.choice(playable_locations_touching_groups)

        return random.choice(playable_locations)
