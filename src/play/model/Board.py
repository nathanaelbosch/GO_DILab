"""Class to purely handle everything that concerns the board"""
import numpy as np
from typing import Tuple, List

# from src.play.model.Game import WHITE, BLACK
WHITE = -1
BLACK = 1


class Board(np.matrix):
    def _get_chain(self, loc: Tuple[int, int]) -> List[Tuple[int, int]]:
        player = self[loc]
        # Check if neighbors of same player
        to_check = [loc]
        group = []
        while len(to_check) > 0:
            current = to_check.pop()
            neighbors = self._get_adjacent_coords(current)
            for n in neighbors:
                if (self[n] == player and
                        n not in group and n not in to_check):
                    to_check.append(n)
            group.append(current)
        return group

    def _check_dead(self, group: List[Tuple[int, int]]) -> bool:
        """Check if a group is dead

        Currently done by getting all the neighbors, and checking if any
        of them is 0.
        """
        total_neighbors = []
        for loc in group:
            total_neighbors += self._get_adjacent_coords(loc)
        for n in total_neighbors:
            if self[n] == 0:
                return False
        return True

    def _get_adjacent_coords(self, loc: Tuple[int, int]):
        neighbors = []
        if loc[0] > 0:
            neighbors.append((loc[0]-1, loc[1]))
        if loc[0] < self.shape[0]-1:
            neighbors.append((loc[0]+1, loc[1]))
        if loc[1] > 0:
            neighbors.append((loc[0], loc[1]-1))
        if loc[1] < self.shape[1]-1:
            neighbors.append((loc[0], loc[1]+1))
        return neighbors

    def _board_to_number(self):
        """Basically just create a unique representation for a board

        I do this because performence gets bad once the board history is
        large
        """
        number = 0
        i = 0
        for entry in np.nditer(self):
            if entry == WHITE:
                number += 1 * 10**i
            elif entry == BLACK:
                number += 2 * 10**i
            else:
                number += 3 * 10**i
            i += 1
        return number
