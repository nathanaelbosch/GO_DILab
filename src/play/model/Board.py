"""Class to purely handle everything that concerns the board"""
from typing import Tuple, List
import numpy as np

from src import Utils

if Utils.use_scipy():
    from scipy import ndimage

"""Just to adjust the internal representation of color at a single location,
instead of all over the code ;) Just in case. Maybe something else as -1 and 1
could be interesting, see the tick tack toe example"""
WHITE = -1
BLACK = 1
EMPTY = 0


class Board(np.matrix):
    """Class that purely handles the board, as well as board_related functions

    The motivation for this was also that we can make a copy of the real board,
    and evaluate all the `get_chain`, `check_dead` etc on the copy
    """
    def get_chain(self, loc: Tuple[int, int]) -> List[Tuple[int, int]]:
        # if run.py was started, we can use scipy and thereby improve performance
        if Utils.use_scipy():
            # This method uses morphological operations to find out the
            # connected components ie., chains. wikipedia link to
            # morphological operation - https://en.wikipedia.org/wiki/Mathematical_morphology
            test_matrix = self == self[loc]
            label_im, nb_labels = ndimage.label(test_matrix)
            label_im = label_im == label_im[loc]
            locations = np.where(label_im)
            group = list(zip(locations[0],locations[1]))
            return group
        # if GTPengine.py was started, via pyinstaller for instance, we can't use scipy
        # because pyinstaller doesn't seem to be able to handle it
        player = self[loc]
        # Check if neighbors of same player
        to_check = [loc]
        group = []
        while len(to_check) > 0:
            current = to_check.pop()
            neighbors = self.get_adjacent_coords(current)
            for n in neighbors:
                if self[n] == player and n not in group and n not in to_check:
                    to_check.append(n)
            group.append(current)
        return group

    def check_dead(self, group: List[Tuple[int, int]]) -> bool:
        """Check if a group is dead

        Currently done by getting all the neighbors, and checking if any
        of them is 0.
        """
        total_neighbors = []
        for loc in group:
            total_neighbors += self.get_adjacent_coords(loc)
        for n in total_neighbors:
            if self[n] == EMPTY:
                return False

        return True
        """ 
        Check if group is dead by morphological operation :
        This method works better with larger groups. Smaller groups can use the previous method
        """
        #recreate the region of interest ie., group
        # group_colour = self[group[0]]
        # roi = self[group]
        # print(roi)
        # a = ndimage.binary_dilation(roi)
        # b = -(a-roi)
        # b = np.lib.pad(self,((1,1),(1,1)),'constant')
        # x = np.lib.pad(self,((1,1),(1,1)),'constant',constant_values=-group_colour)
        # roi = np.lib.pad(roi, ((1, 1), (1, 1)), 'constant')
        # c = np.multiply(b,x) + roi
        # d = ndimage.binary_erosion(c)
        # new_roi = d[group]
        # old_roi = roi[group]
        # if new_roi == old_roi:
        #     return True
        # return False

    # currently only used in Ben's NN-dev
    def place_stone_and_capture_if_applicable(self, loc, player_val, opponent_val, empty_val):
        self[loc] = player_val
        # remove stones if this move captured them
        neighbors = self.get_adjacent_coords(loc)
        groups = []
        for n in neighbors:
            if self[n] == opponent_val:
                groups.append(self.get_chain(n))
        for g in groups:
            if self.check_dead(g):
                for c in g:
                    self[c] = empty_val

    def is_on_board(self, col, row):
        return 0 <= col < self.shape[0] and 0 <= row < self.shape[1]

    def get_all_neighbor_coords(self, loc: Tuple[int, int]):
        neighbors = []
        deltas = [
            (1, 1), (1, 0), (1, -1), (0, -1),
            (-1, -1), (-1, 0), (-1, 1), (0, 1)
        ]
        for delta in deltas:
            col = loc[0] + delta[0]
            row = loc[1] + delta[1]
            if self.is_on_board(col, row):
                neighbors.append((col, row))
        return neighbors

    def get_adjacent_coords(self, loc: Tuple[int, int]):
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

    def to_number(self):
        """Create a unique representation for a board

        Does this by creating an integer, with each position indicating a
        location on the board. I do this because performence gets bad once
        the board history is large
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

    def __str__(self):
        """String representation of the board!

        Just a simple ascii output, quite cool but the code is a bit messy"""
        b = self.copy()
        rows = list(range(b.shape[0]))
        cols = list(range(b.shape[1]))
        rows = [str(self.shape[0] - i) for i in rows]
        cols = [chr(i + ord('a')) if i < 8 else chr(i + 1 + ord('a')) for i in cols]
        # was previously not GTP conform:
        # rows = [chr(i + ord('a')) for i in rows]
        # cols = [chr(i + ord('a')) for i in cols]

        # You might wonder why I do the following, but its so that numpy
        # formats the str representation using a single space
        b[b == BLACK] = 2
        b[b == WHITE] = 3

        matrix_repr = super(Board, b).__str__()
        matrix_repr = matrix_repr.replace('2', 'X')
        matrix_repr = matrix_repr.replace('3', 'O')
        matrix_repr = matrix_repr.replace('0', '·')
        matrix_repr = matrix_repr.replace('[[', ' [')
        matrix_repr = matrix_repr.replace(']]', ']')

        col_index = ' '.join(cols)
        board_repr = ''
        for i in zip(rows, matrix_repr.splitlines()):
            board_repr += i[0]+i[1]+'\n'
        board_repr = ' '*3 + col_index+'\n'+board_repr
        return board_repr

    ###########################################################################
    # Not used yet, but more relevant to `Board` than to `Game`
    def _matrix2csv(self, matrix):
        """Transform a matrix to a string, using ';' as the separator"""
        ls = matrix.tolist()
        ls = [str(entry) for row in ls for entry in row]
        s = ';'.join(ls)
        return s

    def board2file(self, file, mode='a'):
        """Store board to a file

        The idea is also to create csv files that contain
        all boards that were part of a game, so that we can
        use those to train a network on.
        """
        string = self._matrix2csv(self.board)
        with open(file, mode) as f:
            f.write(string)
            f.write('\n')


if __name__ == '__main__':
    import doctest
    # doctest.testmod(extraglobs={'g': Game()})
    doctest.testmod()
