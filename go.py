"""Some first steps towards the GO bot

GO_game is a class that represents a match of GO, including the current board,
a history of all boards and moves, and checking of validity of moves.
It also evaluates the area/territory, according to japanese or chinese rules.

I did my best on the docstrings, and I used some type-annotation from time to
time! Python supports that since 3.?, but it's just used by linters and it
does not affect the code in any way during the runtime (afaik).
"""
import numpy as np
import random as rn
from typing import Tuple, List
import logging
logging.basicConfig(
    # filename='logs/go_game.log',
    level=logging.INFO,
    # format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    format='%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


"""Just to adjust the internal representation of color at a single location,
instead of all over the code ;) Just in case. Maybe something else as -1 and 1
could be interesting, see the tick tack toe example"""
WHITE = -1
BLACK = 1


class GO_Error(Exception):
    pass


class InvalidMove_Error(GO_Error):
    pass


class Game():
    """Class that contains a game of go

    Saves the board as a numpy matrix, using 1 for white and -1 for black.
    The repr is actually quite cool! Try `g=GO_game();print(g)` :)
    Saves the played history
    Checks if groups die and removes them

    Not yet done:
    --------
    Count the points at the end and decide who won
    Rules about not allowing to retake a certain piece
    Checking if someone kills himself (not too bad, noone should do that)
    """
    def __init__(self, setup={}, show_each_turn=False):
        # Dict returned by sgf has values as lists
        setup = {k: v[0] for k, v in setup.items()}

        self.komi = float(setup.get('KM', 7))
        self.size = int(setup.get('SZ', 9))
        self.rules = setup.get('RU', 'japanese').lower()
        self.result = setup.get('RE')
        self.time = int(setup.get('TM', 0))
        self.show_each_turn = show_each_turn
        self.board = np.matrix([[0]*self.size]*self.size)
        self.white_rank = int(setup.get('WR', 0))
        self.black_rank = int(setup.get('BR', 0))

        self.play_history = []
        self.board_history = set()
        if 'AB' in setup.keys():
            for loc in setup['AB']:
                self.board[self._str2index(loc)] = BLACK
        if 'AW' in setup.keys():
            for loc in setup['AW']:
                self.board[self._str2index(loc)] = WHITE
        self.black_player_captured = 0
        self.white_player_captured = 0

    def w(self, loc: str, show_board=False):
        """White plays"""
        self._play(loc, 'w')
        if show_board or self.show_each_turn:
            print(self)

    def b(self, loc: str, show_board=False):
        """Black plays"""
        self._play(loc, 'b')
        if show_board or self.show_each_turn:
            print(self)

    def _play(self, move: str, player: {'w', 'b'}, testing=False):
        """Play at a location, and check for all the rules

        Parameters
        ----------
        move: str
            Location using the sgf notation, e.g. 'ef'
        player: {'w', 'b'}
            Char designation the player
        testing: bool
            I Introduced this so that `_play` is also a function that tests
            for validity. If we're in "testing-mode", just revert the state
            Also we assume noone would call this using `''` (pass)
        """
        # 0. Save board to be able to revert it after
        _starting_board = self.board.copy()

        # 1. Check if the player is passing and if this ends the game
        if move == '':
            self.play_history.append(player+':'+move)
            # Append the move to the move_history
            if (len(self.play_history) > 2 and
                    self.play_history[-2].split(':')[1] == ''):
                logger.info('Game finished!')
                return self.evaluate_points()   # Game ended!
            return                              # There is nothing to do

        # 2. Play the stone
        # Transform the 'ef' type location to a matrix index (row, col)
        loc = self._str2index(move)
        # Use the numerical player representation (-1 or 1 so far)
        color = WHITE if player == 'w' else BLACK
        # Check if the location is empty
        if self.board[loc] != 0:
            raise InvalidMove_Error(
                'There is already a stone at that location')
        # "Play the stone" at the location
        self.board[loc] = color

        # 3. Check if this kills a group of stones and remove them
        # How this is done:
        #   1. Get all neighbors
        #   1b. Only look at those with the enemy color
        #   2. For each of them, get the respective chain
        #   3. For each neighbor of each stone in the chain, check if it is 0
        #   4. If one of them (or more) is 0 they live, else they die
        neighbors = self._get_adjacent_coords(loc)
        groups = []
        for n in neighbors:
            if self.board[n] == -color:
                groups.append(self._get_chain(n))
        for g in groups:
            if self._check_dead(g):
                # Capture the stones!
                if color == BLACK:
                    self.black_player_captured += len(g)
                if color == WHITE:
                    self.white_player_captured += len(g)
                for c in g:
                    self.board[c] = 0

        # 4. Validity Checks
        # 4a. No suicides!
        own_chain = self._get_chain(loc)
        if self._check_dead(own_chain):
            # This play is actually a suicide! Revert changes and raise Error
            self.board[loc] = color
            raise InvalidMove_Error('No suicides')
        # 4b. No board state twice! (Depends on rules, yes, TODO)
        if (len(self.board_history) > 0 and
                self._board_to_number(self.board) in self.board_history):
            self.board[loc] = color
            raise InvalidMove_Error(
                'Same constellation can only appear once')

        # 5. Everything is valid :)
        # If we were testing just revert everything, else append to history
        if not testing:
            # Append move and board to histories
            self.board_history.add(self._board_to_number(self.board.copy()))
            self.play_history.append(player+':'+move)
        else:
            # Revert changes
            self.board = _starting_board

    def _str2index(self, loc: str) -> Tuple[int, int]:
        """Convert the sgf location format to a usable matrix index

        Examples
        --------
        >>> g = GO_game(); g._str2index('ef')
        (5, 4)
        """
        col = self._chr2ord(loc[0])
        row = self._chr2ord(loc[1])
        return (row, col)

    def _index2str(self, index: Tuple[int, int]) -> str:
        """Convert the sgf location format to a usable matrix index

        Examples
        --------
        >>> g = GO_game(); g._str2index('ef')
        (5, 4)
        """
        col = self._ord2chr(index[1])
        row = self._ord2chr(index[0])
        return col+row

    def __repr__(self):
        """String representation of the board!

        Just a simple ascii output, quite cool but the code is a bit messy"""
        b = self.board.copy()
        # You might wonder why I do the following, but its so that numpy
        # formats the str representation using a single space
        b[b == BLACK] = 2
        b[b == WHITE] = 3

        matrix_repr = str(b)
        matrix_repr = matrix_repr.replace('2', 'B')
        matrix_repr = matrix_repr.replace('3', 'W')
        matrix_repr = matrix_repr.replace('0', '·')
        matrix_repr = matrix_repr.replace('[[', ' [')
        matrix_repr = matrix_repr.replace(']]', ']')
        col_index = '   a b c d e f g h i'
        row_index = 'a,b,c,d,e,f,g,h,i'.split(',')
        board_repr = ''
        for i in zip(row_index, matrix_repr.splitlines()):
            board_repr += i[0]+i[1]+'\n'
        board_repr = col_index+'\n'+board_repr
        return board_repr

    def _chr2ord(self, c: str) -> int:
        """Small helper function - letter to number

        Examples
        --------
        >>> g = GO_game(); g._chr2ord('f')
        5
        """
        idx = ord(c) - ord('a')
        if idx < 0 or idx >= self.size:
            raise InvalidMove_Error(c + '=' + str(idx) + ' is an invalid row/column index, board size is '
                                    + str(self.size))
        return idx

    def _ord2chr(self, o: int) -> str:
        """Small helper function - number to letter

        Examples
        --------
        >>> g = GO_game(); g._ord2chr(5)
        'f'
        """
        return chr(o + ord('a'))

    def _get_adjacent_coords(self, loc: Tuple[int, int]):
        neighbors = []
        if loc[0] > 0:
            neighbors.append((loc[0]-1, loc[1]))
        if loc[0] < 8:
            neighbors.append((loc[0]+1, loc[1]))
        if loc[1] > 0:
            neighbors.append((loc[0], loc[1]-1))
        if loc[1] < 8:
            neighbors.append((loc[0], loc[1]+1))
        return neighbors

    def _get_chain(self, loc: Tuple[int, int]) -> List[Tuple[int, int]]:
        player = self.board[loc]
        # Check if neighbors of same player
        to_check = [loc]
        group = []
        while len(to_check) > 0:
            current = to_check.pop()
            neighbors = self._get_adjacent_coords(current)
            for n in neighbors:
                if (self.board[n] == player and
                        n not in group and n not in to_check):
                    to_check.append(n)
            group.append(current)
        return group

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

    def _check_dead(self, group: List[Tuple[int, int]]) -> bool:
        """Check if a group is dead

        Currently done by getting all the neighbors, and checking if any
        of them is 0.
        """
        total_neighbors = []
        for loc in group:
            total_neighbors += self._get_adjacent_coords(loc)
        for n in total_neighbors:
            if self.board[n] == 0:
                return False
        return True

    def evaluate_points(self):
        """Count the area/territory, subtract captured, komi"""
        white_score = 0
        black_score = 0

        empty_locations = np.argwhere(self.board == 0)
        # Numpy is weird. Without tuples a lot of things dont work :/
        empty_locations = [(l[0], l[1]) for l in empty_locations]
        for stone in empty_locations:
            chain = self._get_chain(stone)
            black_neighbor = False
            white_neighbor = False
            for s in chain:
                for n in self._get_adjacent_coords(s):
                    if self.board[n] == BLACK:
                        black_neighbor = True
                    if self.board[n] == WHITE:
                        white_neighbor = True
                if black_neighbor and white_neighbor:
                    # Neutral territory
                    pass
                elif black_neighbor and not white_neighbor:
                    black_score += len(chain)
                elif not black_neighbor and white_neighbor:
                    white_score += len(chain)

        logger.debug(f'White territory: {white_score}')
        logger.debug(f'Black territory: {black_score}')
        logger.debug(f'White captured: {self.white_player_captured}')
        logger.debug(f'Black captured: {self.black_player_captured}')

        # For japanese rules
        if self.rules.lower().startswith('j'):
            black_score -= self.white_player_captured
            white_score -= self.black_player_captured
            white_score += self.komi

            print(f'White: {white_score}\nBlack: {black_score}')
        if self.rules.lower().startswith('c'):
            black_locations = np.argwhere(self.board == BLACK)
            white_locations = np.argwhere(self.board == WHITE)
            black_score += len(black_locations)
            white_score += len(white_locations)
            white_score += self.komi

            logger.debug(f'White: {white_score}')
            logger.debug(f'Black: {black_score}')
        if black_score == white_score:
            print('Same score: Draw!')
            return
        winner = 'Black' if black_score > white_score else 'White'
        print(f'{winner} won by {abs(black_score - white_score)} points!')
        if self.result:
            logger.debug(f'Result according to the sgf: {self.result}')

    def generate_move(self, apply=True, show_board=True):
        """Generate a valid move - ANY valid move

        This is just a proof of concept, no need for it to be smart right now.
        """
        # 0. Which player am I?
        if len(self.play_history) > 0:
            own_color = 'w' if self.play_history[-1].startswith('b') else 'b'
        else:
            own_color = 'b'

        # 1. Get free locations on the board
        empty_locations = np.argwhere(self.board == 0)
        empty_locations = [(l[0], l[1]) for l in empty_locations]

        valid_moves = []
        for location in empty_locations:
            move = self._index2str(location)
            try:
                self._play(move, own_color, testing=True)
                valid_moves.append(move)
            except InvalidMove_Error as e:
                pass

        if len(valid_moves) == 0:
            move = ''
        else:
            move = rn.choice(valid_moves)

        if apply:
            self._play(move, own_color)
        if show_board:
            print(self)

        return move

    def _board_to_number(self, board):
        """Basically just create a unique representation for a board

        I do this because performence gets bad once the board history is
        large
        """
        number = 0
        i = 0
        for entry in np.nditer(board):
            if entry == WHITE:
                number += 1 * 10**i
            elif entry == BLACK:
                number += 2 * 10**i
            else:
                number += 3 * 10**i
            i += 1
        return number


if __name__ == '__main__':
    import doctest
    # doctest.testmod(extraglobs={'g': GO_game()})
    doctest.testmod()
