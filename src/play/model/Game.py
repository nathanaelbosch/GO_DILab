"""Some first steps towards the GO bot

Game is a class that represents a match of GO, including the current board,
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
import copy

"""Just to adjust the internal representation of color at a single location,
instead of all over the code ;) Just in case. Maybe something else as -1 and 1
could be interesting, see the tick tack toe example"""
WHITE = -1
BLACK = 1
EMPTY = 0

from src.play.utils.Move import Move
from src.play.model.Board import Board

logging.basicConfig(
    # filename='logs/Game.log',
    level=logging.INFO,
    # format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    format='%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GO_Error(Exception):
    pass


class InvalidMove_Error(GO_Error):
    pass


class Game:
    """Class that contains a game of go

    Saves the board as a numpy matrix, using 1 for white and -1 for black.
    The repr is actually quite cool! Try `g=Game();print(g)` :)
    Saves the played history
    Checks if groups die and removes them

    Not yet done:
    --------
    Count the points at the end and decide who won
    Rules about not allowing to retake a certain piece
    Checking if someone kills himself (not too bad, noone should do that)
    """
    def __init__(self, setup={}, show_each_turn=False):
        self.is_running = False
        # Dict returned by sgf has values as lists
        setup = {k: (v[0] if isinstance(v, list) else v)
                 for k, v in setup.items()}

        self.komi = float(setup.get('KM', 7))
        self.size = int(setup.get('SZ', 9))
        self.rules = setup.get('RU', 'japanese').lower()
        self.result = setup.get('RE')
        self.time = int(setup.get('TM', 0))
        self.show_each_turn = show_each_turn
        self.board = Board([[EMPTY]*self.size]*self.size)
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

    def start(self):
        self.is_running = True

    # deprecated
    def w(self, loc: str, show_board=False):
        """White plays"""
        self.play_str(loc, 'w')
        if show_board or self.show_each_turn:
            print(self)

    # deprecated
    def b(self, loc: str, show_board=False):
        """Black plays"""
        self.play_str(loc, 'b')
        if show_board or self.show_each_turn:
            print(self)

    def play(self, move: Move, player: {'w', 'b'}, testing=False):
        """Play a move!

        Also checks all the rules.
        Implemented by first playing on test_board. If everything holds
        then we apply the changes to self.board.
        """
        test_board = copy.deepcopy(self.board)
        # test_board = self.board.copy()
        # 1. Check if the player is passing and if this ends the game
        if move.is_pass:
            self.play_history.append(player + ':' + str(move))
            # Append the move to the move_history
            if (len(self.play_history) > 2 and
                    self.play_history[-2].split(':')[1] == 'pass'):
                logger.info('Game finished!')
                self.is_running = False
                return self.evaluate_points()  # Game ended!
            return  # There is nothing to do

        # 2. Play the stone
        loc = move.get_loc_as_matrix_coords()
        # Use the numerical player representation (-1 or 1 so far)
        color = WHITE if player == 'w' else BLACK
        # Check if the location is empty
        if test_board[loc] != EMPTY:
            raise InvalidMove_Error(
                'There is already a stone at that location')
        # "Play the stone" at the location
        test_board[loc] = color

        # 3. Check if this kills a group of stones and remove them
        # How this is done:
        #   1. Get all neighbors
        #   1b. Only look at those with the enemy color
        #   2. For each of them, get the respective chain
        #   3. For each neighbor of each stone in the chain, check if it is 0
        #   4. If one of them (or more) is 0 they live, else they die
        neighbors = test_board.get_adjacent_coords(loc)
        groups = []
        for n in neighbors:
            if test_board[n] == -color:
                groups.append(test_board.get_chain(n))
        for g in groups:
            if test_board.check_dead(g):
                # Capture the stones!
                if color == BLACK:
                    self.black_player_captured += len(g)
                if color == WHITE:
                    self.white_player_captured += len(g)
                for c in g:
                    test_board[c] = EMPTY

        # 4. Validity Checks
        # 4a. No suicides!
        own_chain = test_board.get_chain(loc)
        if test_board.check_dead(own_chain):
            # This play is actually a suicide! Revert changes and raise Error
            test_board[loc] = EMPTY
            raise InvalidMove_Error('No suicides')
        # 4b. No board state twice! (Depends on rules, yes, TODO)
        if (len(self.board_history) > 0 and
                test_board.to_number() in self.board_history):
            test_board[loc] = EMPTY
            raise InvalidMove_Error(
                'Same constellation can only appear once')

        # 5. Everything is valid :)
        # If we were testing just revert everything, else append to history
        if not testing:
            # Append move and board to histories
            self.board = test_board
            self.board_history.add(test_board.to_number())
            self.play_history.append(player + ':' + str(move))

    def _str2index(self, loc: str) -> Tuple[int, int]:
        """Convert the sgf location format to a usable matrix index

        Examples
        --------i
        >>> g = Game(); g._str2index('ef')
        (5, 4)
        """
        col = self._chr2ord(loc[0])
        row = self._chr2ord(loc[1])
        return (row, col)

    def _index2str(self, index: Tuple[int, int]) -> str:
        """Convert the sgf location format to a usable matrix index

        Examples
        --------
        >>> g = Game(); g._str2index('ef')
        (5, 4)
        """
        col = self._ord2chr(index[1])
        row = self._ord2chr(index[0])
        return col+row

    def __str__(self):
        """Game representation = Board representation!"""
        return str(self.board)

    def _chr2ord(self, c: str) -> int:
        """Small helper function - letter to number

        Examples
        --------
        >>> g = Game(); g._chr2ord('f')
        5
        """
        idx = ord(c) - ord('a')
        if idx < 0 or idx >= self.size:
            raise InvalidMove_Error(
                c + '=' + str(idx) +
                ' is an invalid row/column index, board size is ' +
                str(self.size))
        return idx

    def _ord2chr(self, o: int) -> str:
        """Small helper function - number to letter

        Examples
        --------
        >>> g = Game(); g._ord2chr(5)
        'f'
        """
        return chr(o + ord('a'))

    def evaluate_points(self):
        """Count the area/territory, subtract captured, komi"""
        white_score = 0
        black_score = 0

        empty_locations = np.argwhere(self.board == 0)
        # Numpy is weird. Without tuples a lot of things dont work :/
        empty_locations = [(l[0], l[1]) for l in empty_locations]
        for stone in empty_locations:
            chain = self.board.get_chain(stone)
            black_neighbor = False
            white_neighbor = False
            for s in chain:
                for n in self.board.get_adjacent_coords(s):
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

    def get_playable_locations(self, color) -> []:
        empty_locations = np.argwhere(self.board == 0)
        empty_locations = [(l[0], l[1]) for l in empty_locations]
        valid_moves = [Move(is_pass=True)]  # passing is always a valid move
        for location in empty_locations:
            move = Move(location[0], location[1])
            try:
                self.play(move, color, testing=True)
                valid_moves.append(move)
            except InvalidMove_Error as e:
                pass
        return valid_moves

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

        # Passing is always a valid move
        valid_moves = ['']
        for location in empty_locations:
            move = self._index2str(location)
            try:
                self.play(move, own_color, testing=True)
                valid_moves.append(move)
            except InvalidMove_Error as e:
                pass

        if len(valid_moves) == 0:
            move = ''
        else:
            move = rn.choice(valid_moves)

        if apply:
            self.play(move, own_color)
        if show_board:
            print(self)

        return move


if __name__ == '__main__':
    import doctest
    # doctest.testmod(extraglobs={'g': Game()})
    doctest.testmod()
