"""Some first steps towards the GO bot

Game is a class that represents a match of GO, including the current board,
a history of all boards and moves, and checking of validity of moves.
It also evaluates the area/territory, according to japanese or chinese rules.

I did my best on the docstrings, and I used some type-annotation from time to
time! Python supports that since 3.?, but it's just used by linters and it
does not affect the code in any way during the runtime (afaik).
"""
import copy
import logging

from src.play.model.Board import *

from src.play.model.Move import *
from src.play.model.errors import *


logging.basicConfig(
    # filename='logs/Game.log',
    level=logging.INFO,
    # format='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    format='%(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


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
        self.rules = setup.get('RU', 'chinese').lower()
        self.result = setup.get('RE')
        self.time = int(setup.get('TM', 0))
        self.show_each_turn = show_each_turn
        self.board = Board([[EMPTY]*self.size]*self.size)
        try:
            self.white_rank = int(setup.get('WR', 0))
            self.black_rank = int(setup.get('BR', 0))
        except ValueError as e:
            logger.debug('Value Error when reading rank from sgf')
            pass

        self.play_history = []
        self.board_history = set()
        if 'AB' in setup.keys():
            for loc in setup['AB']:
                move = Move.from_sgf(loc)
                self.board[move.to_matrix_location] = BLACK
        if 'AW' in setup.keys():
            for loc in setup['AW']:
                move = Move.from_sgf(loc)
                self.board[move.to_matrix_location] = WHITE
        self.black_player_captured = 0
        self.white_player_captured = 0

    def start(self):
        self.is_running = True

    def play(self, move: Move, player: {'w', 'b'},
             testing=False,
             checking=True):
        """Play a move!

        Also checks all the rules.
        Implemented by first playing on test_board. If everything holds
        then we apply the changes to self.board.
        """
        test_board = copy.deepcopy(self.board)
        # test_board = self.board.copy()
        # 1. Check if the player is passing and if this ends the game
        if move.is_pass:
            if not testing:
                self.play_history.append(player + ':' + str(move))
                # Append the move to the move_history
                if (len(self.play_history) > 2 and
                        self.play_history[-2].split(':')[1] == 'pass'):
                    logger.info('Game finished!')
                    self.is_running = False
                    return self.evaluate_points()  # Game ended!
            return  # There is nothing to do

        # 1b. First quick validity-check
        if checking:
            if (move.col + 1 > self.size or move.row + 1 > self.size or
                    move.col < 0 or move.row < 0):
                raise InvalidMove_Error(
                    'Location is not present on the board: '+str(move))

        # 2. Play the stone
        loc = move.to_matrix_location()
        # Use the numerical player representation (-1 or 1 so far)
        color = WHITE if player == 'w' else BLACK
        # Check if the location is empty
        if checking:
            if test_board[loc] != EMPTY:
                raise InvalidMove_Error(
                    'There is already a stone at location ' + move.to_gtp(self.size))
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
        black_player_captured, white_player_captured = 0, 0
        for n in neighbors:
            if test_board[n] == -color:
                groups.append(test_board.get_chain(n))
        for g in groups:
            if test_board.check_dead(g):
                # Capture the stones!
                if color == BLACK:
                    black_player_captured += len(g)
                if color == WHITE:
                    white_player_captured += len(g)
                for c in g:
                    test_board[c] = EMPTY

        # 4. Validity Checks
        # 4a. No suicides!
        if checking:
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
        # If we were testing we're done
        # If not then apply changes and append move and board to history
        if not testing:
            # Append move and board to histories
            self.board = test_board
            self.black_player_captured += black_player_captured
            self.white_player_captured += white_player_captured
            if checking:
                self.board_history.add(test_board.to_number())
                self.play_history.append(player + ':' + str(move))

    def __str__(self):
        """Game representation = Board representation"""
        return str(self.board)

    def evaluate_points(self):
        """Count the area/territory, subtract captured, komi"""
        black_territory = 0
        white_territory = 0

        # 1. Count territory
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
                black_territory += len(chain)
            elif not black_neighbor and white_neighbor:
                white_territory += len(chain)

        # 2. Count area
        black_locations = np.argwhere(self.board == BLACK)
        white_locations = np.argwhere(self.board == WHITE)
        black_area = black_territory + len(black_locations)
        white_area = white_territory + len(white_locations)

        logger.debug('Black territory: {}'.format(black_territory))
        logger.debug('White territory: {}'.format(white_territory))
        logger.debug('Black area: {}'.format(black_area))
        logger.debug('White area: {}'.format(white_area))
        logger.debug('Black captured: {}'.format(self.black_player_captured))
        logger.debug('White captured: {}'.format(self.white_player_captured))

        # For japanese rules
        if self.rules.lower().startswith('j'):
            black_score = black_territory - self.white_player_captured
            white_score = white_territory - self.black_player_captured
            white_score += self.komi

            logger.debug('Black: {}'.format(black_score))
            logger.debug('White: {}'.format(white_score))
        if self.rules.lower().startswith('c'):
            black_score = black_area
            white_score = white_area
            white_score += self.komi

            logger.debug('Black: {}'.format(black_score))
            logger.debug('White: {}'.format(white_score))
        if black_score == white_score:
            logger.info('Same score: Draw!')
            return 'Draw'
        winner = 'Black' if black_score > white_score else 'White'
        logger.info('{} won by {} points!'.format(
            winner, abs(black_score - white_score)))
        if self.result:
            logger.debug('Result according to the sgf: {}'.format(self.result))
        result_string = winner[0]+'+'+str(abs(black_score - white_score))
        return result_string

    def get_playable_locations(self, color) -> []:
        empty_locations = np.argwhere(self.board == 0)
        empty_locations = [(l[0], l[1]) for l in empty_locations]
        valid_moves = [Move(is_pass=True)]  # passing is always a valid move
        for location in empty_locations:
            move = Move.from_matrix_location(location)
            try:
                self.play(move, color, testing=True)
                valid_moves.append(move)
            except InvalidMove_Error as e:
                pass
        return valid_moves

    def get_invalid_locations(self, color) -> []:
        invalid_moves = []
        for location in np.argwhere(self.board == BLACK):
            invalid_moves.append(Move.from_matrix_location(location))
        for location in np.argwhere(self.board == WHITE):
            invalid_moves.append(Move.from_matrix_location(location))
        empty_locations = np.argwhere(self.board == EMPTY)
        empty_locations = [(l[0], l[1]) for l in empty_locations]
        for location in empty_locations:
            move = Move.from_matrix_location(location)
            try:
                self.play(move, color, testing=True)
            except InvalidMove_Error as e:
                invalid_moves.append(move)
        return invalid_moves


if __name__ == '__main__':
    import doctest
    # doctest.testmod(extraglobs={'g': Game()})
    doctest.testmod()
