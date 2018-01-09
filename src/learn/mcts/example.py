"""Simple Example for MCTS
https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/
"""
import datetime
from random import choice
import random
from math import log, sqrt
import copy

import numpy as np
import keras

from src import Utils
from src.play.model.Game import Game, BLACK, WHITE, EMPTY
from src.play.model.Board import Board as _Board

from src.learn.bots._21.bot import Bot_21
from src.learn.bots._22.bot import Bot_22


class Board(Game):
    @staticmethod
    def to_tuple(board):
        b = tuple(tuple(i for i in row) for row in board.tolist())
        return b

    @staticmethod
    def from_tuple(board_tuple):
        b = [[i for i in j] for j in board_tuple]
        return _Board(b)

    def start(self):
        """Return a representation of the starting state of the game.

        State will be defined as (current board, current player)
        Board is an instance of the src.play.model.Board Board() class
        Player is 1 if BLACK or 2 if WHITE
        """
        b = _Board([[0]*self.size]*self.size)
        return (self.to_tuple(b), 1)

    def current_player(self, state):
        # Takes the game state and returns the current player's
        # number.
        return state[1]

    def next_state(self, state, play):
        # Takes the game state, and the move to be applied.
        # Returns the new game state.
        board, player = state
        board = copy.deepcopy(board)
        board = _Board(board)
        player_val = BLACK if player == 1 else WHITE
        next_player = 2 if player == 1 else 1

        if play.is_pass:
            return (self.to_tuple(board), next_player)
        else:
            board.place_stone_and_capture_if_applicable_default_values(
                play.to_matrix_location(), player_val)
            return (self.to_tuple(board), next_player)

    def legal_plays(self, state_history):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        state = state_history[-1]
        color = 'b' if self.current_player(state) == 1 else 'w'
        board = self.from_tuple(state[0])
        self.board = board
        return self.get_playable_locations(color)

    def winner(self, state_history):
        # Takes a sequence of game states representing the full
        # game history.  If the game is now won, return the player
        # number.  If the game is still ongoing, return zero.  If
        # the game is tied, return a different distinct value, e.g. -1.

        if len(state_history) < 3:
            return 0
        current_board = state_history[-1][0]
        previous_board = state_history[-2][0]
        preprevious_board = state_history[-3][0]
        if current_board == previous_board == preprevious_board:
            # both players passed!
            # print('current_board')
            # print(Board.from_tuple(current_board))
            # print('previous_board')
            # print(Board.from_tuple(previous_board))
            # print('preprevious_board')
            # print(Board.from_tuple(preprevious_board))
            ended = True
        else:
            ended = False

        if not ended:
            return 0
        else:
            self.board = self.from_tuple(current_board)
            result_string = self.evaluate_points()
            # print(result_string)
            if result_string.lower().startswith('b'):
                # Black won!
                return 1
            elif result_string.lower().startswith('w'):
                # White won!
                return 2
            else:
                # Noone won!
                return -1


class MonteCarlo(object):
    def __init__(self, board, **kwargs):
        # Takes an instance of a Board and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self.board = board
        self.states = []

        seconds = kwargs.get('time', 10)
        self.verbose = kwargs.get('verbose', True)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 20)

        self.wins = {}
        self.plays = {}

        self.C = kwargs.get('C', 1.4)

        Utils.set_keras_backend('theano')

        # Value Network
        _arch = 'src/learn/bots/_21/model_architecture.json'
        _weights = 'src/learn/bots/_21/model_weights.h5'
        with open(_arch, 'r') as f:
            self.value_net = keras.models.model_from_json(f.read())
        self.value_net.load_weights(_weights)

        # Policy Network
        _arch = 'src/learn/bots/_22/model_architecture.json'
        _weights = 'src/learn/bots/_22/model_weights.h5'
        with open(_arch, 'r') as f:
            self.policy_net = keras.models.model_from_json(f.read())
        self.policy_net.load_weights(_weights)

    def update(self, state):
        # Takes a game state, and appends it to the history.
        self.states.append(state)

    def estimate_outcome(self, state):
        # 1. state to board
        board = state[0]
        board = Board.from_tuple(board)
        flat_board = np.array(board).flatten()
        flat_board = flat_board.reshape(1, len(flat_board))
        color = WHITE if state[1] == 2 else BLACK

        # 2. board to input
        inp = Bot_21.generate_nn_input(flat_board, color)
        # print(inp)

        # 3. input to predicted outcome
        pred = self.value_net.predict(inp)[0]
        if np.argmax(pred) == 0:
            winner = 1 if color == BLACK else 2
        else:
            winner = 2 if color == BLACK else 1
        return winner

    @staticmethod
    def _weighted_choice(choices):
        total = sum(w for c, w in choices)
        r = random.uniform(0, total)
        upto = 0
        for c, w in choices:
            if upto + w >= r:
                return c
            upto += w
        assert False, "Shouldn't get here"

    def policy_weighted_choice(self, state, choices):
        board, player = state
        flat_board = np.array(board).flatten()
        flat_board = flat_board.reshape(1, len(flat_board))
        color = WHITE if player == 2 else BLACK
        inp = Bot_22.generate_nn_input(flat_board, color)
        pred = self.policy_net.predict(inp)[0]
        # print(pred)
        # print(pred.shape)
        weights = []
        for move, state in choices:
            if move.is_pass:
                # print(move, 'has probability', pred[81])
                weights.append(pred[81])
            else:
                # print('Move to flat idx:', move.to_flat_idx())
                # print(move, 'has probability', pred[move.to_flat_idx()])
                weights.append(pred[move.to_flat_idx()])
        choices_and_weights = list(zip(choices, weights))
        return self._weighted_choice(choices_and_weights)

    def get_play(self):
        # Causes the AI to calculate the best move from the
        # current game state and return it.
        self.max_depth = 0
        state = self.states[-1]
        player = self.board.current_player(state)
        legal = self.board.legal_plays(self.states[:])

        # Bail out early if there is no real choice to be made.
        if not legal:
            return
        if len(legal) == 1:
            return legal[0]

        games = 0
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation()
            games += 1

        moves_states = [(p, self.board.next_state(state, p)) for p in legal]

        # Display the number of calls of `run_simulation` and the
        # time elapsed.
        if self.verbose:
            print('Simulations run:', games, '\tTime elapsed:',
                  datetime.datetime.utcnow() - begin)

        # Pick the move with the highest percentage of wins.
        percent_wins, move = max((
            (self.wins.get((player, S), 0) /
             self.plays.get((player, S), 1),
             p)
            for p, S in moves_states),
            key=lambda x: x[0]
        )

        # Display the stats for each possible play.
        if self.verbose:
            for x in sorted(
                ((100 * self.wins.get((player, S), 0) /
                  self.plays.get((player, S), 1),
                  self.wins.get((player, S), 0),
                  self.plays.get((player, S), 0), p)
                 for p, S in moves_states),
                reverse=True,
                key=lambda x: x[0]
            ):
                print("{3}: {0:.2f}% ({1} / {2})".format(*x))

            print("Maximum depth searched:", self.max_depth)

        return move

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        plays, wins = self.plays, self.wins

        visited_states = set()
        states_copy = self.states[:]
        state = states_copy[-1]
        player = self.board.current_player(state)

        expand = True
        for t in range(1, self.max_moves + 1):
            legal = self.board.legal_plays(states_copy)
            moves_states = [(p, self.board.next_state(state, p)) for p in legal]
            # if all(plays.get((player, S)) for p, S in moves_states):
            #     # If we have stats on all of the legal moves here, use them.
            #     log_total = log(
            #         sum(plays[(player, S)] for p, S in moves_states))
            #     value, move, state = max(
            #         (((wins[(player, S)] / plays[(player, S)]) +
            #           self.C * sqrt(log_total / plays[(player, S)]), p, S)
            #          for p, S in moves_states),
            #         key=lambda x: x[0]
            #     )
            # else:
            #     # Otherwise, just make an arbitrary decision.
            #     move, state = choice(moves_states)

            # Instead of `move, state = choice(move_states)`
            # Do a sampling according to the policy network
            # print(self.policy_weighted_choice(state, moves_states))
            move, state = self.policy_weighted_choice(state, moves_states)

            states_copy.append(state)

            # `player` here and below refers to the player
            # who moved into that particular state.
            if expand and (player, state) not in plays:
                expand = False
                plays[(player, state)] = 0
                wins[(player, state)] = 0
                if t > self.max_depth:
                    self.max_depth = t

            visited_states.add((player, state))

            player = self.board.current_player(state)
            winner = self.board.winner(states_copy)
            if winner:
                # print('Found a winner!')
                # print(winner)
                # print(player)
                # print(Board.from_tuple(states_copy[-1][0]))
                # print(states_copy)
                break

        if not winner:
            winner = self.estimate_outcome(states_copy[-1])

        for player, state in visited_states:
            if (player, state) not in plays:
                continue
            plays[(player, state)] += 1
            if player == winner:
                wins[(player, state)] += 1


def main():
    board = Board()
    # board = Board({'SZ': [3]})
    mc = MonteCarlo(
        board,
        max_moves=2,
        time=30,
    )
    mc.update(board.start())
    print(mc.get_play())


if __name__ == '__main__':
    main()
