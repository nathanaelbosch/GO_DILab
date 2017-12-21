"""Let two bots play a number of times and evaluate the results

The difference between this and `src/play/run.py` is that the focus
here is on quickly playing games, preferrably in parallel, while only
printing the result. In `src/play/run.py` we want to watch a game, or be
able to play one ourselves.
"""
import argparse

from src.play.model.Game import Game
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot
from src.learn.dev_nath.SimplestNNBot import SimplestNNBot
from src.learn.dev_nath_win_prediction.WinPredictionBot import WinPredictionBot
from src.learn.dev_ben.NNBot_ben1 import NNBot_ben1
from src.learn.dev_kar.LibertyNNBot import LibertyNNBot
from src.learn.dev_yu.MovePredictionBot import MovePredictionBot
from src.learn.bots._11.bot import Bot_11
from src.learn.bots._21.bot import Bot_21
from src.learn.bots._12.bot import Bot_12
from src.learn.bots._22.bot import Bot_22
from src.learn.bots._31.bot import Bot_31
from src.learn.bots._32.bot import Bot_32


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--number',
        help='Number of games to play',
        type=int,
        default=100)
    parser.add_argument(
        '-p1', '--player1',
        help=('Player 1 - black - options: "human", "random", ' +
              '"random_grouping", "dev_nn_nath", "win_prediction", ' +
              '"dev_nn_ben" or "dev_nn_yu"'),
        default='random')
    parser.add_argument(
        '-p2', '--player2',
        help=('Player 2 - white - options: "human", "random", ' +
              '"random_grouping", "dev_nn_nath", "win_prediction", ' +
              '"dev_nn_ben" or "dev_nn_yu"'),
        default='random')
    parser.add_argument(
        '-v', '--verbose',
        help='Print more',
        action='store_true')
    return parser.parse_args()


PLAYERS = {
    'random': RandomBot,
    'random_grouping': RandomGroupingBot,
    'dev_nn_nath': SimplestNNBot,
    'win_prediction': WinPredictionBot,
    'dev_nn_ben': NNBot_ben1,
    'dev_nn_kar': LibertyNNBot,
    'dev_nn_yu': MovePredictionBot,
    '11': Bot_11,
    '21': Bot_21,
    '12': Bot_12,
    '22': Bot_22,
    '31': Bot_31,
    '32': Bot_32,
}


def play(args):
    player1, player2, verbose = args
    current, other = player1, player2
    current_col, other_col = 'b', 'w'
    last_move = None
    game = Game({'SZ': 9})

    while True:
        if verbose:
            print(game)

        move = current.genmove(current_col, game)
        # print(move.to_gtp(9))
        result = game.play(move, player=current_col)
        if (last_move is not None and
                last_move.is_pass and
                move.is_pass):
            if verbose:
                print(current_col, current)
                print(other_col, other)
            return result

        current, other = other, current
        current_col, other_col = other_col, current_col
        last_move = move


def main():
    args = parse_args()
    PLAYER1 = PLAYERS[args.player1]()
    PLAYER2 = PLAYERS[args.player2]()
    VERBOSE = args.verbose
    NUM_GAMES = args.number

    print('Evaluating {} vs. {} over {} games'.format(
        PLAYER1, PLAYER2, NUM_GAMES))

    multicore = False
    if multicore:
        import multiprocessing
        pool = multiprocessing.Pool()
        results = list(pool.map(play,
            [(PLAYER1, PLAYER2, VERBOSE)]*int(NUM_GAMES/2) +
            [(PLAYER2, PLAYER1, VERBOSE)]*int(NUM_GAMES/2)))
    else:
        results = list(map(play,
            [(PLAYER1, PLAYER2, VERBOSE)]*int(NUM_GAMES/2) +
            [(PLAYER2, PLAYER1, VERBOSE)]*int(NUM_GAMES/2)))

    results1 = results[:int(NUM_GAMES/2)]
    results2 = results[int(NUM_GAMES/2):]
    print('Playerd {} games'.format(int(NUM_GAMES/2)*2))
    print('Black ({}) : White ({})\t\t\t{}:{}'.format(
        PLAYER1, PLAYER2,
        len([r for r in results1 if r.startswith('B')]),
        len([r for r in results1 if r.startswith('W')])))
    print('Black ({}) : White ({})\t\t\t{}:{}'.format(
        PLAYER2, PLAYER1,
        len([r for r in results2 if r.startswith('B')]),
        len([r for r in results2 if r.startswith('W')])))


if __name__ == '__main__':
    main()
