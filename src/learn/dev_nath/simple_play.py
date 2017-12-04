"""Play against the random bot as a human!"""
import sys
from src.play.model.Game import Game
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot
from src.learn.dev_nath.SimplestNNBot import SimplestNNBot
from src.learn.dev_nath_win_prediction.WinPredictionBot import WinPredictionBot

from pprint import pprint

sys.path.append('.')


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

    PLAYER2 = RandomBot()
    PLAYER1 = WinPredictionBot()
    VERBOSE = False
    NUM_GAMES = 100

    # import multiprocessing
    # pool = multiprocessing.Pool()
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
