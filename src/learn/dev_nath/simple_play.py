"""Play against the random bot as a human!"""
import sys
from src.play.model.Game import Game
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot
from src.learn.dev_nath.SimplestNNBot import SimplestNNBot
from src.learn.dev_nath_win_prediction.WinPredictionBot import WinPredictionBot

from pprint import pprint

sys.path.append('.')


def main():
    player1 = RandomBot()
    player2 = WinPredictionBot()

    num_games = 50
    results = []
    for i in range(num_games):
        current, other = player1, player2
        current_col, other_col = 'b', 'w'
        last_move = None
        game = Game({'SZ': 9})

        while True:
            print(game)

            move = current.genmove(current_col, game)
            # print(move.to_gtp(9))
            result = game.play(move, player=current_col)
            if (last_move is not None and
                    last_move.is_pass and
                    move.is_pass):
                print(current_col, current)
                print(other_col, other)
                results.append(result)
                break

            current, other = other, current
            current_col, other_col = other_col, current_col
            last_move = move

    pprint(results)
    print('Playerd {} games'.format(num_games))
    print('Black ({}) won {} times'.format(
        player1, len([r for r in results if r.startswith('B')])))
    print('White ({}) won {} times'.format(
        player2, len([r for r in results if r.startswith('W')])))


if __name__ == '__main__':
    main()
