"""Play agains the random bot as a human!"""
import sys
import time

sys.path.append('.')
from go import Game, InvalidMove_Error


game = Game({'SZ': 3})
while True:
    print(game)
    print('Your move:')
    location = input()
    try:
        game.b(location)
        print(game)
        # time.sleep(1)
        move = game.generate_move(apply=True, show_board=False)
        if move != '':
            print(f'Bot plays {move}')
        else:
            print('Bot passes')
    except InvalidMove_Error as e:
        print(' '.join(e.args))
