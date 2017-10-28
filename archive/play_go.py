"""Play against the random bot as a human!"""
import sys
from src import Game
from src.model.Game import InvalidMove_Error

sys.path.append('.')

game = Game({'SZ': 5})
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
