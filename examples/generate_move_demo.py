"""generate_move Example - Quite cool to see!

Yes, I know, that's not how one should import. I need to learn that :/"""
import sys
from src import Game

sys.path.append('.')


def main():
    game = Game()
    last_move = 'something'
    while True:
        move = game.generate_move()
        print(move)
        if last_move == '' and move == '':
            # Game Finished!
            break
        last_move = move
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
