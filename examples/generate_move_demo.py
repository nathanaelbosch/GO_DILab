"""generate_move Example - Quite cool to see!

Yes, I know, that's not how one should import. I need to learn that :/"""
import sys
import time

sys.path.append('.')
import go


def main():
    game = go.Game()
    while True:
        game.generate_move()
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
