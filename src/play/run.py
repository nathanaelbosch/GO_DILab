import sys
import argparse
from os.path import dirname, abspath
# set GO_DILab as PYTHONPATH, therewith the script can be run from anywhere
# each dirname is one level up
# project_dir = dirname(dirname(dirname(abspath(__file__))))
# sys.path.append(project_dir)
from Game import Game
from src.play.controller.GameController import GameController
from src.play.controller.players.HumanConsolePlayer import HumanConsolePlayer
from src.play.controller.players.HumanGuiPlayer import HumanGuiPlayer
from src.play.controller.players.RandomBotPlayer import RandomBotPlayer
from src.play.view.ConsoleView import ConsoleView
from src.play.view.PygameGuiView import PygameGuiView


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='No GUI')
    parser.add_argument(
        '-p1', '--player1',
        help='Player 1 - black - options: "human" or "random"',
        default='human')
    parser.add_argument(
        '-p2', '--player2',
        help='Player 2 - white - options: "human" or "random"',
        default='random')
    return parser.parse_args()


def main():
    args = parse_args()
    game = Game()

    player_types = {
        'human_nogui': HumanConsolePlayer,
        'human': HumanGuiPlayer,
        'random': RandomBotPlayer
    }

    name1 = 'Max' if args.player1 == 'human' else 'Robo'
    name2 = 'Alice' if args.player2 == 'human' else 'Robo'

    if args.no_gui:
        view = ConsoleView(game)
        if args.player1 == 'human':
            args.player1 = 'human_nogui'
        if args.player2 == 'human':
            args.player2 = 'human_nogui'
    else:
        view = PygameGuiView(game)

    player1 = player_types[args.player1](name1, "b", game)
    player2 = player_types[args.player2](name2, "w", game)

    game_controller = GameController(game, view, player1, player2)
    game_controller.start()

    view.open(game_controller)


if __name__ == '__main__':
    main()
