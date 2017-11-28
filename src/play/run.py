import logging
import sys
import os.path
from os.path import dirname, abspath
import argparse


project_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(project_dir)

from src.play.controller.GTPcontroller import GTPcontroller
from src.play.view.PygameView import PygameView
from src.play.controller.bots.HumanConsole import HumanConsole
from src.play.controller.bots.HumanGui import HumanGui
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot
from src.learn.dev_nath.SimplestNNBot import SimplestNNBot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='No GUI')
    parser.add_argument(
        '-p1', '--player1',
        help=('Player 1 - black - options: "human", '+
              '"random", "random_grouping" or "dev_nn_nath"'),
        default='random')
    parser.add_argument(
        '-p2', '--player2',
        help=('Player 2 - white - options: "human", '+
              '"random", "random_grouping" or "dev_nn_nath"'),
        default='random')
    parser.add_argument(
        '-r', '--replay',
        help='replays the game in .sgf format that is being passed as path')
    parser.add_argument(
        '-s', '--sleep',
        help='time in seconds to sleep at the end of each turn',
        default='0.5')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.replay:
        sgf_path = args.replay
        if not os.path.isfile(sgf_path):
            print('invalid path to .sgf file: ' + sgf_path)
            sys.exit(1)
        # TODO

    if args.no_gui:
        if args.player1 == 'human':
            args.player1 = 'human_console'
        if args.player2 == 'human':
            args.player2 = 'human_console'

    player_types = {
        'human': HumanGui,
        'human_console': HumanConsole,
        'random': RandomBot,
        'random_grouping': RandomGroupingBot,
        'dev_nn_nath': SimplestNNBot,
    }
    player1type = player_types[args.player1].__name__
    player2type = player_types[args.player2].__name__

    # if you don't want logfiles: change the logging-level to something
    # more critical than INFO (e.g. WARNING)
    controller = GTPcontroller(
        player1type, player2type, logging.INFO, float(args.sleep))
    controller.start()
    if not args.no_gui:
        PygameView(controller).open()


if __name__ == '__main__':
    main()
