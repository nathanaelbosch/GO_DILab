import logging
import sys
from os.path import dirname, abspath

project_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(project_dir)

from src.learn.simplest_move_prediction.SimplestNNBot import SimplestNNBot
from src.play.controller.GTPcontroller import GTPcontroller
from src.play.controller.bots.HumanConsole import HumanConsole
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot


def main():
    GTPcontroller(
        RandomBot.__name__,
        RandomGroupingBot.__name__,
        logging.INFO,  # anything more critical than INFO will cause no logfiles to be written
    ).run()


if __name__ == '__main__':
    main()
