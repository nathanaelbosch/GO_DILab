import logging
import sys
from os.path import dirname, abspath

project_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(project_dir)

from src.play.controller.GTPcontroller import GTPcontroller
from src.play.controller.bots import RandomBot, RandomGroupingBot


def main():
    GTPcontroller(
        RandomBot.__name__,
        RandomGroupingBot.__name__,
        logging.INFO,  # anything more critical than INFO will cause no logfiles to be written
    ).run()


if __name__ == '__main__':
    main()
