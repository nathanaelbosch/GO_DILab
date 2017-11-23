import logging
import sys
from os.path import dirname, abspath

from src.play.view.PygameView import PygameView

project_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(project_dir)

from src.play.controller.GTPcontroller import GTPcontroller

from src.play.controller.bots.HumanConsole import HumanConsole
from src.play.controller.bots.HumanGui import HumanGui
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot
from src.learn.dev_nath.SimplestNNBot import SimplestNNBot


def main():
    # if you don't want logfiles: change the logging-level to something more critical than INFO (e.g. WARNING)

    controller = GTPcontroller(
        RandomBot.__name__,
        RandomGroupingBot.__name__,
        logging.INFO,
    )
    view = PygameView(controller)
    controller.start()
    view.open()


if __name__ == '__main__':
    main()
