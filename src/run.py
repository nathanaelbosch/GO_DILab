import sys
from os.path import dirname, abspath
# set GO_DILab as PYTHONPATH, therewith the script can be run from anywhere
sys.path.append(dirname(dirname(abspath(__file__))))
from src import Game, PygameGuiView, HumanConsolePlayer, HumanGuiPlayer, RandomBotPlayer, GameController


def main():
    
    game = Game()

    view = PygameGuiView(game)

    player1 = HumanGuiPlayer("PersonA", "b", game)
    player2 = RandomBotPlayer("RandomBot", "w", game)

    game_controller = GameController(game, view, player1, player2)
    game_controller.start()
    view.open(game_controller)


if __name__ == '__main__':
    main()
