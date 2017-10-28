import sys
from os.path import dirname, abspath
# set GO_DILab as PYTHONPATH, therewith the script can be run from anywhere
project_dir = dirname(dirname(dirname(abspath(__file__))))  # each dirname is one level up
sys.path.append(project_dir)
from src.play import Game, PygameGuiView, HumanConsolePlayer, HumanGuiPlayer, RandomBotPlayer, GameController


def main():
    
    game = Game()

    view = PygameGuiView(game)

    player1 = HumanGuiPlayer("Max", "b", game)
    player2 = RandomBotPlayer("Robo", "w", game)

    game_controller = GameController(game, view, player1, player2)
    game_controller.start()

    view.open(game_controller)


if __name__ == '__main__':
    main()
