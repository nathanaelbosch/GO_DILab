from src import Game, ConsoleView, PygameGuiView, HumanConsolePlayer, HumanGuiPlayer, RandomBotPlayer, GameController
from src.utils.Utils import call_method_on_each


def main():
    
    game = Game()

    views = [
        ConsoleView(game),
        PygameGuiView(game)
    ]

    player1 = HumanGuiPlayer("PersonA", "b", game)
    player2 = RandomBotPlayer("RandomBot", "w", game)

    game_controller = GameController(game, views, player1, player2)
    game_controller.start()
    call_method_on_each(views, "open", game_controller)


if __name__ == '__main__':
    main()
