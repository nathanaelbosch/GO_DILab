from src import Game, ConsoleView, HumanConsolePlayer, RandomBotPlayer, GameController


def main():
    game = Game()
    view = ConsoleView(game)
    player1 = HumanConsolePlayer("PersonA", "w", game)
    player2 = HumanConsolePlayer("PersonB", "b", game)
    #player2 = RandomBotPlayer("RandomBot", "b", game)
    game_controller = GameController(game, view, player1, player2)
    game_controller.start()


if __name__ == '__main__':
    main()
