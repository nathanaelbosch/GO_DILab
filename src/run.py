from src import Game, ConsoleView, PygameGuiView, HumanConsolePlayer, HumanGuiPlayer, RandomBotPlayer, GameController


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
