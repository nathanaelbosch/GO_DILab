from src import Game, ConsoleView, SimplePlottingView, HumanConsolePlayer, RandomBotPlayer, GameController


def main():
    game = Game()
    player1 = RandomBotPlayer("RandomBot", "w", game)
    player2 = RandomBotPlayer("RandomBot", "b", game)
    # player1 = HumanConsolePlayer("PersonA", "w", self.game)
    # player2 = HumanConsolePlayer("PersonB", "b", self.game)

    view = SimplePlottingView(game)
    game_controller = GameController(game, view, player1, player2)
    view.open(game_controller)


if __name__ == '__main__':
    main()
