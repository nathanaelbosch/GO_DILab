from src import Game, ConsoleView, HumanConsolePlayer, RandomBotPlayer, GameController

game = Game()
view = ConsoleView(game)
player1 = HumanConsolePlayer("PersonA", "w", game)
player2 = RandomBotPlayer("RandomBot", "b", game)
gameController = GameController(game, view, player1, player2)
gameController.start()
