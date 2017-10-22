from src import Game, ConsoleView, HumanConsolePlayer, GameController

game = Game()
view = ConsoleView(game)
player1 = HumanConsolePlayer("PersonA", game)
player2 = HumanConsolePlayer("PersonB", game)
gameController = GameController(game, view, player1, player2)
gameController.start()
