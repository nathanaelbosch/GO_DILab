from src import Game, ConsoleView, SimplePlottingView, GUIView, HumanConsolePlayer, RandomBotPlayer, GameController


def main():
    game = Game()
    view = ConsoleView(game)
    view2 = GUIView(game)
    player1 = HumanConsolePlayer("PersonA", "w", game)
    player2 = RandomBotPlayer("RandomBot", "b", game)
    game_controller = GameController(game, view, player1, player2,view2)
    game_controller.start()
    # view.open(game_controller)
    

if __name__ == '__main__':
    main()
