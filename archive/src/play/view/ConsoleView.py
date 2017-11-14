from archive.src.play.view.View import View


class ConsoleView(View):

    def __init__(self, game):
        View.__init__(self, game)

    def open(self, game_controller):
        pass

    def show_player_turn_start(self, name):
        print('It\'s player ' + name + '\'s turn')

    def show_player_turn_end(self, name):
        self.print_board()

    def print_board(self):
        """This is handled in the game.__str__()"""
        print(self.game)

    def show_error(self, msg):
        print(msg)
