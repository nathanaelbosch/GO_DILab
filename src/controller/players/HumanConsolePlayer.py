from src import Player


class HumanConsolePlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)

    def make_move(self):
        loc = input()
        self.game.play(loc, self.color)
