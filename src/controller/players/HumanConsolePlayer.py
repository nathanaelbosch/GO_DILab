from src import Player
from src.utils.Utils import str2move


class HumanConsolePlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)

    def make_move(self):
        loc_str = input()
        print('Submit your desired location...')
        move = str2move(loc_str.lower(), self.game.size)
        self.game.play(move, self.color)
