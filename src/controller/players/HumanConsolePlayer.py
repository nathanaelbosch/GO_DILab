from src import Player
from src.utils.Utils import str2move


class HumanConsolePlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)

    def get_move(self):
        print('Submit your desired location...')
        loc_str = input()  # ensure user input is valid TODO
        move = str2move(loc_str.lower(), self.game.size)
        return move
