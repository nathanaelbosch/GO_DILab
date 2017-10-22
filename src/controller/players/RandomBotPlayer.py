from src import Player
import random as rn


class RandomBotPlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)

    def make_move(self):
        playable_locations = self.game.get_playable_locations(self.color)
        random_choice = rn.choice(playable_locations)
        self.game.play(random_choice, self.color)
