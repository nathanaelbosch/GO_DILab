from src.play import Player
import random as rn
import time


class RandomBotPlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)

    def get_move(self):
        playable_locations = self.game.get_playable_locations(self.color)
        random_choice = rn.choice(playable_locations)
        # time.sleep(0.3)
        return random_choice
