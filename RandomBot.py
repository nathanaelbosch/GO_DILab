import random


class RandomBot:

    def __init__(self, game):
        self.game = game

    def genmove(self, color):
        return random.choice(self.game.get_playable_locations(color))
