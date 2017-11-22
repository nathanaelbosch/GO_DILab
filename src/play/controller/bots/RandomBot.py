import random

from src.play.model import Move


class RandomBot:

    @staticmethod
    def genmove(color, game) -> Move:
        return random.choice(game.get_playable_locations(color))
