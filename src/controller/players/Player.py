from abc import ABC, abstractmethod


# abstract base class for all Players, can't be instantiated
class Player(ABC):

    def __init__(self, name, color, game):
        self.name = name
        self.color = color
        self.game = game

    @abstractmethod
    def make_move(self):
        pass
