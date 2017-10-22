from abc import ABC, abstractmethod


# abstract base class for all Players, can't be instantiated
class Player(ABC):

    def __init__(self, name, game):
        self.name = name
        self.game = game

    @abstractmethod
    def make_move(self):
        pass
