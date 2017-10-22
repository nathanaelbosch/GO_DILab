from abc import ABC, abstractmethod


# abstract base class for all Views, can't be instantiated
class View(ABC):

    def __init__(self, game):
        self.game = game

    @abstractmethod
    def update_view(self):
        pass
