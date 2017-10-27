from abc import ABC, abstractmethod


# abstract base class for all Views, can't be instantiated
class View(ABC):

    def __init__(self, game):
        self.game = game
        self.is_ready = False

    @abstractmethod
    def open(self, game_controller):
        pass

    @abstractmethod
    def show_player_turn_start(self, name):
        pass

    @abstractmethod
    def show_player_turn_end(self, name):
        pass

    @abstractmethod
    def show_error(self, msg):
        pass
