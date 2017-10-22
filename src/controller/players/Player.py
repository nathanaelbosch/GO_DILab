from abc import ABC, abstractmethod


# abstract base class for all Players, can't be instantiated
class Player(ABC):

    def __init__(self, name, color, game):
        self.name = name
        self.color = color
        self.game = game

    # TODO
    # Its probably better style if this method returns Tuple[int, int]
    # And then ConverterController passes the location as coordinates
    # (not as string) to Game. str2index is prepared in Utils
    @abstractmethod
    def make_move(self):
        pass
