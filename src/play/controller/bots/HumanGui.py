from src.play.model.Move import Move


class HumanGui:

    def __init__(self):
        self.move = None

    def genmove(self, color, game) -> Move:
        while self.move is None:
            pass
        move = self.move
        self.move = None
        return move
