from src import Player


class HumanGuiPlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)
        self.next_move = None

    def make_move(self):
        while self.next_move is None:
            pass
        self.game.play(self.next_move, self.color)
        self.next_move = None

    def receive_next_move_from_gui(self, move):
        self.next_move = move
