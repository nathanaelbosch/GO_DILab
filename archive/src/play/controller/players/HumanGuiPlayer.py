from archive.src.play.controller.players.Player import Player


class HumanGuiPlayer(Player):

    def __init__(self, name, color, game):
        Player.__init__(self, name, color, game)
        self.next_move = None

    def get_move(self):
        if self.next_move is None:
            return None
        move = self.next_move
        self.next_move = None
        return move

    def receive_next_move_from_gui(self, move):
        self.next_move = move
