
class GameController:

    def __init__(self, game, view, player1, player2):
        self.game = game
        self.view = view
        self.player1 = player1
        self.player2 = player2
        self.current_player = None

    def next_player(self):
        # start with player1 if not otherwise defined
        if self.current_player is None:
            self.current_player = self.player1
            return
        # swap player1 and player2
        if self.current_player == self.player1:
            self.current_player = self.player2
        else:
            self.current_player = self.player1

    def start(self):
        self.game.start()
        while self.game.is_running:
            self.next_player()
            self.current_player.make_move()
            self.view.update_view()
