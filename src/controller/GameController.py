import threading

from src.model.Game import InvalidMove_Error


class GameController(threading.Thread):

    def __init__(self, game, views, player1, player2):
        threading.Thread.__init__(self)
        self.game = game
        self.views = views
        self.player1 = player1
        self.player2 = player2
        self.current_player = None

    def next_turn(self):
        # start with player1 if not otherwise defined
        if self.current_player is None:
            self.current_player = self.player1
            return
        # swap player1 and player2
        if self.current_player == self.player1:
            self.current_player = self.player2
        else:
            self.current_player = self.player1

    def call_views(self, method, *args):  # via stackoverflow.com/a/2682075/2474159
        for view in self.views:
            getattr(view, method)(*args)

    def run(self):
        self.game.start()
        while self.game.is_running:
            self.next_turn()
            self.call_views('show_player_turn_start', self.current_player.name)
            # loop until a move is valid
            # (can lead to inf-loops when bots fail to produce valid moves)
            while True:
                try:
                    self.current_player.make_move()
                    break
                except InvalidMove_Error as e:
                    self.call_views('show_error', ' '.join(e.args))
            self.call_views('show_player_turn_end', self.current_player.name)

        # TODO
        # relieve the Game-class from the task to print end-of-game
        # things, view must do this: self.view.show_game_ended()
