from src.model.Game import InvalidMove_Error


class GameController:

    def __init__(self, game, view, player1, player2):
        self.game = game
        self.view = view
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

    def start(self):
        self.game.start()
        while self.game.is_running:
            self.next_turn()
            self.view.show_player_turn_start(self.current_player.name)
            # loop until a move is valid
            # (can lead to inf-loops when bots fail to produce valid moves)
            while True:
                try:
                    self.current_player.make_move()
                    break
                except InvalidMove_Error as e:
                    self.view.show_error(' '.join(e.args))
            self.view.show_player_turn_end(self.current_player.name)
