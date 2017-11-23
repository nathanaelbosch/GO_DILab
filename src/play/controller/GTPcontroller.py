import threading
import time
import sys

from src import Utils
from src.play.controller.bots.HumanGui import HumanGui
from src.play.model.Game import Game, InvalidMove_Error
from src.play.model.Move import Move

from src.play.controller.GTPengine import GTPengine

END_OF_TURN_SLEEP_TIME = 0


class GTPcontroller(threading.Thread):

    def __init__(self, player1type, player2type, logging_level):
        threading.Thread.__init__(self)
        self.logger = Utils.get_unique_file_logger(self, logging_level)
        self.game = Game()
        self.view = None
        self.player1 = Player('b', logging_level)
        self.player1.engine.controller = self
        self.player2 = Player('w', logging_level)
        self.player2.engine.controller = self
        self.map = {
            self.player1.engine: self.player1,
            self.player2.engine: self.player2,
        }
        self.send_to_player(self.player1, 'set_player_type ' + player1type)
        self.send_to_player(self.player2, 'set_player_type ' + player2type)
        self.player1.name = self.wait_for_response(self.player1, 'name')[2:]
        self.player2.name = self.wait_for_response(self.player2, 'name')[2:]
        self.current_player = self.player1
        self.other_player = self.player2

    def log_and_print(self, message):
        self.logger.info(message)
        print(message)

    def send_to_player(self, player, command):
        self.log_and_print('      send to ' + player.name + ' (' + player.color + '): ' + command)
        player.engine.handle_input_from_controller(command)

    def broadcast(self, command):
        self.send_to_player(self.player1, 'quit')
        self.send_to_player(self.player2, 'quit')

    def wait_for_response(self, player, message):
        self.send_to_player(player, message)
        while player.latest_response is None:
            pass
        return player.get_latest_response()

    def run(self):
        self.game.start()
        while self.game.is_running:
            print('\nnext turn\n')
            response = self.wait_for_response(self.current_player, 'genmove ' + self.current_player.color)
            move = response[2:]  # strip away the "= "
            self.send_to_player(self.other_player, 'play ' + self.current_player.color + ' ' + move)

            self.game.play(Move().from_gtp(move, self.game.size), self.current_player.color)
            print('\n' + self.game.__str__())

            time.sleep(END_OF_TURN_SLEEP_TIME)

            # swap players for next turn
            if self.current_player == self.player1:
                self.current_player = self.player2
                self.other_player = self.player1
            else:
                self.current_player = self.player1
                self.other_player = self.player2

        self.broadcast('quit')
        print('\n' + self.game.__str__())
        if self.view is not None:
            self.view.game_ended()
        sys.exit(0)

    def handle_input_from_engine(self, engine, input):
        input = input.strip()
        player = self.map[engine]
        self.log_and_print('received from ' + player.name + ' (' + player.color + '): ' + input)
        player.latest_response = input

    def receive_move_from_gui(self, move):
        human = self.current_player.engine.bot
        if type(human) is HumanGui:
            try:
                self.game.play(move, self.current_player.color, testing=True)
                human.move = move
            except InvalidMove_Error as e:
                print('\ninvalid move')


class Player:
    def __init__(self, color, logging_level):
        self.engine = GTPengine(logging_level)
        self.color = color
        self.name = 'unknown'
        self.latest_response = None

    def get_latest_response(self):
        resp = self.latest_response
        self.latest_response = None
        return resp
