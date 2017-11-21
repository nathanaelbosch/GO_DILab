import time

from GTPengine import GTPengine
from Game import Game
from Move import Move
from bots.HumanConsole import HumanConsole
from bots.RandomBot import RandomBot
from bots.RandomGroupingBot import RandomGroupingBot


END_OF_TURN_SLEEP_TIME = 0.5


class GTPcontroller:

    def __init__(self, player1type, player2type):
        self.game = Game()
        self.player1 = Player('b')
        self.player1.engine.controller = self
        self.player2 = Player('w')
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

    @staticmethod
    def send_to_player(player, command):
        print('      send to ' + player.name + ' (' + player.color + '): ' + command)
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
        exit(0)

    def handle_input_from_engine(self, engine, input):
        input = input.strip()
        player = self.map[engine]
        print('received from ' + player.name + ' (' + player.color + '): ' + input)
        player.latest_response = input


class Player:
    def __init__(self, color):
        self.engine = GTPengine()
        self.color = color
        self.name = 'unknown'
        self.latest_response = None

    def get_latest_response(self):
        resp = self.latest_response
        self.latest_response = None
        return resp


def main():
    GTPcontroller(
        RandomBot.__name__,
        RandomGroupingBot.__name__,
        # HumanConsole.__name__,
    ).run()


if __name__ == '__main__':
    main()
