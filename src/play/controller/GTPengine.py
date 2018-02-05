import argparse
import sys
from os.path import dirname, abspath

project_root_dir = dirname(dirname(dirname(dirname(abspath(__file__)))))
sys.path.append(project_root_dir)

from src.play.model.Game import *

from src.play.controller.bots.HumanConsole import HumanConsole
from src.play.controller.bots.HumanGui import HumanGui
from src.play.controller.bots.RandomBot import RandomBot
from src.play.controller.bots.RandomGroupingBot import RandomGroupingBot
from src.learn.dev_nath.SimplestNNBot import SimplestNNBot
from src.learn.dev_nath_win_prediction.WinPredictionBot import WinPredictionBot
from src.learn.dev_ben.NNBot_ben1 import NNBot_ben1
from src.learn.dev_yu.MovePredictionBot import MovePredictionBot
from src.learn.dev_kar.LibertyNNBot import LibertyNNBot
from src.learn.bots._11.bot import Bot_11
from src.learn.bots._21.bot import Bot_21
from src.learn.bots._12.bot import Bot_12
from src.learn.bots._22.bot import Bot_22
from src.learn.bots._31.bot import Bot_31
from src.learn.bots._32.bot import Bot_32
from src.learn.mcts.MCTSBot import MCTSBot
from src.learn.conv.bot import ConvBot_value, ConvBot_policy
from src.learn.conv.bot2 import NewBot


class GTPengine:

    def __init__(self, logging_level, player_type_str='RandomBot'):
        self.game = Game()

        self.player_types = {}
        player_types_arr = [
            HumanConsole,
            HumanGui,
            RandomBot,
            RandomGroupingBot,
            SimplestNNBot,
            WinPredictionBot,
            NNBot_ben1,
            MovePredictionBot,
            LibertyNNBot,
            Bot_11,
            Bot_21,
            Bot_12,
            Bot_22,
            Bot_31,
            Bot_32,
            MCTSBot,
            ConvBot_value,
            ConvBot_policy,
            NewBot
        ]
        for player_type in player_types_arr:
            self.player_types[player_type.__name__.lower()] = player_type
        player_type_str = player_type_str.lower()
        if player_type_str not in self.player_types:
            print('no player-type named: ' + player_type_str)
            sys.exit(1)

        self.bot = self.player_types[player_type_str]()
        self.controller = None
        self.stdin = None
        self.stdout = None
        self.logger = Utils.get_unique_file_logger(self, logging_level)
        self.logger.info(
            '  start: ' + self.bot.__class__.__name__ + ', ' + __file__)
        self.gtp_commands = {}
        gtp_methods = [
            self.set_player_type,  # not a GTP-command
            self.protocol_version,
            self.name,
            self.version,
            self.known_command,
            self.list_commands,
            self.quit,
            self.boardsize,
            self.clear_board,
            self.komi,
            self.play,
            self.genmove,
            self.showboard,
        ]
        # create a dictionary with the method-name as key and the method as
        # value. in that way, valid GTP commands serve directly as keys to
        # get the corresponding method
        for method in gtp_methods:
            self.gtp_commands[method.__name__] = method

    def set_player_type(self, args):
        if len(args) == 0:
            self.send_failure_response('no player type passed')
            return
        player_type = args[0].lower()
        if player_type not in self.player_types:
            self.send_failure_response(
                'player type ' + player_type + ' unknown')
            return
        self.bot = self.player_types[player_type]()
        self.send_success_response('switched to player type ' + player_type)

    def handle_input_from_controller(self, input):
        self.logger.info('receive: ' + input.strip())
        parts = input.split(' ')
        command = parts[0]
        if command not in self.gtp_commands:
            self.send_failure_response('command unknown: ' + input)
            return
        self.gtp_commands[command](parts[1:])

    def run(self):
        while True:
            stdin_line = self.stdin.readline().strip()
            if len(stdin_line) == 0:  # ignore empty lines
                continue
            self.handle_input_from_controller(stdin_line)

    def write_out(self, message):
        self.logger.info('   send: ' + message.strip())
        if self.stdout is not None:
            self.stdout.write(message)
            self.stdout.flush()
        if self.controller is not None:
            self.controller.handle_input_from_engine(self, message)

    def send_success_response(self, message=''):
        # following lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#SECTION00044000000000000000
        self.write_out('= ' + message + '\n\n')

    def send_failure_response(self, message):
        self.write_out('? ' + message + '\n\n')

    @staticmethod  # returns None if color is invalid
    def validate_color(col):
        col = col.lower()
        if col == 'white':
            return 'w'
        if col == 'black':
            return 'b'
        if col in ('w', 'b'):
            return col
        return None

    # ---------- COMMAND METHODS ----------

    def protocol_version(self, args):
        # version of the GTP specification:
        # lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
        self.send_success_response('2')

    def name(self, args):
        self.send_success_response(self.bot.__class__.__name__)

    def version(self, args):
        self.send_success_response('1.0')

    def known_command(self, args):
        if len(args) == 0:
            self.send_failure_response('no command passed')
        else:
            self.send_success_response(
                str(args[0] in self.gtp_commands).lower())

    def list_commands(self, args):
        self.send_success_response(
            '\n'+'\n'.join(list(self.gtp_commands.keys())))

    def quit(self, args):
        self.send_success_response()
        if self.stdout is not None:
            self.stdout.close()
        if self.controller is None:  # otherwise controller does this
            sys.exit(0)

    def boardsize(self, args):
        if len(args) == 0:
            self.send_failure_response('no value passed')
            return
        board_size = int(args[0])
        if not 19 >= board_size > 1:  # whats a meaningful min value?
            self.send_failure_response('unacceptable size')
            return
        self.game.size = board_size  # can't just change it during the game like that though - trigger new game?
        self.send_success_response()

    def clear_board(self, args):
        self.game = Game()  # just like that? seems kinda brutal
        self.send_success_response()

    def komi(self, args):
        if len(args) == 0:
            self.send_failure_response('no value passed')
            return
        try:
            komi = float(args[0])
            self.game.komi = komi
            self.send_success_response()
        except ValueError:
            self.send_failure_response('not an acceptable value: ' + args[0])

    def play(self, args):
        if len(args) < 2:
            self.send_failure_response('no color and/or move provided')
            return
        color = self.validate_color(args[0])
        if color is None:
            self.send_failure_response('invalid color ' + args[0])
            return
        gtp_move = args[1]
        if 'i' in gtp_move or 'I' in gtp_move:
            self.send_failure_response(
                'i is excluded from board coordinates in GTP')
            return
        move = Move().from_gtp(gtp_move, self.game.size)

        if not move.is_on_board(self.game.size):
            self.send_failure_response(
                'Location ' + gtp_move + ' is outside of board with size ' +
                str(self.game.size))
            return
        try:
            self.game.play(move, color)
            self.send_success_response()
        except InvalidMove_Error:
            self.send_failure_response('illegal move')
            return

    def genmove(self, args):
        if len(args) == 0:
            self.send_failure_response('no color provided')
            return
        color = self.validate_color(args[0])
        if color is None:
            self.send_failure_response('invalid color ' + args[0])
            return
        move = self.bot.genmove(color, self.game)
        try:
            self.game.play(move, color)
            self.send_success_response(move.to_gtp(self.game.size))
        except InvalidMove_Error as e:
            self.send_failure_response('Sorry, I generated an invalid move: ' + str(e))

    def showboard(self, args):
        print(self.game.board.__str__())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--player',
        help=('options: "HumanConsole", "HumanGui", "RandomBot", "RandomGroupingBot", ' +
              '"SimplestNNBot", "WinPredictionBot", "NNBot_ben1", "MovePredictionBot", "LibertyNNBot"')
    )
    player_type_str = parser.parse_args().player
    if player_type_str is None:
        player_type_str = 'RandomBot'
    gtp_engine = GTPengine(logging_level=logging.INFO, player_type_str=player_type_str)
    gtp_engine.stdin = sys.stdin
    gtp_engine.stdout = sys.stdout
    gtp_engine.run()


if __name__ == '__main__':
    main()
