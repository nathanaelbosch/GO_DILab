import sys
from time import strftime
from datetime import datetime
# from os.path import dirname, abspath
# project_dir = dirname(dirname(dirname(abspath(__file__))))
# sys.path.append(project_dir)

from Game import *
from bots.RandomBot import RandomBot
from bots.RandomGroupingBot import RandomGroupingBot


class GTPprogram:

    def __init__(self, game, bot, logfile):
        self.game = game
        self.bot = bot
        self.stdout = sys.stdout
        self.logfile = open(logfile, 'w')
        self.write_log('  start: ', self.bot.__class__.__name__ + ', ' + __file__)
        self.gtp_commands = {}
        gtp_methods = [
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

    def run(self):
        while True:
            stdin_line = sys.stdin.readline().strip()
            self.write_log('receive: ', stdin_line)
            if len(stdin_line) == 0:  # ignore empty lines
                continue
            # print('reading input: ' + stdin_line)
            parts = stdin_line.split(' ')
            command = parts[0]
            if command not in self.gtp_commands:
                self.send_failure_response('command unknown: ' + stdin_line)
                continue
            self.gtp_commands[command](parts[1:])

    def write_log(self, type, message):
        self.logfile.write('[' + datetime.now().strftime('%d.%m.%Y %H:%M:%S.%f') + '] ' + type + message.strip() + '\n')
        self.logfile.flush()

    def write_out(self, message):
        self.stdout.write(message)
        self.stdout.flush()
        self.write_log('   send: ', message)

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
        # version of the GTP specification lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
        self.send_success_response('2')

    def name(self, args):
        self.send_success_response('Hikaru - ' + self.bot.__class__.__name__)

    def version(self, args):
        self.send_success_response('1.0')

    def known_command(self, args):
        if len(args) == 0:
            self.send_failure_response('no command passed')
        else:
            self.send_success_response(str(args[0] in self.gtp_commands).lower())

    def list_commands(self, args):
        self.send_success_response('\n'+'\n'.join(list(self.gtp_commands.keys())))

    def quit(self, args):
        self.send_success_response()
        self.stdout.close()
        self.logfile.close()
        exit(0)

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
            self.send_failure_response('i is excluded from board coordinates in GTP')
            return
        move = Move().from_gtp(gtp_move, self.game.size)

        if not move.is_on_board(self.game.size):
            self.send_failure_response('Location ' + gtp_move + ' is outside of board with size ' + str(self.game.size))
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
        self.game.play(move, color)
        self.send_success_response(move.to_gtp(self.game.size))

    def showboard(self, args):
        print(self.game.board.__str__())


def main():
    game = Game()
    logfile = 'log_' + strftime('%d-%m-%Y_%H-%M-%S') + '.txt'
    # bot = RandomBot(game)
    bot = RandomGroupingBot()
    gtp_program = GTPprogram(game, bot, logfile)
    gtp_program.run()


if __name__ == '__main__':
    main()
