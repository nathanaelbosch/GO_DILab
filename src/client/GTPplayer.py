import sys
import random as rn

from src.play import Game
from src.play.utils.Move import Move


class GTPplayer:

    def __init__(self, game):
        self.game = game
        self.out = sys.stdout

        self.gtp_commands = {
            self.genmove.__name__: self.genmove,
            self.play.__name__: self.play,
            self.list_commands.__name__: self.list_commands,
            self.known_command.__name__: self.known_command,
            self.quit.__name__: self.quit,
            self.protocol_version.__name__: self.protocol_version,
            self.version.__name__: self.version,
            self.name.__name__: self.name,
        }

    def send_success_response(self, message=''):
        # following lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html#SECTION00044000000000000000
        self.out.write('= ' + message + '\n\n')
        self.out.flush()

    def send_failure_response(self, message):
        self.out.write('? ' + message + '\n\n')
        self.out.flush()

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

    def genmove(self, args):
        if len(args) == 0:
            self.send_failure_response('no color provided')
            return
        color = self.validate_color(args[0])
        if color is None:
            self.send_failure_response('invalid color ' + args[0])
            return
        random_move = rn.choice(self.game.get_playable_locations(color))
        self.game.play(random_move, color)  # or does the controller tell us to do this?
        self.send_success_response(random_move.to_gtp())

    def play(self, args):
        if len(args) < 2:
            self.send_failure_response('no color and/or move provided')
            return
        color = self.validate_color(args[0])
        if color is None:
            self.send_failure_response('invalid color ' + args[0])
            return
        gtp_move = args[1]
        move = Move().from_gtp(gtp_move)
        self.game.play(move, color)
        self.send_success_response()

    def list_commands(self, args):
        self.send_success_response('\n'+'\n'.join(list(self.gtp_commands.keys())))

    def known_command(self, args):
        if len(args) == 0:
            self.send_failure_response('no command passed')
        else:
            self.send_success_response(str(args[0] in self.gtp_commands))

    def quit(self, args):
        self.send_success_response()
        self.out.close()
        exit(0)

    def protocol_version(self, args):
        # version of the GTP specification lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
        self.send_success_response('2')

    def version(self, args):
        self.send_success_response('1.0')

    def name(self, args):
        self.send_success_response('Hikaru - Random Bot')

    def run(self):
        while True:
            stdin_line = sys.stdin.readline().strip()
            if len(stdin_line) == 0:  # ignore empty lines
                continue
            # print('reading input: ' + stdin_line)
            parts = stdin_line.split(' ')
            command = parts[0]
            if command not in self.gtp_commands:
                self.send_failure_response('command unknown: ' + stdin_line)
                continue
            self.gtp_commands[command](parts[1:])


def main():
    game = Game()
    gtp_player = GTPplayer(game)
    gtp_player.run()


if __name__ == '__main__':
    main()
