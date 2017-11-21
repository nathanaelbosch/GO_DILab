from Game import InvalidMove_Error
from Move import Move


class HumanConsole:

    @staticmethod
    def genmove(color, game) -> Move:
        move = None
        while move is None:
            try:
                print('\nsubmit your move:')
                move_str = input()
                move = Move().from_gtp(move_str, game.size)
                game.play(move, color, testing=True)
            except InvalidMove_Error as e:
                move = None
                print('\ninvalid move, choose another location or "pass":')
        return move
