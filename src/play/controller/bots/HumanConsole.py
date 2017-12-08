from src.play.model.errors import InvalidMove_Error, BadInput_Error

from src.play.model.Move import Move


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
            except BadInput_Error as e:
                move = None
                print('\nbad input, retry or "pass":')
        return move
