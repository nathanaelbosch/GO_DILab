from Move import Move


class HumanConsole:

    @staticmethod
    def genmove(color, game) -> Move:
        move_str = input()
        return Move().from_gtp(move_str, game.size)
