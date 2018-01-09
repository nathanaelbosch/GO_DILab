from os.path import abspath
import numpy as np
import copy
from pprint import pprint

from src.learn.BaseNNBot import BaseNNBot
from src.play.model.Board import EMPTY, BLACK, WHITE
from src.play.model.Move import Move

import src.learn.mcts.example as ex


class MCTSBot():

    def __init__(self, n=10):
        # Number of simulations
        # self.n = n

        board = ex.Board()
        self.mc = ex.MonteCarlo(
            board,
            max_moves=2,
            time=3,
            verbose=True,
        )
        self.mc.update(board.start())

    def __str__(self):
        return 'MCTS-Bot'

    def genmove(self, color, game) -> Move:
        """Simulate some games for each move and return the best one"""
        # print(color)
        # print(game.play_history)
        # print(self.mc.states)
        if not len(game.play_history) == (len(self.mc.states) - 1):
            # Last play not yet in our states:
            last_player, last_move = game.play_history[-1]
            # pprint(game.play_history)
            # print(last_player, last_move)
            missing_state = self.mc.board.next_state(
                self.mc.states[-1], last_move)
            self.mc.update(missing_state)

        # print('Current board in our mc:')
        # _b = self.mc.states[-1][0]
        # _b = self.mc.board.from_tuple(_b)
        # print(_b)

        move = self.mc.get_play()

        # Update our saved states
        resulting_state = self.mc.board.next_state(
            self.mc.states[-1], move)
        self.mc.update(resulting_state)

        return move
