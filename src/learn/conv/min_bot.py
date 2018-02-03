import copy

import numpy as np
try:
    import torch
    from torch.autograd import Variable
except Exception:
    pass

from src.play.model.Move import Move
from src.play.model.Game import BLACK, WHITE
from .utils import network_input
from .our_model import ConvNet


class NewBot():

    def __init__(self, verbose=False):
        # Load Model
        self.model = ConvNet(in_channels=4, conv_depth=9, n_filters=64)
        self.model.load_state_dict(torch.load(
            'src/learn/conv/9depth_64filters_1.0mtsize/nets/epoch12.pth',
            map_location=lambda storage, loc: storage))
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.verbose = verbose

    @staticmethod
    def generate_input(board, player_value):
        X = network_input(board.reshape(1, 9, 9), np.array([player_value]))
        X = Variable(torch.from_numpy(X.astype(float)).float(), volatile=True)
        return X

    def genmove(self, color, game) -> Move:
        board = np.array(game.board)
        my_value = WHITE if color == 'w' else BLACK
        # enemy_value = BLACK if my_value == WHITE else WHITE
        inp = self.generate_input(board, my_value)
        if self.verbose:
            print(inp)
        policy = self.model(inp)
        policy = policy.data.numpy().flatten()

        playable_locations = game.get_playable_locations(color)

        # Default: passing
        policy_move = Move(is_pass=True)
        policy_move_prob = policy[81]

        for move in playable_locations:
            if self.verbose:
                print(move)
            if move.is_pass:
                continue

            if policy[move.to_flat_idx()] > policy_move_prob:
                policy_move = move
                policy_move_prob = policy[move.to_flat_idx()]

        return policy_move
