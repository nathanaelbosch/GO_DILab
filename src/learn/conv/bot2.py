import copy

import numpy as np
try:
    import torch
    from torch.autograd import Variable
except Exception:
    pass

from src.play.model.Move import Move
from src.play.model.Game import BLACK, WHITE
from .utils import minimal_network_input
from .our_model import ConvNet


class NewBot():

    def __init__(self, verbose=False):

        # original saved file with DataParallel
        self.model = ConvNet(in_channels=2, conv_depth=9, n_filters=256)
        file = 'src/learn/conv/saved/9depth_256filters_15.0mtsize/nets/epoch7.pth'
        state_dict = torch.load(
            file, map_location=lambda storage, loc: storage)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]        # remove `module.`
            new_state_dict[name] = v

        # load params
        self.model.load_state_dict(new_state_dict)

        # No gradient required
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.verbose = verbose

    @staticmethod
    def generate_input(board, player_value):
        X = minimal_network_input(board.reshape(1, 9, 9), np.array([player_value]))
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
