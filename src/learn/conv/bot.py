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
from .model_zero import ConvNet


class ConvBot():

    def __init__(self, logic: {'policy', 'value'}, verbose=False):
        # Load Model
        self.model = ConvNet(in_channels=4, conv_depth=5)
        self.model.load_state_dict(torch.load(
            'src/learn/conv/saved_nets/5depth_2.5m_tanh.pth',
            map_location=lambda storage, loc: storage))
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Bot logic
        self.logic = logic

        self.verbose = verbose

    def generate_input(self, board, player_value):
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
        policy, value = self.model(inp)
        policy = policy.data.numpy().flatten()
        value = value.data.numpy().flatten()

        playable_locations = game.get_playable_locations(color)

        # Default: passing
        policy_move = value_move = Move(is_pass=True)
        policy_move_prob = policy[81]
        value_move_prob = value

        for move in playable_locations:
            if self.verbose:
                print(move)
            if move.is_pass:
                continue

            if self.logic == 'value':
                # Play move on a test board
                test_board = copy.deepcopy(game.board)
                test_board.place_stone_and_capture_if_applicable_default_values(
                    move.to_matrix_location(), my_value)

                # Evaluate state - attention: Enemy's turn!
                # inp = self.generate_input(np.array(test_board), enemy_value)
                # _, enemy_win_prob = self.model(inp)
                # enemy_win_prob = enemy_win_prob.data.numpy().flatten()
                # my_new_value = -enemy_win_prob

                # Disregard that right now and just get my own win prob
                inp = self.generate_input(np.array(test_board), my_value)
                _, new_value = self.model(inp)
                new_value = new_value.data.numpy().flatten()
                if new_value > value_move_prob:
                    value_move = move
                    value_move_prob = new_value

            if self.logic == 'policy':
                if policy[move.to_flat_idx()] > policy_move_prob:
                    policy_move = move
                    policy_move_prob = policy[move.to_flat_idx()]

        if self.logic == 'policy':
            out_move = policy_move
        if self.logic == 'value':
            out_move = value_move

        return out_move


class ConvBot_value(ConvBot):
    def __init__(self):
        super().__init__('value')


class ConvBot_policy(ConvBot):
    def __init__(self):
        super().__init__('policy')
