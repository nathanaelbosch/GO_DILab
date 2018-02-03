import numpy as np


from src.play.model.Game import BLACK, WHITE, EMPTY


def encode_board(boards, player_value):
    """Generate a categorical board encoding

    The board will be encoded into a 3*81-vector.
    Each of those 81-vectors basically stand for a True/False-encoding for
    player, opponent, and "empty".
    """
    try:
        player_value = player_value.reshape((len(player_value), 1, 1))
        # pass
    except AttributeError:
        pass

    player_board = np.where(
        player_value==WHITE,
        boards==WHITE,
        boards==BLACK)
    opponent_board = np.where(
        player_value==WHITE,
        boards==BLACK,
        boards==WHITE)
    empty_board = (boards==EMPTY)

    out = np.stack(
        (player_board,
         opponent_board,
         empty_board),
        axis=1)

    assert (out.sum(axis=1) == 1).all(), 'Something went wrong'

    return out


def network_input(boards, player_values):
    """Strongly inspired by AlphaGO"""
    encoded_boards = encode_board(boards, player_values).astype(float)
    current_player = np.multiply(
        np.ones((encoded_boards.shape[0], 1, 9, 9)),
        player_values.reshape(-1, 1, 1, 1))
    X = np.concatenate((encoded_boards, current_player), axis=1)
    return X


def minimal_network_input(boards, player_values):
    """Stronglyer inspired by AlphaGO"""
    encoded_boards = encode_board(boards, player_values).astype(float)
    X = encoded_boards[:, :2, :, :]
    return X


def policy_output_categorical(moves):
    moves[moves==-1] = 81
    return moves


def value_output(results, player_value):
    """Contains a single scala in {1, -1} to designate the winning player"""
    black_wins = results.str.lower().str.startswith('b')
    white_wins = results.str.lower().str.startswith('w')
    player_wins = np.where(
        player_value==WHITE,
        white_wins,
        black_wins)
    player_wins = (player_wins * 2) - 1
    return player_wins
