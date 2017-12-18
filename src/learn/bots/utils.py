"""Utils

The multiple bots here are just all combinations of input and output type
presented in the markdown. It therefore makes sense to define the
generation of those data formats here, as they will be shared accross
multiple bots.
"""
import numpy as np
from keras.utils import to_categorical

from src.play.model.Game import BLACK, WHITE, EMPTY


def separate_data(data):
    """Given the output of the SQL call separate that array into it's parts"""
    results = data[:, -1]
    ids = data[:, 0]
    colors = data[:, 1].astype(int)[:, None]
    moves = data[:, 2].astype(int)[:, None]
    boards = data[:, 3:-1].astype(np.float64)
    out = {'results': results,
           'ids': ids,
           'colors': colors,
           'moves': moves,
           'boards': boards}
    return out


def value_output(results):
    r = np.chararray(results.shape)
    r[:] = results[:]
    black_wins = r.lower().startswith(b'b')[:, None]
    white_wins = r.lower().startswith(b'w')[:, None]
    # draws = results.lower().startswith('D')
    out = np.concatenate((black_wins, white_wins), axis=1)
    return out


def policy_output(moves):
    moves[moves==-1] = 81
    out = to_categorical(moves)
    assert out.shape[1] == 82
    assert (out.sum(axis=1) == 1).all()

    return out


def simple_board(boards):
    """Generate the simplest board encoding

    The board will be encoded into a 81-vector, with -1, 0, 1 values.
    The boards are actually already given in this format
    """
    return boards


def encode_board(boards, colors):
    """Generate a categorical board encoding

    The board will be encoded into a 3*81-vector.
    Each of those 81-vectors basically stand for a True/False-encoding for
    player, opponent, and "empty".
    """
    player_board = np.where(
        colors==WHITE,
        boards==WHITE,
        boards==BLACK)
    opponent_board = np.where(
        colors==WHITE,
        boards==BLACK,
        boards==WHITE)
    empty_board = (boards==EMPTY)

    def normalize(board):
        return (board * 3 - 1)/np.sqrt(2)

    out = np.concatenate(
        (normalize(player_board),
         normalize(opponent_board),
         normalize(empty_board)),
        axis=1)

    return out


