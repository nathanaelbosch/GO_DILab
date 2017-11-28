"""Replay game from sgf, then generate the training data out of it

On each move we can create a new datapoint, or even 8 adding symmetries!
"""
import os
import sgf
import random as rn
import numpy as np
import pickle
from src.play.model.Game import Game
from src.play.model.Game import WHITE, BLACK, EMPTY
from src.play.model.Move import Move


# OUT_PATH = 'data/training_data/simplest_move_prediction/'
DIR_PATH = 'src/learn/dev_nath_win_prediction/'


def error_resistance(funct):
    def asdf(*args, **kwargs):
        try:
            return funct(*args, **kwargs)
        except Exception as e:
            # raise e
            return
    return asdf


@error_resistance
def replay_game(sgf_line, func):
    """Simply recreate a game from a sgf file

    More of a proof-of-concept or example than really a necessary function.
    We will use some modified version of this to create the training data.
    """
    collection = sgf.parse(sgf_line)

    # This all only works if the SGF contains only one game
    game_tree = collection.children[0]
    game_properties = game_tree.nodes[0].properties
    # game_id = game_properties['GN'][0]

    if not(game_properties['RE'][0].startswith('B') or
            game_properties['RE'][0].startswith('W')):
        return None
    black_win = True if game_properties['RE'][0].startswith('B') else False

    game = Game(game_properties)
    # board = Board([[0]*9]*9)
    out = []
    for n in game_tree.nodes[1:]:
        player_color = list(n.properties.keys())[0]
        move = Move.from_sgf(str(n.properties[player_color][0]))
        # board[move.to_matrix_location()] = 1 if player_color=='b' else -1
        # neighbors = board.get_all_neigh
        game.play(move, player_color.lower(), checking=False)
        out.append(func(game, player_color.lower(), black_win))
    out = np.stack(out)
    # print(out.shape)
    return out


def to_numpy(game, player, black_win):
    """
    In:     My Board; Enemy Board
    Out:    1 if I win else -1
    """
    b = game.board

    black_board = (b == BLACK) * 2 - 1
    white_board = (b == WHITE) * 2 - 1

    black_board_vect = black_board.reshape(
        1, black_board.shape[0]*black_board.shape[1])
    white_board_vect = white_board.reshape(
        1, white_board.shape[0]*white_board.shape[1])

    if black_win:
        result = [1, 0]
    else:
        result = [0, 1]

    vect = np.append([black_board_vect, white_board_vect], result)

    return vect


def foo(line):
    """I only do this because pool.map can not work with lambda functions!"""
    return replay_game(line, to_numpy)


def main():
    import multiprocessing
    pool = multiprocessing.Pool()

    file = 'data/full_file.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    numgames = 5000
    lines = rn.sample(lines, numgames)

    if False:
        # Do the data-generation in batches
        max_batchsize = 5000
        i = 1
        still_todo = numgames
        filepath = os.path.join(DIR_PATH, '{}_games.csv'.format(numgames))
        if os.path.isfile(filepath):
            os.remove(filepath)
        f = open(filepath, 'ab')
        while still_todo > 0:
            print('Batch', i)
            i += 1
            batch_lines = lines[:max_batchsize]
            still_todo = still_todo - max_batchsize
            if still_todo > 0:
                lines = lines[max_batchsize:]

            data = pool.map(foo, batch_lines)

            data = [d for d in data if d is not None]
            data = np.concatenate(data)
            print(data.shape)
            print('White wins {}'.format(data[:, -1].sum() / data.shape[0]))
            print('Black wins {}'.format(data[:, -2].sum() / data.shape[0]))
            np.savetxt(f, data, delimiter=',', fmt='%d')
        f.close()
    else:
        data = pool.map(foo, lines)

        data = [d for d in data if d is not None]
        data = np.concatenate(data)
        print(data.shape)
        print('White wins {}'.format(data[:, -1].sum() / data.shape[0]))
        print('Black wins {}'.format(data[:, -2].sum() / data.shape[0]))

        np.save(os.path.join(
            DIR_PATH, '{}_games.npy'.format(numgames)),
            data)


if __name__ == '__main__':
    main()
