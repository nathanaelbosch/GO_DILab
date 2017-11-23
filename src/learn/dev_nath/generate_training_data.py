"""Replay game from sgf, then generate the training data out of it

On each move we can create a new datapoint, or even 8 adding symmetries!
"""
import os
import sgf
import random as rn
import numpy as np
from src.play.model.Game import Game
from src.play.model.Game import WHITE, BLACK, EMPTY
from src.play.model.Move import Move


OUT_PATH = 'data/training_data/simplest_move_prediction/'


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
    n_0 = game_tree.nodes[0]
    game_id = n_0.properties['GN'][0]

    game = Game(n_0.properties)
    # board = Board([[0]*9]*9)
    out = []
    for n in game_tree.nodes[1:]:
        player_color = list(n.properties.keys())[0]
        move = Move.from_sgf(str(n.properties[player_color][0]))
        # board[move.to_matrix_location()] = 1 if player_color=='b' else -1
        # neighbors = board.get_all_neigh
        game.play(move, player_color.lower(), checking=False)
        out.append(func(game, move, player_color.lower()))
    out = np.stack(out)
    # print(out.shape)
    return out


def to_numpy(game, move, player):
    b = game.board
    if player == 'b':
        me = BLACK
        other = WHITE
    else:
        me = WHITE
        other = BLACK
    my_board = (b == me) * 2 - 1
    other_board = (b == other) * 2 - 1

    my_board_vect = my_board.reshape(
        1, my_board.shape[0]*my_board.shape[1])
    other_board_vect = other_board.reshape(
        1, other_board.shape[0]*other_board.shape[1])

    move_board = np.matrix([[0]*9]*9)
    move_board[move.to_matrix_location()] = 1
    move_vect = move_board.reshape(move_board.shape[0]*move_board.shape[1])

    # print(my_board_vect.shape)
    # print(other_board_vect.shape)
    # print(move_vect.shape)
    vect = np.append([my_board_vect, other_board_vect], move_vect)
    # print(vect)
    return vect


def foo(line):
    return replay_game(line, to_numpy)


def main():
    file = 'data/full_file.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = rn.sample(lines, 1000)
    # print(lines)

    import multiprocessing
    pool = multiprocessing.Pool()
    data = pool.map(foo, lines)

    # data = map(lambda x: replay_game(x, to_numpy), lines)
    data = [d for d in data if d is not None]
    data = np.concatenate(data)
    print(data.shape)
    np.savetxt('test.out', data, delimiter=',', fmt='%d')
    # print(data[0].shape)
    # print(data[1].shape)
    # print(data[0][0].shape)


if __name__ == '__main__':
    main()
