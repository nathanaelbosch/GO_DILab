"""Replay game from sgf, then generate the training data out of it

On each move we can create a new datapoint, or even 8 adding symmetries!
"""
import os
import sgf
import random as rn
import numpy as np
from src.play.model.Game import Game
from src.play.utils.Move import Move


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
    out = []
    for n in game_tree.nodes[1:]:
        player_color = list(n.properties.keys())[0]
        move = Move.from_sgf(str(n.properties[player_color][0]))
        game.play(move, player_color.lower())
        out.append(func(game, move, player_color.lower()))
    out = np.stack(out)
    print(out.shape)
    return out


def to_numpy(game, move, player):
    b = game.board
    board_vect = b.reshape(1, b.shape[0]*b.shape[1])
    new_game = Game()
    new_game.play(move, player.lower())
    b = new_game.board
    move_vect = b.reshape(1, b.shape[0]*b.shape[1])
    color = 1 if player.lower == 'b' else -1
    vect = np.append(color, (board_vect, move_vect))
    # print(vect)
    return vect


def main():
    file = 'data/full_file.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    lines = rn.sample(lines, 10)

    # import multiprocessing
    # pool = multiprocessing.Pool()
    # data = pool.map(foo, lines)

    data = map(lambda x: replay_game(x, to_numpy), lines)
    data = [d for d in data if d is not None]
    print(data[0].shape)
    print(data[0][0].shape)


if __name__ == '__main__':
    main()
