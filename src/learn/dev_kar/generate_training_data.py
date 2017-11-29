"""Replay game from sgf, then generate the training data out of it

On each move we can create a new datapoint, or even 8 adding symmetries!
"""
import os
import sgf
import random as rn
import numpy as np
import time
from scipy import ndimage
from src.play.model.Game import Game
from src.play.model.Game import WHITE, BLACK, EMPTY
from src.play.model.Move import Move



# OUT_PATH = 'data/training_data/simplest_move_prediction/'


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
    my_board = (b == me)*1
    other_board = (b == other)*1
    empty_board = (np.matrix([[1]*9]*9)) - my_board - other_board
    my_board_vals = (np.matrix([[0]*9]*9))
    other_board_vals = (np.matrix([[0]*9]*9))

    label_mine, mine_labels = ndimage.label(my_board)
    label_other,other_labels = ndimage.label(other_board)

    for label in range(1,mine_labels+1):
        my_board_label = (label_mine == label)*1
        dilated = ndimage.binary_dilation(my_board_label)
        dilated = ((dilated - other_board - my_board_label)==1)
        L = np.count_nonzero(dilated)    # L = Total number of liberties of group
        stone_list = list(zip(np.where(my_board_label)[0],np.where(my_board_label)[1]))
        for location in stone_list:
            stone_dilated = np.matrix([[0]*9]*9)
            stone_dilated[location] = 1
            stone_dilated = ndimage.binary_dilation(stone_dilated)
            stone_liberty = (stone_dilated - other_board - my_board_label) == 1
            sL = np.count_nonzero(stone_liberty)
            if L == 0:
                break
            my_board_vals[location] = sL/L

    for label in range(1,other_labels+1):
        other_board_label = (label_other == label) * 1
        dilated = ndimage.binary_dilation(other_board_label)
        dilated = ((dilated - other_board - my_board_label)==1)
        L = np.count_nonzero(dilated)
        stone_list = list(zip(np.where(other_board_label)[0],np.where(other_board_label)[1]))
        for location in stone_list:
            stone_dilated = np.matrix([[0]*9]*9)
            stone_dilated[location] = 1
            stone_dilated = ndimage.binary_dilation(stone_dilated)
            stone_liberty = (stone_dilated - other_board - my_board_label) == 1
            sL = np.count_nonzero(stone_liberty)
            if L == 0:
                break
            other_board_vals[location] = sL / L

    my_board_vect = my_board_vals.reshape(
        1, my_board_vals.shape[0]*my_board_vals.shape[1])
    other_board_vect = other_board_vals.reshape(
        1, other_board_vals.shape[0]*other_board_vals.shape[1])

    move_board = np.matrix([[0]*9]*9)
    move_board[move.to_matrix_location()] = 1
    move_vect = move_board.reshape(move_board.shape[0]*move_board.shape[1])

    # print(my_board_vect.shape)
    # print(other_board_vect.shape)
    # print(move_vect.shape)
    vect = np.append([my_board_vect, other_board_vect], move_vect)

    return vect


def foo(line):
    return replay_game(line, to_numpy)


def main():
    import multiprocessing
    pool = multiprocessing.Pool()

    file = '../../../data/full_file.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
    numgames = 100
    lines = rn.sample(lines, numgames)
    # print(lines)

    max_batchsize = 5000
    first = True; i=1
    still_todo = numgames
    filepath = '/Users/karthikeyakaushik/Documents/GO_DILab/src/learn/dev_kar/{}_games.csv'.format(numgames)
    if os.path.isfile(filepath):
        os.remove(filepath)
    f = open('/Users/karthikeyakaushik/Documents/GO_DILab/src/learn/dev_kar/{}_games.csv'.format(numgames), 'w')
    while still_todo > 0:
        print('Batch', i); i+=1
        batch_lines = lines[:max_batchsize]
        still_todo = still_todo - max_batchsize
        if still_todo > 0:
            lines = lines[max_batchsize:]

        data = pool.map(foo, batch_lines)

        # data = map(lambda x: replay_game(x, to_numpy), lines)
        data = [d for d in data if d is not None]
        data = np.concatenate(data)
        print(data.shape)
        print (data.dtype)
        print(data[0].shape)
        print(data[1].shape)
        print(data[0][0].shape)


        np.savetxt(f, data, delimiter=',', fmt='%10.5f')

        print(data[0].shape)
        print(data[1].shape)
        print(data[0][0].shape)

    f.close()


if __name__ == '__main__':
    main()
