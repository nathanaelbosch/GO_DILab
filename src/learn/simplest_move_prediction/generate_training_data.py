"""Replay game from sgf, then generate the training data out of it

On each move we can create a new datapoint, or even 8 adding symmetries!
"""
import os
import sgf
import random as rn
from src.play.model.Game import Game
from src.play.utils.Move import Move


OUT = 'data/training_data/simplest_move_prediction/train.csv'


def list_all_sgf_files(dir):
    """List all sgf-files in a dir

    Recursively explores a path and returns the filepaths
    for all files ending in .sgf
    """
    root_dir = os.path.abspath(dir)
    sgf_files = []
    for root, sub_dirs, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(root, file)
            name, extension = os.path.splitext(path)
            if extension == '.sgf':
                sgf_files.append(path)
    return sgf_files


def replay_game(file, func):
    """Simply recreate a game from a sgf file

    More of a proof-of-concept or example than really a necessary function.
    We will use some modified version of this to create the training data.
    """
    with open(file, 'r') as f:
        content = f.read()
        collection = sgf.parse(content)

    # This all only works if the SGF contains only one game
    game_tree = collection.children[0]
    n_0 = game_tree.nodes[0]

    game = Game(n_0.properties)
    for n in game_tree.nodes[1:]:
        player_color = list(n.properties.keys())[0]
        move = Move.from_sgf(str(n.properties[player_color][0]))
        game.play(move, player_color.lower())
        func(game, move, player_color.lower())

def board_to_csv(board):
    ls = board.tolist()
    ls = [str(entry) for row in ls for entry in row]
    s = ';'.join(ls)
    return s

def move_to_csv(move, player):
    if move.is_pass:
        # s = ';'.join(['0']*(9*9)+['1'])
        s = ';'.join(['0']*(9*9))
        return s
    else:
        new_game = Game()
        new_game.play(move, player.lower())
        ls = new_game.board.tolist()
        ls = [str(entry) for row in ls for entry in row]
        # s = ';'.join(ls+['0'])
        s = ';'.join(ls)
        return s

def get_string(game, move, player):


    b = board_to_csv(game.board)
    m = move_to_csv(move, player)
    s = b+';'+m
    # print(s)
    return s


def foo(file):
    return replay_game(file, to_training_file)


def to_training_file(*args, **kwargs):
    with open(OUT, 'a') as f:
        f.write(get_string(*args, **kwargs))
        f.write('\n')


def main():
    files = list_all_sgf_files('data')
    files = rn.sample(files, 1000)

    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(foo, files)
    # map(lambda x: replay_game(x, to_training_file), files)

    # if os.path.isfile(OUT):
    #     os.remove(OUT)

    # for file in files:
    #     replay_game(file, to_training_file)


if __name__ == '__main__':
    main()
