"""Replay game from sgf, then generate the training data out of it

On each move we can create a new datapoint, or even 8 adding symmetries!
"""
import os
import sgf
import random as rn
from src.play.model.Game import Game
from src.play.utils.Move import Move


OUT_PATH = 'data/training_data/simplest_move_prediction/'


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
    game_id = n_0.properties['GN'][0]

    game = Game(n_0.properties)
    for n in game_tree.nodes[1:]:
        player_color = list(n.properties.keys())[0]
        move = Move.from_sgf(str(n.properties[player_color][0]))
        game.play(move, player_color.lower())
        func(game_id+'.csv', game, move, player_color.lower())


def get_string(game, move, player):
    ls = game.board.tolist()
    ls = [str(entry) for row in ls for entry in row]
    board_string = ';'.join(ls)

    if move.is_pass:
        # s = ';'.join(['0']*(9*9)+['1'])
        move_string = ';'.join(['0']*(9*9))
    else:
        new_game = Game()
        new_game.play(move, player.lower())
        ls = new_game.board.tolist()
        ls = [str(entry) for row in ls for entry in row]
        # s = ';'.join(ls+['0'])
        move_string = ';'.join(ls)

    s = board_string + ';' + move_string
    return s


def to_training_file(out, *args, **kwargs):
    out = os.path.join(OUT_PATH, out)
    with open(out, 'a') as f:
        f.write(get_string(*args, **kwargs))
        f.write('\n')


def foo(file):
    try:
        replay_game(file, to_training_file)
    except Exception:
        return


def main():
    files = list_all_sgf_files('data')
    # files = rn.sample(files, 1000)

    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(foo, files)


if __name__ == '__main__':
    main()
