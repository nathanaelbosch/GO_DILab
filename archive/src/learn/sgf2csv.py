import os
import sgf
from src.play.utils.Utils import str2move
from src.play.model.Game import Game

# root_dir = 'data/unpacked'

root_dir = os.path.abspath('../../example_data')
sgf_files = []
for root, sub_dirs, files in os.walk(root_dir):
    for file in files:
        path = os.path.join(root, file)
        name, extension = os.path.splitext(path)
        if extension == '.sgf':
            sgf_files.append(path)


for file in sgf_files:
    with open(file, 'r') as f:
        content = f.read()
        try:
            collection = sgf.parse(content)
        except Exception as e:
            print('Failed to parse ' + file + ' as sgf-collection')
            continue

    # Assume the sgf file contains one game
    game_tree = collection.children[0]
    n_0 = game_tree.nodes[0]
    # n_0.properties contains the initial game setup
    game_id = n_0.properties['GN'][0]
    out_file = os.path.join(root_dir, game_id + '.csv')
    if os.path.isfile(out_file):
        os.remove(out_file)

    # very similar to play_from_sgf.py, unify these parts TODO
    board_size = int(n_0.properties['SZ'][0])
    game = Game(n_0.properties, show_each_turn=True)
    for n in game_tree.nodes[1:]:
        player_color = list(n.properties.keys())[0]
        move_str = str(n.properties[player_color][0])
        move = str2move(move_str, board_size)
        game.play(move, player_color.lower())

    game.board2file(out_file, 'a')
