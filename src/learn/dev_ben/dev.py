import os
import sgf
from os.path import dirname, abspath

data_dir = os.path.join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'data')
path = os.path.join(data_dir, 'some_game.sgf')

sgf_file = open(path, 'r')
collection = sgf.parse(sgf_file.read())
game_tree = collection.children[0]

meta = game_tree.nodes[0].properties
moves = game_tree.nodes[1:]

# see SGF properties here: www.red-bean.com/sgf/properties.html

for move in moves:
    player_color = list(move.properties.keys())[0]
    sgf_move = str(move.properties[player_color][0])

    print(player_color, sgf_move)
