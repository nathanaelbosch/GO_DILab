"""Read a single SGF file and recreate the game"""
import sgf
import sys

sys.path.append('.')
from go import Game


file = 'data/unpacked/2017/09/08/338554.sgf'
with open(file, 'r') as f:
    content = f.read()
    collection = sgf.parse(content)

# Assume the sgf file contains one game
game_tree = collection.children[0]
n_0 = game_tree.nodes[0]
n_0.properties      # Contains the initial game setup

game = Game(n_0.properties, show_each_turn=True)
for node in game_tree.nodes[1:]:
    if 'W' in node.properties.keys():
        game.w(node.properties['W'][0])
    if 'B' in node.properties.keys():
        game.b(node.properties['B'][0])
