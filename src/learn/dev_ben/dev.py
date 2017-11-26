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

cols = rows = 9


def print_matrix(m):
    matrix_str = ''
    for row in range(0, rows):
        row_str = ''
        for col in range(0, cols):
            row_str += str(m[col][row]) + ' '
        matrix_str += row_str + '\n'
    print(matrix_str)


def serialize_matrix(m):
    entries = []
    for row in range(0, rows):
        for col in range(0, cols):
            entries.append(m[col][row])
    return entries


empty_val = 0  # -1.35, 0.45, 1.05
matrix = [[empty_val for x in range(rows)] for y in range(cols)]
matrix[1][1] = 1
print_matrix(matrix)

for move in moves:
    player_color = list(move.properties.keys())[0]
    sgf_move = str(move.properties[player_color][0])
    # print(player_color, sgf_move)
