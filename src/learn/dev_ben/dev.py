import os
import sgf
from os.path import dirname, abspath
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


data_dir = os.path.join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'data')
path = os.path.join(data_dir, 'some_game.sgf')

sgf_file = open(path, 'r')
collection = sgf.parse(sgf_file.read())
game_tree = collection.children[0]

meta = game_tree.nodes[0].properties
moves = game_tree.nodes[1:]

# see SGF properties here: www.red-bean.com/sgf/properties.html

cols = rows = 9


def print_matrix(m, gtp_format=False):
    matrix_str = ''
    for row in range(0, rows):
        row_str = ''
        for col in range(0, cols):
            row_str += str(m[col][row] if gtp_format else m[row][col]) + ' '
        matrix_str += row_str + '\n'
    print(matrix_str)


def serialize_matrix(m):
    entries = []
    for row in range(0, rows):
        for col in range(0, cols):
            entries.append(m[row][col])
    return entries


EMPTY = 0.45
BLACK = -1.35
WHITE = 1.05
matrix = [[EMPTY for x in range(cols)] for y in range(rows)]
out = [[0 for x in range(cols)] for y in range(rows)]

for i in range(0, 10):
    move = moves[i]
    # can't rely on the order in keys(), apparently must extract it like this:
    player_color = 'B' if 'B' in move.properties.keys() else 'W'
    sgf_move = move.properties[player_color][0]
    print(sgf_move, i)

    if len(sgf_move) == 2:
        # .sgf coords are col/row, we want row/col
        col = ord(sgf_move[1]) - ord('a')
        row = ord(sgf_move[0]) - ord('a')
        if i == 9:
            out[row][col] = 1
            break
        matrix[row][col] = BLACK if player_color == 'B' else WHITE
    else:
        # is pass, no idea yet what to do with it
        pass


print_matrix(matrix, True)
inp = serialize_matrix(matrix)
print(inp)
print_matrix(out, True)
outp = serialize_matrix(out)
print(outp)


X = np.array([
    inp,
])
Y = np.array([
    outp,
])

# set up network topology
model = Sequential()
dim = rows * cols
# first arg of Dense is # of neurons
model.add(Dense(162, input_dim=dim, activation='relu'))
# last layer = output layer, must have 81 again
model.add(Dense(dim, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=1)
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print(model.predict(X))
