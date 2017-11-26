import os
import sgf
from os.path import dirname, abspath
from src.play.model.Board import Board

size = 9
EMPTY_val = 'E'  # 0.45
BLACK_val = 'B'  # -1.35
WHITE_val = 'W'  # 1.05

data_dir = os.path.join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'data')
sgf_files = [
    os.path.join(data_dir, 'game_57083.sgf'),
    os.path.join(data_dir, 'game_100672.sgf'),
    os.path.join(data_dir, 'some_game.sgf'),
]

training_data_dir = os.path.join(data_dir, 'training_data')
if not os.path.exists(training_data_dir):  # create the folder if it does not exist yet
    os.makedirs(training_data_dir)

for path in sgf_files:
    sgf_file = open(path, 'r')
    training_data_file = open(os.path.join(training_data_dir, os.path.basename(path) + '.csv'), 'w')
    collection = sgf.parse(sgf_file.read())
    game_tree = collection.children[0]
    moves = game_tree.nodes[1:]
    # meta = game_tree.nodes[0].properties
    # see SGF properties here: www.red-bean.com/sgf/properties.html

    board = Board([[EMPTY_val] * size] * size)
    lines = [board.matrix2csv()]

    for move in moves:
        keys = move.properties.keys()
        if 'B' not in keys and 'W' not in keys:  # don't know how to deal with special stuff yet
            continue
        # can't rely on the order in keys(), apparently must extract it like this
        player_color = 'B' if 'B' in move.properties.keys() else 'W'
        sgf_move = move.properties[player_color][0]
        append_to_previous_line = player_color + ':'
        if len(sgf_move) is 2:  # otherwise its a pass
            loc = ord(sgf_move[1]) - ord('a'), ord(sgf_move[0]) - ord('a')
            player_val = BLACK_val if player_color == 'B' else WHITE_val
            opponent_val = WHITE_val if player_color == 'B' else BLACK_val
            board.place_stone_and_capture_if_applicable(loc, player_val, opponent_val, EMPTY_val)
            append_to_previous_line += str(loc[0] * size + loc[1])
        else:
            append_to_previous_line += 'PASS'

        lines[len(lines) - 1] += ';' + append_to_previous_line
        lines.append(board.matrix2csv())
    lines[len(lines) - 1] += ';END'

    for line in lines:
        training_data_file.write(line + '\n')
    training_data_file.close()
