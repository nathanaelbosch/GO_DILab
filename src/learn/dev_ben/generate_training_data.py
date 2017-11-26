import os
import sgf
from os.path import dirname, abspath
import numpy as np
from src.play.model.Board import Board

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


def serialize_matrix(m, invert_color=False):  # Board.matrix2csv(), but with inversion added
    ls = m.tolist()
    ls = [str(entry) for _row in ls for entry in _row]
    if not invert_color:
        return ';'.join(ls)
    for k, val in enumerate(ls):
        if val is BLACK_val:
            ls[k] = WHITE_val
        elif val is WHITE_val:
            ls[k] = BLACK_val
    return ';'.join(ls)


for path in sgf_files:
    sgf_file = open(path, 'r')
    training_data_file = open(os.path.join(training_data_dir, os.path.basename(path) + '.csv'), 'w')
    collection = sgf.parse(sgf_file.read())
    game_tree = collection.children[0]
    moves = game_tree.nodes[1:]
    # meta = game_tree.nodes[0].properties
    # see SGF properties here: www.red-bean.com/sgf/properties.html

    board = Board([[EMPTY_val] * 9] * 9)
    lines = []

    # for i in range(2, 3):
    for move in moves:
        # 8 symmetries (Dihedral group D4)

        original = board.copy()

        keys = move.properties.keys()
        if 'B' not in keys and 'W' not in keys:  # don't know how to deal with special stuff yet
            continue
        # can't rely on the order in keys(), apparently must extract it like this
        player_color = 'B' if 'B' in move.properties.keys() else 'W'
        sgf_move = move.properties[player_color][0]

        loc = None
        if len(sgf_move) is 2:  # otherwise its a pass
            loc = ord(sgf_move[1]) - ord('a'), ord(sgf_move[0]) - ord('a')
            player_val = BLACK_val if player_color == 'B' else WHITE_val
            opponent_val = WHITE_val if player_color == 'B' else BLACK_val
            board.place_stone_and_capture_if_applicable(loc, player_val, opponent_val, EMPTY_val)

        loc_str = 'PASS'

        if loc is not None:
            # from matrix coords to origin at 4/4
            row = loc[0]
            col = loc[1]
            y = 4 - row
            x = col - 4
            loc_str = str(loc[0] * 9 + loc[1])
        lines.append(serialize_matrix(original) + ';' + loc_str + ';original')
        lines.append(serialize_matrix(original, True) + ';' + loc_str + ';original_inv')

        rot90 = np.rot90(original)
        if loc is not None:  # 90° means x/y becomes -y/x
            x_rot = -y
            y_rot = x
            # back into matrix coords
            row_transf = 4 - y_rot
            col_transf = x_rot + 4
            loc_str = str(row_transf * 9 + col_transf)
        lines.append(serialize_matrix(rot90) + ';' + loc_str + ';90ccw')
        lines.append(serialize_matrix(rot90, True) + ';' + loc_str + ';90ccw_inv')

        rot180 = np.rot90(rot90)
        if loc is not None:  # 180° means x/y becomes -y/x
            row_transf = 8 - row
            col_transf = 8 - col
            loc_str = str(row_transf * 9 + col_transf)
        lines.append(serialize_matrix(rot180) + ';' + loc_str + ';180')
        lines.append(serialize_matrix(rot180, True) + ';' + loc_str + ';180_inv')

        rot270 = np.rot90(rot180)
        if loc is not None:   # 270° means x/y becomes y/-x
            x_rot = y
            y_rot = -x
            row_transf = 4 - y_rot
            col_transf = x_rot + 4
            loc_str = str(row_transf * 9 + col_transf)
        lines.append(serialize_matrix(rot270) + ';' + loc_str + ';270ccw')
        lines.append(serialize_matrix(rot270, True) + ';' + loc_str + ';270ccw_inv')

        hflip = np.fliplr(original)
        if loc is not None:  # invert col
            row = row
            col = 8 - col
            y = 4 - row
            x = col - 4
            loc_str = str(row * 9 + col)
        lines.append(serialize_matrix(hflip) + ';' + loc_str + ';hflip')
        lines.append(serialize_matrix(hflip, True) + ';' + loc_str + ';hflip_inv')

        hflip_rot90 = np.rot90(hflip)
        if loc is not None:
            x_rot = -y
            y_rot = x
            row_transf = 4 - y_rot
            col_transf = x_rot + 4
            loc_str = str(row_transf * 9 + col_transf)
        lines.append(serialize_matrix(hflip_rot90) + ';' + loc_str + ';hflip90ccw')
        lines.append(serialize_matrix(hflip_rot90, True) + ';' + loc_str + ';hflip90ccw_inv')

        hflip_rot180 = np.rot90(hflip_rot90)
        if loc is not None:
            row_transf = 8 - row
            col_transf = 8 - col
            loc_str = str(row_transf * 9 + col_transf)
        lines.append(serialize_matrix(hflip_rot180) + ';' + loc_str + ';hflip180')
        lines.append(serialize_matrix(hflip_rot180, True) + ';' + loc_str + ';hflip180_inv')

        hflip_rot270 = np.rot90(hflip_rot180)
        if loc is not None:
            x_rot = y
            y_rot = -x
            row_transf = 4 - y_rot
            col_transf = x_rot + 4
            loc_str = str(row_transf * 9 + col_transf)
        lines.append(serialize_matrix(hflip_rot270) + ';' + loc_str + ';hflip270ccw')
        lines.append(serialize_matrix(hflip_rot270, True) + ';' + loc_str + ';hflip270ccw_inv')

        lines.append('')

    for line in lines:
        training_data_file.write(line + '\n')
    training_data_file.close()
