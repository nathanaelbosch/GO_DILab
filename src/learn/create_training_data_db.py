import glob
import sqlite3
import os
import string
from os.path import dirname, abspath
import sgf

from src.play.model.Board import Board, EMPTY, BLACK, WHITE


project_root_dir = dirname(dirname(dirname(abspath(__file__))))  # GO_DILab
data_dir = os.path.join(project_root_dir, 'data')

db = sqlite3.connect(os.path.join(data_dir, 'db.sqlite'))
cursor = db.cursor()

flat_matrix_table_column_names = []
for row in range(0, 9):
    for col in range(0, 9):
        flat_matrix_table_column_names.append('loc_' + str(row) + '_' + str(col) + '_' + str(row * 9 + col))


def setup():
    cursor.execute('CREATE TABLE meta(id INTEGER PRIMARY KEY, all_moves_imported INTEGER, sgf_content TEXT)')
    db.commit()
    cursor.execute('CREATE TABLE games(id INTEGER, color INTEGER, move INTEGER'
                   + ''.join([', ' + _str + ' INTEGER' for _str in flat_matrix_table_column_names]) + ')')
    db.commit()


def import_data():
    sgf_dir = os.path.join(data_dir, 'dgs')
    if not os.path.exists(sgf_dir):
        print(sgf_dir + ' does not exist')
        exit(1)
    sgf_files = glob.glob(os.path.join(sgf_dir, '*'))
    if len(sgf_files) is 0:
        print('no sgf files in ' + sgf_dir)
        exit(1)

    table_insert_command = 'INSERT INTO games(id, color, move' \
                           + ''.join([', ' + _str for _str in flat_matrix_table_column_names]) + ') ' \
                           + 'VALUES(' + ''.join(['?,' for i in range(0, 84)])[:-1] + ')'

    for i, path in enumerate(sgf_files):
        if i > 5: break  # dev-restriction

        # not ignoring errors caused UnicodeDecodeError: 'ascii' codec can't decode byte 0xf6
        sgf_file = open(path, 'r', errors='ignore')  # via stackoverflow.com/a/12468274/2474159
        filename = os.path.basename(path)
        game_id = int(filename.split('_')[1][:-4])  # get x in game_x.sgf
        sgf_content = sgf_file.read().replace('\n', '')
        sgf_file.close()
        collection = sgf.parse(sgf_content)
        game_tree = collection.children[0]
        moves = game_tree.nodes[1:]
        board = Board([[EMPTY] * 9] * 9)

        all_moves_imported = True

        for j, move in enumerate(moves):
            keys = move.properties.keys()
            if 'B' not in keys and 'W' not in keys:  # don't know how to deal with special stuff (yet?)
                all_moves_imported = False
                break  # or just continue? would are the moves afterwards still definitely be useful? not sure
            # can't rely on the order in keys(), apparently must extract it like this
            player_color = 'B' if 'B' in move.properties.keys() else 'W'
            player_val = BLACK if player_color == 'B' else WHITE
            sgf_move = move.properties[player_color][0]

            flat_move = -1
            if len(sgf_move) is 2:  # otherwise its a pass
                row = string.ascii_lowercase.index(sgf_move[1])
                col = string.ascii_lowercase.index(sgf_move[0])
                move = row, col
                flat_move = row * 9 + col
                board.place_stone_and_capture_if_applicable_default_values(move, player_val)

            flat_matrix = [val for _row in board.tolist() for val in _row]
            values = [game_id, player_val, flat_move]
            values.extend(flat_matrix)
            cursor.execute(table_insert_command, values)

        cursor.execute('INSERT INTO meta(id, all_moves_imported, sgf_content) VALUES(?,?,?)',
                       (game_id, all_moves_imported, sgf_content))
        db.commit()


setup()
import_data()

db.close()
