import os
import sgf
import sys
import glob
import time
import string
import sqlite3
from os.path import dirname, abspath

project_root_dir = dirname(dirname(dirname(abspath(__file__))))  # GO_DILab
sys.path.append(project_root_dir)
from src.play.model.Board import Board, EMPTY, BLACK, WHITE


data_dir = os.path.join(project_root_dir, 'data')

db_name = 'db.sqlite'
db_path = os.path.join(data_dir, db_name)

if os.path.exists(db_path):
    print('connecting to existing db: ' + db_name)
    os.remove(db_path)
else:
    print('creating new db and connecting to it: ' + db_name)

db = sqlite3.connect(db_path)
cursor = db.cursor()

flat_matrix_table_column_names = []
for row in range(0, 9):
    for col in range(0, 9):
        flat_matrix_table_column_names.append('loc_{}_{}_{}'.format(row, col, row*9 + col))


def setup():
    print('creating tables meta and games')
    cursor.execute(
        '''CREATE TABLE meta(id INT PRIMARY KEY,
                             all_moves_imported INT,
                             size INT,
                             rules TEXT,
                             turns INT,
                             komi REAL,
                             rank_black TEXT,
                             rank_white TEXT,
                             result TEXT,
                             sgf_content TEXT)''')
    db.commit()
    cursor.execute('CREATE TABLE games(id INTEGER, color INTEGER, move INTEGER'
                   + ''.join([', ' + _str + ' INTEGER' for _str in flat_matrix_table_column_names]) + ')')
    db.commit()


def game_to_database(sgf_content, game_id):
    """Replay game and save to database

    No real changes here, just in a function for convenience, as we want
    to apply it both to the single sgf files given by Bernhard and the
    one big file from Nath
    """
    table_insert_command = '''INSERT INTO games
                              VALUES({})'''.format(','.join(['?']*84))

    collection = sgf.parse(sgf_content)
    game_tree = collection.children[0]
    game_properties = game_tree.nodes[0].properties
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
            _row = string.ascii_lowercase.index(sgf_move[1])
            _col = string.ascii_lowercase.index(sgf_move[0])
            flat_move = _row * 9 + _col

        values = [game_id, player_val, flat_move]
        flat_matrix = [val for _row in board.tolist() for val in _row]
        values.extend(flat_matrix)
        cursor.execute(table_insert_command, values)

        # Apply move only at the end
        if flat_move != -1:
            board.place_stone_and_capture_if_applicable_default_values((_row, _col), player_val)

    size = int(game_properties['SZ'][0])
    rules = game_properties['RU'][0]
    komi = float(game_properties['KM'][0])
    rank_black = game_properties['BR'][0]
    rank_white = game_properties['WR'][0]
    result = game_properties['RE'][0]

    # Insert some data about this game into the `meta` table
    cursor.execute('''INSERT INTO meta(id,
                                       all_moves_imported,
                                       size,
                                       rules,
                                       turns,
                                       komi,
                                       rank_black, 
                                       rank_white,
                                       result,
                                       sgf_content)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (game_id,
                    all_moves_imported,
                    size,
                    rules,
                    len(moves),
                    komi,
                    rank_black,
                    rank_white,
                    result,
                    sgf_content))
    db.commit()


def import_data():
    sgf_dir = os.path.join(data_dir, 'dgs')
    if not os.path.exists(sgf_dir):
        print(sgf_dir + ' does not exist')
        exit(1)
    sgf_files = glob.glob(os.path.join(sgf_dir, '*'))

    lines = []
    full_file_path = os.path.join(data_dir, 'full_file.txt')
    if os.path.isfile(full_file_path):
        with open(full_file_path) as f:
            lines = f.readlines()

    total_lengths = len(lines) + len(sgf_files)

    if len(sgf_files) is 0:
        print('no sgf files in ' + sgf_dir)
        exit(1)

    print('importing {} sgf-files into {}...'.format(
        total_lengths, db_name))

    start_time = time.time()

    def print_time_info(k, game_id):
        """Print info on elapsed and remaining time

        Thought it would be cleaner as a function, and I can use
        it in both loops"""
        if k == 0:  # would cause a ZeroDivisionError
            return
        elapsed_time = time.time() - start_time
        time_remaining = ((elapsed_time / k) *
                          (total_lengths - k))

        print('{}\t{}/{}\t{:.2f}%\t{:.0f}s elapsed\t~{:.0f}s remaining'.format(
            game_id, k, total_lengths,
            (k / total_lengths) * 100,
            elapsed_time, time_remaining))

    # import full_file.txt, 344374 games merged into one file by Nath
    for i, line in enumerate(lines):
        game_id = 1000000 + i
        print_time_info(i, game_id)
        game_to_database(lines[i], game_id)

    # import 76440 .sgf games from the dgs-folder, from Bernhard
    for j, path in enumerate(sgf_files):
        # not ignoring errors caused UnicodeDecodeError: 'ascii' codec can't decode byte 0xf6
        with open(path, 'r', errors='ignore') as f:
            sgf_content = f.read()
        filename = os.path.basename(path)
        game_id = int(filename.split('_')[1][:-4])  # get x in game_x.sgf
        print_time_info(j + len(lines), game_id)
        game_to_database(sgf_content, game_id)


setup()
import_data()

db.close()
