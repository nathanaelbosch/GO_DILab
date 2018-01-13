import os
import sqlite3
from os.path import dirname, abspath


project_root_dir = dirname(dirname(dirname(abspath(__file__))))
db_path = os.path.join(project_root_dir, 'data', 'db.sqlite')

db = sqlite3.connect(db_path)
cursor = db.cursor()

# to get not just count but the rows of the games-table, use:
# SELECT games.*
# FROM games, meta
# WHERE games.id == meta.id
# ...

command = '''
    SELECT COUNT(*)
    FROM meta
    WHERE meta.all_moves_imported IS NOT 0
    AND meta.elo_black IS NOT ""
    AND meta.elo_black > 2000
    AND meta.elo_white IS NOT ""
    AND meta.elo_white > 2000
    AND meta.turns > 30
'''

result = cursor.execute(command).fetchall()
print(result)
