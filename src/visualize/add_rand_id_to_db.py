import sqlite3
import pandas as pd
import numpy as np


DB_PATH = 'data/db.sqlite'
conn = sqlite3.connect(DB_PATH)

print('[#1] Get data')
data = pd.read_sql_query('''SELECT *
                            FROM elo_ordered_games''', conn)

print('[#2] Add ids')
data['rand_id'] = np.random.permutation(len(data))

print('[#3] Write table')
data.to_sql('elo_ordered_games', con=conn)

"""
c = conn.cursor()
c.execute('''SELECT count(*)
             FROM elo_ordered_games''')
nrows = c.fetchone()[0]

c.execute('''
    ALTER TABLE elo_ordered_games
    ADD COLUMN rand_id INT''')
conn.commit()
rand_ids = np.random.permutation(nrows)
"""