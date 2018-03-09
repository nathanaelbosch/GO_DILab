import sqlite3
import pandas as pd


DB_PATH = 'data/db.sqlite'
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute('''SELECT count(*)
             FROM elo_ordered_games''')
nrows = c.fetchone()[0]

c.execute('''
    ALTER TABLE elo_ordered_games
    ADD COLUMN rand_id INT''')
