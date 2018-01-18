import sqlite3
import pandas as pd
import numpy as np
from tqdm import tqdm

DB_PATH = 'data/db.sqlite'
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()


print('[#1] Get number of rows')
c.execute('''SELECT count(*)
             FROM elo_ordered_games''')
nrows = c.fetchone()[0]

print('[#2] Create random ids')
rand_id = np.random.permutation(nrows)


print('[#3] Start loop')
rows_each_time = 100
start = 0
end = rows_each_time

i = 1
bar = tqdm(range(nrows//rows_each_time + 1))
for i in bar:
    print('Iteration', i)
    print('[{}] Get data'.format(i))
    data = pd.read_sql_query('''SELECT *
                                FROM elo_ordered_games
                                LIMIT ?, ?''',
                             conn,
                             params=[start, rows_each_time])

    print('[{}] Add ids'.format(i))
    print(data, data.shape)
    print(rand_id[start:end], len(rand_id[start:end]))
    data['rand_id'] = rand_id[start:end]

    print('[{}] Write table'.format(i))
    data.to_sql(
        'games_to_use', con=conn,
        if_exists='replace' if i==0 else 'append')

    start, end = end, end+rows_each_time
