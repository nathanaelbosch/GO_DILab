import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DB_PATH = 'data/db.sqlite'
conn = sqlite3.connect(DB_PATH)


data = pd.read_sql_query('''SELECT result_text
                            FROM meta
                            LIMIT 10''', conn)

results = data.result_text
