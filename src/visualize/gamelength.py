import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DB_PATH = 'data/db.sqlite'
conn = sqlite3.connect(DB_PATH)


data = pd.read_sql_query('''SELECT turns, id
                            FROM meta''', conn)

# Remove the one single outlier:
data = data.loc[data.turns < 1000]

# General plot:
ax = sns.distplot(data.turns, kde=False)
ax.set_title('Number of turns per game')
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("plots/turns.png")
plt.clf()

# Only Bernhard's data:
ax = sns.distplot(data.loc[data.id < 1000000].turns, kde=False)
ax.set_title('Number of turns per game - Bernhard\'s data')
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("plots/turns_bernhard.png")
plt.clf()

# Only Nath's data:
ax = sns.distplot(data.loc[data.id > 1000000].turns, kde=False)
ax.set_title('Number of turns per game - Nath\'s data')
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("plots/turns_nath.png")
plt.clf()
