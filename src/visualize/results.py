import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


DB_PATH = 'data/db.sqlite'
conn = sqlite3.connect(DB_PATH)


data = pd.read_sql_query('''SELECT result_text
                            FROM meta''', conn)

results = data.result_text

# How many types of ends? Resigns, Time limits, or regular ends?
resigns = results.str.contains('Resign').sum()
time = results.str.contains('Time').sum()
played_till_the_end = len(results) - resigns - time
result_types = pd.DataFrame({
    'Number of games': [resigns, time, played_till_the_end],
    'Result Types': ['Resign', 'Overtime', 'Regular End']})
ax = sns.barplot(data=result_types, x='Result Types', y='Number of games')
ax.set_title('Numer of games per result type')
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("plots/result_types.png")
plt.clf()

# How are the result points distributed
points = pd.DataFrame(results.loc[
    (~results.str.contains('Resign')) & (~results.str.contains('Time'))])
points['absolute'] = pd.np.where(
    points.result_text=='Draw', 0,
    points.result_text.str.split('+').str.get(1))
points['absolute'] = points['absolute'].apply(pd.to_numeric, errors='coerce')

points['relative'] = pd.np.where(
    points.result_text.str.startswith('W'),
    -points.absolute,
    points.absolute)

points.dropna(inplace=True)

ax = sns.distplot(points.relative, kde=False)
ax.set_title('Game results')
ax.set_xlabel('Relative Points')
plt.tight_layout()
fig = ax.get_figure()
fig.savefig("plots/result_points.png")
plt.clf()
