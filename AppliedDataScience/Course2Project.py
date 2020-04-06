import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df1 = pd.read_excel('10s0077.xls', usecols='A,D', skiprows=12, skipfooter=16)
df1.columns = ['region', 'data']
df1 = df1.set_index('region')
df2 = pd.read_excel('10s0019.xls', usecols='A,D,E,M', skiprows=15, skipfooter=8)
df2.columns = ['region', 'total population',
               'white population', 'white percentage']
df2 = df2.set_index('region')
df = df1.merge(df2, left_index=True, right_index=True)
df['christian percentage'] = (df['data'] / df['total population']) * 100

df = df.reset_index()
total = df.loc[0]
df = df.drop(0)
mi = df.loc[23]
x = df['white percentage']
y = df['christian percentage']

plt.figure()
gspec = gridspec.GridSpec(3, 3)

main = plt.subplot(gspec[:, 1:])
main.axvline(total['white percentage'], color='tab:gray', alpha = 0.9)
main.axhline(total['christian percentage'], color='tab:gray', alpha = 0.9)
main.scatter(x, y, c='tab:gray', alpha=0.5)
main.scatter(mi['white percentage'], mi['christian percentage'], c='b')
main.annotate('MI', (mi['white percentage'], mi['christian percentage']))
main.legend(['US Average'], frameon=False)
main.spines['left'].set_visible(False)
main.spines['top'].set_visible(False)
main.yaxis.set_label_position("right")
main.yaxis.tick_right()
plt.xlabel('White Population %')
plt.ylabel('Christian Church Adherents %')

left = plt.subplot(gspec[:, 0])
left.hist(y, bins=30, orientation='horizontal', alpha=0.7)
left.invert_xaxis()
left.spines['right'].set_visible(False)
left.spines['top'].set_visible(False)
plt.xlabel('Number of States')
plt.ylabel('Christian Church Adherents %')

fig = plt.gcf()
fig.suptitle('''Correlation between Christian Church Adherents %
and White Population % of 50 states in the U.S.''')
plt.show()

