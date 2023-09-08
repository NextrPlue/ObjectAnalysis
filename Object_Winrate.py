import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
import warnings

warnings.filterwarnings(action='ignore')

st.title("Correlation between objects and win rate")
st.header("Test Header1")
st.header("Test Header2")

st.write("hi")

League = pd.read_csv('2023_LoL_esports_match_data_from_OraclesElixir.csv')

print(pd.unique(League.datacompleteness))
League = League[League['datacompleteness'] == 'complete']
print(pd.unique(League.datacompleteness))

print(pd.unique(League.position))
League = League[League['position'] == 'team']
print(pd.unique(League.position))

League = League[['teamname', 'result', 'firstdragon', 'firstherald', 'firstbaron', 'dragons', 'heralds', 'barons']]
League['dragon_buff'] = (League['dragons'] >= 4.0) * 1
League

League_Object = League.groupby('teamname').agg({'result':'mean'}).sort_values('result')
League_Object['count'] = League.groupby('teamname').agg({'result':'count'})
League_Object['firstdragon'] = League.groupby('teamname').agg({'firstdragon':'mean'})
League_Object['firstherald'] = League.groupby('teamname').agg({'firstherald':'mean'})
League_Object['firstbaron'] = League.groupby('teamname').agg({'firstbaron':'mean'})
League_Object['dragons'] = League.groupby('teamname').agg({'dragons' : 'mean'})
League_Object['heralds'] = League.groupby('teamname').agg({'heralds' : 'mean'})
League_Object['barons'] = League.groupby('teamname').agg({'barons' : 'mean'})
League_Object['dragon_buff'] = League.groupby('teamname').agg({'dragon_buff' : 'mean'})
League_Object.drop(League_Object[(League_Object['count'] < 20)].index, inplace=True)

League_Object

League_Object['firstdragon_win'] = League.drop(League[(League['firstdragon'] == 0)].index).groupby('teamname').agg({'result':'mean'})
League_Object['firstherald_win'] = League.drop(League[(League['firstherald'] == 0)].index).groupby('teamname').agg({'result':'mean'})
League_Object['firstbaron_win'] = League.drop(League[(League['firstbaron'] == 0)].index).groupby('teamname').agg({'result':'mean'})

League_Object

FirstObj_Win = pd.DataFrame({'type':['firstdragon', 'firstherald', 'firstbaron'],
                              'win_rate':[np.average(League_Object['firstdragon_win']), np.average(League_Object['firstherald_win']), np.average(League_Object['firstbaron_win'])]})

FirstObj_Win

fig = sb.lmplot(x="firstdragon", y="result", data=League_Object, line_kws={'color' : 'red'})
plt.show()

fig = sb.lmplot(x="firstherald", y="result", data=League_Object, line_kws={'color' : 'red'})
plt.show()

fig = sb.lmplot(x="firstbaron", y="result", data=League_Object, line_kws={'color' : 'red'})
plt.show()

df_long = pd.melt(League_Object, id_vars=['result'], value_vars=['firstdragon', 'firstherald', 'firstbaron'], 
                  var_name='Variable', value_name='Value')
sb.lmplot(x='Value', y='result', hue='Variable', data=df_long, height=8, aspect=1.2)
plt.title('Linear Relationship between firstdragon, firstherald, firstbaron and result')

plt.show()

FirstObj_Win = pd.DataFrame({
    'type': ['firstdragon', 'firstherald', 'firstbaron'],
    'win_rate': [
        np.average(League_Object['firstdragon_win']),
        np.average(League_Object['firstherald_win']),
        np.average(League_Object['firstbaron_win'])
    ]
})

sb.barplot(x='type', y='win_rate', data=FirstObj_Win)

plt.title('Average Win Rate for First Objectives')
plt.xlabel('Type of Objective')
plt.ylabel('Average Win Rate')

plt.show()

sb.lmplot(x='dragon_buff', y='result', data=League_Object, height=8, aspect=1.2, line_kws={'color' : 'red'})
plt.title('Linear Relationship between dragon_buff and result')

plt.show()

sb.lmplot(x='dragons', y='result', data=League_Object, height=8, aspect=1.2, line_kws={'color' : 'red'})
plt.title('Linear Relationship between dragon_kills and result')

plt.show()

sb.lmplot(x='heralds', y='result', data=League_Object, height=8, aspect=1.2, line_kws={'color' : 'red'})
plt.title('Linear Relationship between herald_kills and result')

plt.show()

sb.lmplot(x='barons', y='result', data=League_Object, height=8, aspect=1.2, line_kws={'color' : 'red'})
plt.title('Linear Relationship between baron_kills and result')

plt.show()

df = pd.melt(League_Object, id_vars=['result'], value_vars=['dragons', 'heralds', 'barons'], 
                   var_name='Variable', value_name='Value')
sb.lmplot(x='Value', y='result', hue='Variable', data=df, height=8, aspect=1.2)
plt.title('Linear Relationship between dragon_kills, herald_kills, baron_kills and result')

plt.show()