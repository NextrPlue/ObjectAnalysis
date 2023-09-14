import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings

from sklearn.linear_model import LinearRegression

League = pd.read_csv(f"2018_LoL_esports_match_data_from_OraclesElixir.csv")

League = League[League['datacompleteness'] == 'complete']
League = League[League['position'] == 'team']
League = League[['teamname', 'league', 'result', 'firstdragon', 'firstherald', 'infernals', 'mountains', 'clouds', 'oceans', 'chemtechs', 'hextechs', 'dragons', 'heralds', 'firsttower']]
League['dragon_buff'] = (League['dragons'] >= 4.0) * 1
League['infernal_buff'] = ((League['infernals'] >= 2.0) & League['dragon_buff']) * 1
League['mountain_buff'] = ((League['mountains'] >= 2.0) & League['dragon_buff']) * 1
League['cloud_buff'] = ((League['clouds'] >= 2.0) & League['dragon_buff']) * 1
League['ocean_buff'] = ((League['oceans'] >= 2.0) & League['dragon_buff']) * 1
League['chemtech_buff'] = ((League['chemtechs'] >= 2.0) & League['dragon_buff']) * 1
League['hextech_buff'] = ((League['hextechs'] >= 2.0) & League['dragon_buff']) * 1
League['herald_firsttower'] = ((League['heralds'] > 0) & League['firsttower']) * 1

League.to_csv("2018.csv")