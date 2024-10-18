import pandas as pd
import csv

df = pd.read_csv('check4.csv')
df['isHomeGame'] = df['MATCHUP'].str.contains('@')
df['isHomeGame'] = df['isHomeGame'].astype(int)

# print(df.head())
# df.to_csv('Luka_data_w_home_games.csv')

#TODO: player rest days, possibly distance to home? set up if player used to play for team. possibly rating of defender that would matchup against player? standing of team at moment that they play against each other. and stnading from last year