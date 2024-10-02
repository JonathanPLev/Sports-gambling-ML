import pandas as pd
import csv

df = pd.read_csv('check4.csv')
df['isHomeGame'] = df['MATCHUP'].str.contains('@')
df['isHomeGame'] = df['isHomeGame'].astype(int)

print(df.head())
df.to_csv('Luka_data_w_home_games.csv')