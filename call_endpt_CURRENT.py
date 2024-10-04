from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll


# Nikola Jokić
print("happened0")
career = playergamelog.PlayerGameLog(player_id=203999, season=SeasonAll.all) 
print("happened")

# pandas data frames (optional: pip install pandas)
df = career.get_data_frames()[0]
print(df.head())
print("happened2")
df.to_csv('check4.csv')
print("happened3")


# # json
# career.get_json()

# # dictionary
# career.get_dict()


