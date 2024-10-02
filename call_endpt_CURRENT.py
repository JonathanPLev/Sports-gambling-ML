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

'''
i need you to generate me some data based on a csv file that im going to send you. Can you do that if i give you instructions?


I am going to send you a csv file with a list of Jokic's games and his stats for those games for his career. I need you to create me a column for each game where there is a projected line (like a sporst betting line) based on his last games. Then, I want you to see if he beats that projected line that you created for that game based on his previous games, and i want you to set a 1 in a new column if he does. otherwise, set a 0. can you do that?




'''

