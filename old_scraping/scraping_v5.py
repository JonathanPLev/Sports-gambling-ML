import requests
from bs4 import BeautifulSoup
import pandas as pandas
import csv

footballPlayer = 'Daniel Jones'
url = 'https://www.pro-football-reference.com/players/J/JoneDa05'
seasonYear = ['/gamelog/2022/', 'gamelog/2021/', '/gamelog/2020/']


r = requests.get(url + seasonYear[0])

soup = BeautifulSoup(r.text,'html.parser')

season_table = soup.find('table', id="stats")

f = open('testGameData.csv', 'w') as f:
