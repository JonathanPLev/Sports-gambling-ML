ML_Sports_Predictions
Gathering data from gambling websites API's on player lines and using historical data to predict favorable lines that beat the bookies odds.

Currently only on data gathering.

Step 0: Create a python venv, and pip install pandas, nba-api that is found here: https://github.com/swar/nba_api, scikit-learn, and lightgbm

Step 1: Follow steps in nba_api to get a player's game stats. Use 'call_endpt_CURRENT.py' to scrape data from NBA API. Then use 'Luka_data_CURRENT.py' for creating labels and projections.

Step 2: TODO: train model. Currently in process of training Lightgbm model.