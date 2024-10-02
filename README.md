# ML_Sports_Predictions
---
## ML Model for Player-Prop Betting

`ML_Sports_Predictions` is a machine learning model that predicts NBA players' future stats in upcoming games and compares it to given sportsbook predictions, to give predictions on best value sportsbook stat prediction lines.
---
## Getting Started
`ML_Sports_Predictions` requires Python 3.7+ along with `pandas`, `lightgbm`, and `scikit-learn` modules.

```Python
pip install pandas lightgbm scikit-learn
```
---
## Steps to Run Model
    ### Step 1:
        Run `call_endpt_CURRENT.py`. Code from: `https://github.com/swar/nba_api`. The NBA_API github has different APIs that you can call for a variety of data on NBA players and teams for model adjustment and feature changes.
    ### Step 2: 
        Process Data using `Luka_data_CURRENT.py`. Currently, the data processes Nikola Jokic's data from his full career. 
    ### Step 3:
        Run model training using `training_CURRENT.py`. 

## Current Results
    At the moment, model outputs poor results and requires extra feature experimentation and implementation

## Future Work
    1. Expand past using data only for one player and expand from just using NBA data. NFL data would be most interesting because it is the most sparse and hardest to train on (E.g. players have a max of 100 games in their career)
    2. Try different models and model types to figure out which combination would be the most accurate.






Currently only on data gathering.

Step 0: Create a python venv, and pip install pandas, nba-api that is found here: https://github.com/swar/nba_api, scikit-learn, and lightgbm

Step 1: Follow steps in nba_api to get a player's game stats. Use 'call_endpt_CURRENT.py' to scrape data from NBA API. Then use 'Luka_data_CURRENT.py' for creating labels and projections.

Step 2: TODO: train model. Currently in process of training Lightgbm model.