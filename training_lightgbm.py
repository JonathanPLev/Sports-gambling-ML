import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
# PREPARE DATA FOR LIGHTGBM:
import lightgbm as lgb
from lightgbm import early_stopping
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the CSV file
df = pd.read_csv('Luka_data_with_projections.csv')

# Create lag features for points (PTS)
num_lags = 5
for lag in range(1, num_lags + 1):
    df[f'PTS_Lag_{lag}'] = df['PTS'].shift(lag)

# Drop rows with NaN values (which will be the first 'num_lags' rows)
df = df.dropna()


# Encode the opponent team as a categorical feature
df = pd.get_dummies(df, columns=['MATCHUP'], drop_first=True)
print("one_hot_encoding complete")

# Extract month from the game date
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df['Month'] = df['GAME_DATE'].dt.month

# Encode the home/away games
df['isHomeGame'] = df['isHomeGame'].astype(int)


# Calculate lag features
for stat in ['PTS', 'FGM', 'FGA', 'AST', 'REB']:
    for lag in range(1, 6):
        df[f'{stat}_Lag_{lag}'] = df[stat].shift(lag)

# Calculate rolling averages
for stat in ['PTS', 'FGM', 'FGA', 'AST', 'REB']:
    df[f'{stat}_Rolling_Avg_5'] = df[stat].rolling(window=5).mean()


# Save the transformed data to a new CSV file
df.to_csv('Luka_data_with_qualitative_features_updated.csv', index=False)

# Define the features and target
features = [f'PTS_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [col for col in df.columns if 'MATCHUP_' in col] + \
           [f'FGM_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'FGA_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'AST_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'REB_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           ['Month', 'isHomeGame', 'Projected_Line']
target = 'Beats_Projected_Line'


# # Columns to drop
# drop_columns = ['Unnamed: 0.1', 'Unnamed: 0']

# # Check for existing columns and drop if found
# existing_columns = [col for col in drop_columns if col in df.columns]

# # Drop the existing columns if any are found
# if existing_columns:
#     df.drop(existing_columns, inplace=True)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Define the parameters for the LightGBM model
params = {
    'objective': 'binary', # regression or binary
    'metric': 'binary_logloss', # RMSE or binary_logloss
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the LightGBM model
model = lgb.train(
    params, 
    train_data, 
    valid_sets=[test_data], 
    num_boost_round=1000, 
    callbacks=[early_stopping(stopping_rounds=10)])


y_pred = model.predict(X_test)
# For classification
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy}')

# For regression
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')

# Save the model
# model.save_model('lightgbm_model_with_qualitative_features.txt')

# MAKE PREDICTIONS:
# # Load the model
# model = lgb.Booster(model_file='lightgbm_model_with_qualitative_features.txt')

# # Make predictions on the test set
# predictions = model.predict(X_test)

# # Evaluate the model
# from sklearn.metrics import mean_squared_error
# rmse = mean_squared_error(y_test, predictions, squared=False)
# print(f'RMSE: {rmse}')


    