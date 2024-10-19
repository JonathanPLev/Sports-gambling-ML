from autogluon.tabular import TabularPredictor, TabularDataset

df = TabularDataset('Luka_data_with_qualitative_features_updated4.csv')


num_lags = 20
features = [f'PTS_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [col for col in df.columns if 'MATCHUP_' in col] + \
           [f'FGM_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'FGA_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'AST_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'REB_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           ['Month', 'isHomeGame', 'Projected_Line', 'W', 'L', 'PLUS_MINUS_opponent',
            'versus_team_id', 'Rest_Days_for_Previous_Game', 'Rest_Days', 'DaysUntilNextGame',
            'Minutes_LastGame']

train_data = df.sample(frac=.8, random_state=42)
test_data = df.drop(train_data.index)
train_data = train_data[features + ['Beats_Projected_Line']]
test_data = test_data[features + ['Beats_Projected_Line']]

label = 'Beats_Projected_Line'
predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(
    train_data=train_data,
    presets='best_quality'
    )