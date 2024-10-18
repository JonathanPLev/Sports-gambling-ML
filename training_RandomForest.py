import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# PREPARE DATA FOR LIGHTGBM:
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Luka_data_with_qualitative_features_updated2.csv")

num_lags = 20
features = [f'PTS_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [col for col in df.columns if 'MATCHUP_' in col] + \
           [f'FGM_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'FGA_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'AST_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           [f'REB_Lag_{lag}' for lag in range(1, num_lags + 1)] + \
           ['Month', 'isHomeGame', 'Projected_Line']

from sklearn.model_selection import train_test_split

df_subset = df.iloc[20:]
X = df_subset[features]
y = df_subset['Beats_Projected_Line']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print('ROC-AUC Score: ', roc_auc_score(y_test, y_prob))

importances = model.feature_importances_
feature_names = X.columns  # Assuming X is a DataFrame

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)



# # Plot feature importances
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
# plt.title('Feature Importances')
# plt.show()




