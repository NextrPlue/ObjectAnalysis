import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the uploaded CSV file
League_Predict = pd.read_csv("2018.csv")

League_Predict['chemtechs'].fillna(0, inplace=True)
League_Predict['hextechs'].fillna(0, inplace=True)
League_Predict['infernals'].fillna(0, inplace=True)
League_Predict['mountains'].fillna(0, inplace=True)
League_Predict['clouds'].fillna(0, inplace=True)
League_Predict['oceans'].fillna(0, inplace=True)
League_Predict['dragons'].fillna(0, inplace=True)

columns_to_fill_mean = ['firstdragon', 'heralds', 'firsttower', 'firstherald']
for column in columns_to_fill_mean:
    mean_value = League_Predict[column].mean()
    League_Predict[column].fillna(mean_value, inplace=True)

# Check for missing values in the League_Predictset
missing_values_count = League_Predict.isnull().sum()

# Display columns with missing values, if any
missing_values_count[missing_values_count > 0]

print(missing_values_count)

features = [
    'firstdragon', 'firstherald', 'infernals', 'mountains', 'clouds', 'oceans', 
    'chemtechs', 'hextechs', 'dragons', 'heralds', 'firsttower', 'dragon_buff', 
    'infernal_buff', 'mountain_buff', 'cloud_buff', 'ocean_buff', 'chemtech_buff', 
    'hextech_buff', 'herald_firsttower'
]
target = 'result'

# Split the League_Predict into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(League_Predict[features], League_Predict[target], test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gradient_boosting_model.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = gradient_boosting_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
'''
# Filter the League_Predict to include only the matches involving the specified teams
team1_League_Predict = League_Predict[League_Predict['teamname'] == 'Liiv SANDBOX']
team2_League_Predict = League_Predict[League_Predict['teamname'] == 'T1']

# Calculate the mean statistics for each team
team1_mean_stats = team1_League_Predict[features].mean()
team2_mean_stats = team2_League_Predict[features].mean()

# Reshape the League_Predict to match the model's input shape
team1_mean_stats = team1_mean_stats.values.reshape(1, -1)
team2_mean_stats = team2_mean_stats.values.reshape(1, -1)

# Use the best model to predict the win probability for each team
team1_win_prob = gradient_boosting_model.predict_proba(team1_mean_stats)[:, 1]
team2_win_prob = gradient_boosting_model.predict_proba(team2_mean_stats)[:, 1]

# Calculate the normalized win probabilities for each team
total_prob = team1_win_prob + team2_win_prob
normalized_team1_win_prob = (team1_win_prob / total_prob) * 100
normalized_team2_win_prob = (team2_win_prob / total_prob) * 100

total_prob = team1_win_prob + team2_win_prob
normalized_team1_win_prob = (team1_win_prob / total_prob) * 100
normalized_team2_win_prob = (team2_win_prob / total_prob) * 100

print(normalized_team1_win_prob)
print(normalized_team2_win_prob)

'''
# 모델 저장하기
joblib.dump(gradient_boosting_model, '2018_model.joblib')
'''
# 모델 불러오기
loaded_model = joblib.load('gradient_boosting_model.joblib')
'''