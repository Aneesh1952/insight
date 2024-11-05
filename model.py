import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

# Load the data
data = pd.read_csv("jmedia_insight.csv")



# Label encoding for categorical variables
label_encoders = {}
for column in ['Gender', 'Region', 'Interest_Tags', 'Subscription_Status']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Ensure that the columns for Average_Session_Duration and Click_Through_Rate are numeric
data['Average_Session_Duration'] = pd.to_numeric(data['Average_Session_Duration'], errors='coerce')
data['Click_Through_Rate'] = pd.to_numeric(data['Click_Through_Rate (CTR)'], errors='coerce')

# Define features and target
features = ['Age', 'Gender', 'Region', 'Interest_Tags', 'Average_Session_Duration', 'Click_Through_Rate', 'Pages_Viewed']
X = data[features]
y = data['Subscription_Status']

# Split data into train and test sets (not meaningful here due to small data, but included for completion)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a simple decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the model and encoders
with open("user_preference_model2.pkl", "wb") as f:
    pickle.dump({'model': model, 'encoders': label_encoders}, f)
