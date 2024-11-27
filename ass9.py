import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Data Collection
data = pd.read_csv('healthcare_noshows.csv') # Load your dataset here
data = data.fillna(0) # Fill missing values with 0 or an appropriate method

# Step 3: Exploratory Data Analysis (EDA)
print(data.head()) # Show the first few rows of the dataset
print(data.info()) # Get info on the data types and missing values
print(data.describe()) # Get statistical summary of the dataset

# Step 4: Feature Engineering
# Convert categorical variables to numerical if needed
# Assuming the relevant columns are categorical, we will convert them
data['Scholarship'] = data['Scholarship'].astype(int)
data['Hipertension'] = data['Hipertension'].astype(int)
data['Diabetes'] = data['Diabetes'].astype(int)
data['Alcoholism'] = data['Alcoholism'].astype(int)
data['Handcap'] = data['Handcap'].astype(int)
data['SMS_received'] = data['SMS_received'].astype(int)
data['Showed_up'] = data['Showed_up'].astype(int)# Target variable

# Step 5: Splitting the dataset into training and testing sets (Train-Test Split)
X = data[['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = data['Showed_up'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70% training, 30% testing

# Step 6: Model Training
model = GaussianNB() # Create an instance of the Gaussian Naive Bayes model
model.fit(X_train, y_train) # Fit the model on training data

# Step 7: Model Evaluation
y_pred = model.predict(X_test) # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
conf_matrix = confusion_matrix(y_test, y_pred) # Get confusion matrix
class_report = classification_report(y_test, y_pred) # Get classification report

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualization of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
 xticklabels=['Did not show', 'Showed up'], 
 yticklabels=['Did not show', 'Showed up'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Single User Input Testing
# Create a function for predicting a single user input
def predict_user_input(input_data):
    input_df = pd.DataFrame([input_data], columns=X.columns)
    prediction = model.predict(input_df)
    return "Showed Up" if prediction[0] == 1 else "Did Not Show Up"


user_input = [30, 0, 1, 0, 0, 0, 1] 
result = predict_user_input(user_input)
print(f'Prediction for user input {user_input}: {result}')
