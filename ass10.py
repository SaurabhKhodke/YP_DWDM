import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Load the dataset and use only the first 100 rows
df = pd.read_csv('healthcare_noshows.csv').head(100)

# Step 2: Data Preprocessing
# Convert ScheduledDay and AppointmentDay to datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Convert categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Neighbourhood'], drop_first=True)

# Convert boolean columns to integer (0, 1)
df['Scholarship'] = df['Scholarship'].astype(int)
df['Hipertension'] = df['Hipertension'].astype(int)
df['Diabetes'] = df['Diabetes'].astype(int)
df['Alcoholism'] = df['Alcoholism'].astype(int)
df['Handcap'] = df['Handcap'].astype(int)
df['SMS_received'] = df['SMS_received'].astype(int)

# Convert Showed_up column (target variable) to integer
df['Showed_up'] = df['Showed_up'].astype(int)

# Step 3: Define the features (X) and target (y)
X = df.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Showed_up'], axis=1)
y = df['Showed_up']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='gini')  # or 'entropy' for Information Gain
clf.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = clf.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n {cm}')

# Step 8: Visualize the Decision Tree
plt.figure(figsize=(12,8))
tree.plot_tree(clf, filled=True)
plt.show()