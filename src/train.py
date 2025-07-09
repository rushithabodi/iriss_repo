import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load iris dataset
df = pd.read_csv('iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Split into features and labels
X = df.drop('class', axis=1)
y = df['class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)  # You can tune this value
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save the trained model
joblib.dump(model, 'model/iris_model.pkl')

print("✅ KNN Iris model trained and saved to 'model/iris_model.pkl'.")
