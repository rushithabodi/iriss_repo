import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load iris data
df = pd.read_csv('data/iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Prepare features and labels
X = df.drop('class', axis=1)
y = df['class']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can change k value
knn.fit(X_train, y_train)

# Save model
joblib.dump(knn, 'iris_knn_model.pkl')
print("✅ KNN Iris model trained and saved as 'iris_knn_model.pkl'")
