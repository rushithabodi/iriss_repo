import pandas as pd
import joblib

# Load trained model
model = joblib.load('iris_knn_model.pkl')

# Sample data to predict
sample = pd.DataFrame({
    'sepal_length': [6.1],
    'sepal_width': [2.9],
    'petal_length': [4.7],
    'petal_width': [1.4]
})

# Make prediction
prediction = model.predict(sample)[0]
print(f"🌸 Predicted Iris class: {prediction}")
