import pandas as pd
import joblib

# Load model
model = joblib.load('iris_model.pkl')

# Sample input (change values as needed)
sample = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

# Predict
prediction = model.predict(sample)[0]
print(f"🌸 Predicted Iris class: {prediction}")
