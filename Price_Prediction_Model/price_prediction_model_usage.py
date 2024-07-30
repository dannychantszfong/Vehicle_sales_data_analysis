import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

# Load the model from the file

model_path = "Price_Prediction_Model\scaler\car_price_prediction_model.h5"
loaded_model = tf.keras.models.load_model(model_path)
print(model_path)
# Load the scalers used during training
feature_scaler_path = r"Price_Prediction_Model\scaler\feature_scaler.pkl"
target_scaler_path = r"Price_Prediction_Model\scaler\target_scaler.pkl"
print(feature_scaler_path)
print(target_scaler_path)

feature_scaler = joblib.load(feature_scaler_path)
target_scaler = joblib.load(target_scaler_path)

# Example new data (should be preprocessed in the same way as the training data)
new_data = pd.DataFrame({
    'year': [2015, 2016, 2017, 2018],
    'make': ['Audi', 'BMW', 'Ford', 'Tesla'],
    'model': ['A3', 'X5', 'F150', 'Model S'],
    'trim': ['1.8 TFSI Premium', 'xDrive35i', 'Lariat', '75D'],
    'body': ['Sedan', 'SUV', 'Truck', 'Sedan'],
    'transmission': ['automatic', 'automatic', 'automatic', 'automatic'],
    'state': ['ca', 'ny', 'tx', 'ca'],
    'condition': [49, 50, 48, 47],
    'odometer': [5826, 15000, 30000, 20000],
    'color': ['gray', 'black', 'white', 'red'],
    'interior': ['black', 'beige', 'gray', 'white'],
    'seller': ['audi north scottsdale', 'bmw manhattan', 'ford dallas', 'tesla fremont'],
    'mmr': [24000, 45000, 30000, 80000]
})

# Define categorical and numerical columns
categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller']
numerical_columns = [col for col in new_data.columns if col not in categorical_columns]

# Standardize the numerical features using the same scaler as during training
new_data[numerical_columns] = feature_scaler.transform(new_data[numerical_columns])

# Convert categorical variables to the appropriate format
categorical_layers = []
for col in categorical_columns:
    vocab = new_data[col].unique()
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=vocab)
    cat_col_one_hot = tf.feature_column.indicator_column(cat_col)
    categorical_layers.append(cat_col_one_hot)

numerical_layers = [tf.feature_column.numeric_column(key=col) for col in numerical_columns]
feature_columns = categorical_layers + numerical_layers
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Preprocess the new data
def preprocess_new_data(new_data):
    new_data = {col: np.array(new_data[col]) for col in new_data.columns}
    return new_data

new_data_processed = preprocess_new_data(new_data)

# Create a TensorFlow dataset for the new data
def make_input_fn_new(data, batch_size=32):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((data))
        dataset = dataset.batch(batch_size)
        return dataset
    return input_function

new_input_fn = make_input_fn_new(new_data_processed)

# Make predictions
predictions = loaded_model.predict(new_input_fn())

# Flatten predictions
y_pred = np.array([pred[0] for pred in predictions])

# Inverse transform the target variable using the same scaler as during training
y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1))

# Convert to readable format (without scientific notation)
readable_predictions = [int(pred) for pred in y_pred_rescaled.flatten()]

print("Predictions:", readable_predictions)
