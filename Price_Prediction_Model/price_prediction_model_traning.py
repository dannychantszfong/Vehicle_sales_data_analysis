import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


file_path = r"Data\car_prices_clean.csv"

# Load the dataset
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(['sellingprice', 'vin', 'saledate'], axis=1)
y = data['sellingprice']

# Define categorical and numerical columns
categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 'state', 'color', 'interior', 'seller']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# Standardize the numerical features
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Scale the target variable
target_scaler = StandardScaler()
y = target_scaler.fit_transform(y.values.reshape(-1, 1))

# Save the scalers
scaler_path = r"Price_Prediction_Model\scaler\feature_scaler.pkl"
print(scaler_path)
joblib.dump(scaler, scaler_path)
target_scaler_path = r"Price_Prediction_Model\scaler\target_scaler.pkl"
print(target_scaler_path)
joblib.dump(target_scaler, target_scaler_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TensorFlow preprocessing layers for categorical columns
inputs = {col: tf.keras.layers.Input(name=col, shape=(), dtype='string') for col in categorical_columns}
inputs.update({col: tf.keras.layers.Input(name=col, shape=(), dtype='float32') for col in numerical_columns})

feature_columns = []

for col in categorical_columns:
    vocab = X_train[col].unique()
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=col, vocabulary_list=vocab)
    cat_col_one_hot = tf.feature_column.indicator_column(cat_col)
    feature_columns.append(cat_col_one_hot)

for col in numerical_columns:
    num_col = tf.feature_column.numeric_column(key=col)
    feature_columns.append(num_col)

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Create the Functional API model
dense_features = feature_layer(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(dense_features)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=inputs, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Create input function
def make_input_fn(X, y, num_epochs=100, shuffle=True, batch_size=64):
    def input_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(num_epochs)
        return dataset
    return input_function

# Train the model
train_input_fn = make_input_fn(X_train, y_train)
model.fit(train_input_fn(), epochs=10, steps_per_epoch=1000)

# Evaluate the model
test_input_fn = make_input_fn(X_test, y_test, num_epochs=1, shuffle=False)
predictions = model.predict(test_input_fn())

# Flatten predictions
y_pred = np.array([pred[0] for pred in predictions])

# Inverse transform the target variable
y_pred_rescaled = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = target_scaler.inverse_transform(y_test)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Visualize predictions
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_rescaled.flatten(), y=y_pred_rescaled.flatten(), alpha=0.6)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], [y_test_rescaled.min(), y_test_rescaled.max()], 'r--', linewidth=2)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs Predicted Selling Price')
plt.show()

# Save the model
model.save(r'Price_Prediction_Model\scaler\car_price_prediction_model.h5')
