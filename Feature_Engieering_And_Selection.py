import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("Data/car_prices_clean.csv")

# Convert 'saledate' to datetime
df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')

# Feature Engineering
current_year = datetime.now().year
df['car_age'] = current_year - df['year']
df['mileage_per_year'] = df['odometer'] / df['car_age']

# Define the features and target variable
features = ['year', 'condition', 'odometer', 'mmr', 'car_age', 'mileage_per_year']
target = 'sellingprice'


# Standardize the numerical features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Split the data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to TensorFlow datasets
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)
    dataframe = dataframe.drop(columns=['saledate'])  # Ensure 'saledate' is not included
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

batch_size = 32
train_ds = df_to_dataset(df.sample(frac=0.8, random_state=42), batch_size=batch_size)
val_ds = df_to_dataset(df.drop(df.sample(frac=0.8, random_state=42).index), shuffle=False, batch_size=batch_size)

# Create TensorFlow feature columns
feature_columns = []

for feature_name in features:
    feature_columns.append(tf.feature_column.numeric_column(feature_name))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Build the model using Functional API
inputs = {colname: tf.keras.layers.Input(name=colname, shape=(), dtype='float32') for colname in features}
x = feature_layer(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=1)

# Debug: print the model summary to check layers
model.summary()

# Get feature importances
def get_feature_importance(model, feature_columns):
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}")

    dense_layer_weights = model.layers[7].get_weights()[0]  # Accessing the first dense layer weights
    feature_importances = {col.name: np.mean(np.abs(dense_layer_weights[:, i])) for i, col in enumerate(feature_columns)}
    return feature_importances

feature_importances = get_feature_importance(model, feature_columns)

# Visualize feature importance
feature_importance_df = pd.DataFrame({
    'Feature': list(feature_importances.keys()),
    'Importance': list(feature_importances.values())
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()

print(feature_importance_df)
