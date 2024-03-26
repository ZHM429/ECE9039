import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from itertools import product
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:/Users/zhang/BTC-Dataset.csv")

# Data preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Normalize the target feature first
target_features = 'Close'
target_scaler = MinMaxScaler()
df[target_features] = target_scaler.fit_transform(df[[target_features]])

# Normalize the additional features
additional_features = ['Open', 'High', 'Low', 'Volume', 'NASDAQ']
feature_scaler = MinMaxScaler()
df[additional_features] = feature_scaler.fit_transform(df[additional_features])

# Refine feature columns and target column after Normalization
additional_features = ['Open', 'High', 'Low', 'Volume', 'NASDAQ'] 
target_features = 'Close'

# Fill missing values with column averages
df.fillna(df.mean(), inplace=True)

# Function to create sequences with multiple features
def create_sequences(df, sequence_length, additional_features, target_features):
    X = []
    y = []
    for i in range(len(df) - sequence_length):
        X.append(df[additional_features].iloc[i:i+sequence_length].values)
        y.append(df[target_features].iloc[i + sequence_length])
    return np.array(X), np.array(y)

# Define sequence length and create sequences
sequence_length = 10
X, y = create_sequences(df, sequence_length, additional_features, target_features)

# Splitting data into training, validation, and test sets
train_set = int(0.70 * len(X))  
validation_set = int(0.10 * len(X))  
test_set = len(X) - train_set - validation_set 

X_train, X_val, X_test = X[:train_set], X[train_set:train_set+validation_set], X[train_set+validation_set:]
y_train, y_val, y_test = y[:train_set], y[train_set:train_set+validation_set], y[train_set+validation_set:]

# Reshape sequences for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(additional_features)))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], len(additional_features)))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(additional_features)))

# Define different hyperparameters for tuning
layer_sizes = [16, 32, 64]  
layer_numbers = [1, 2, 3]  
batch_sizes = [16, 32, 64]  
learning_rates = [0.0001, 0.001, 0.01]  

lowest_loss = float('inf')
best_model = None
best_config = {}

# Define the early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Grid search through hyperparameters with early stopping
for size, num, batch, lrs in product(layer_sizes, layer_numbers, batch_sizes, learning_rates):
    print(f"Training with Size: {size}, Num: {num}, Batch: {batch}, LR: {lrs}")

    model = Sequential()
    model.add(LSTM(size, activation='relu', return_sequences=True, input_shape=(sequence_length, len(additional_features))))
    for _ in range(num - 1):
        model.add(LSTM(size, activation='relu', return_sequences=True))
    model.add(LSTM(size, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=lrs), loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=10, batch_size=batch, 
                        validation_data=(X_val, y_val), verbose=0,
                        callbacks=[early_stopping])

    # Store validation loss for each configuration
    val_loss = history.history['val_loss'][-1]

    # Evaluate model by compare current loss and the lowest loss
    if val_loss < lowest_loss:
        lowest_loss = val_loss
        best_model = model
        best_config = {'Size': size, 'Num': num, 'Batch': batch, 'LR': lrs}
        
print("Best parameters:", best_config)

# Train the best model with early stopping
best_model.compile(optimizer=Adam(learning_rate=best_config['LR']), loss='mean_squared_error')
best_model.fit(X_train, y_train, epochs=10, batch_size=best_config['Batch'], 
               validation_data=(X_val, y_val), verbose=1, callbacks=[early_stopping])

# Get training and validation losses from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting training and validation losses
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training Progress: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Get LSTM predictions on the test set
lstm_predictions = best_model.predict(X_test).flatten()

# Reverse normalization for LSTM predictions to original scale
lstm_predictions_scaled = target_scaler.inverse_transform(lstm_predictions.reshape(-1, 1)).flatten()

# Reverse normalization for the actual test set values to original scale
y_test_scaled = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Plotting LSTM Predictions valus against Actual values
plt.figure(figsize=(14, 7))
plt.plot(y_test_scaled, label='Actual Values', color='blue')
plt.plot(lstm_predictions_scaled, label='LSTM Predictions', color='red', alpha=0.6)
plt.title('LSTM Model Predictions vs Actual Values')
plt.xlabel('Time')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

# Calculating evaluation metrics (MSE and R-Squared) for the LSTM model on the test set
mse_lstm = mean_squared_error(y_test_scaled, lstm_predictions_scaled)
r2_lstm = r2_score(y_test_scaled, lstm_predictions_scaled)

print("\nLSTM Model Test Metrics:")
print(f"Mean Squared Error (MSE): {mse_lstm}")
print(f"R-squared: {r2_lstm}")

# Calculate residuals
residuals_test = y_test_scaled - lstm_predictions_scaled

# Fit SVR model on the residuals
svr_model = SVR(C=1.0, epsilon=0.05)
svr_model.fit(np.arange(len(residuals_test)).reshape(-1, 1), residuals_test)

# Use the SVR model to predict adjustments
svr_adjustments = svr_model.predict(np.arange(len(residuals_test)).reshape(-1, 1))

# Apply SVR adjustments to LSTM predictions
hybrid_predictions_scaled = lstm_predictions_scaled + svr_adjustments

# Plotting Hybrid Model Predictions vs Actual values
plt.figure(figsize=(14, 7))
plt.plot(np.arange(len(y_test_scaled)), y_test_scaled, label='Actual Values', color='blue')
plt.plot(np.arange(len(hybrid_predictions_scaled)), hybrid_predictions_scaled, label='Hybrid Model Predictions', color='green', alpha=0.6)
plt.title('Hybrid Model (LSTM + SVR) Predictions vs Actual Values')
plt.xlabel('Time')
plt.ylabel('Scaled Value')
plt.legend()
plt.show()

# Calculating evaluation metrics (MSE and R-Squared) for Hybrid model on the test set
mse_hybrid = mean_squared_error(y_test_scaled, hybrid_predictions_scaled)
r_squared_hybrid = r2_score(y_test_scaled, hybrid_predictions_scaled)

print("\nHybrid Model Metrics:")
print(f"Test Mean Squared Error (MSE): {mse_hybrid}")
print(f"R-squared: {r_squared_hybrid}")