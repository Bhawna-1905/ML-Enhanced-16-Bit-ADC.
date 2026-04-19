import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1. LOAD YOUR DATA
df = pd.read_csv('training_data.csv')
print("✅ Columns found in CSV:", df.columns.tolist())

# MANUALLY SELECTING THE CORRECT COLUMN
# We use 'adc_value' because 'timestamp' is just the time!
data_col = 'adc_value' 
print(f"📊 Training on: '{data_col}'")

# --- HYBRID STEP: PRE-FILTERING ---
# Traditional Moving Average to clean the "low-hanging" noise
df['smoothed'] = df[data_col].rolling(window=5).mean().bfill()

# 2. PREPROCESSING
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['smoothed']].values)

def create_sequences(data, window=50):
    x_list, y_list = [], []
    for i in range(len(data) - window):
        x_list.append(data[i:i+window])
        y_list.append(data[i+window])
    return np.array(x_list), np.array(y_list)

# Defining X and y properly
X, y = create_sequences(scaled_data)

# 3. BUILD THE MODEL
# Using the new 'Input' layer to avoid that Keras UserWarning you saw
model = Sequential([
    Input(shape=(50, 1)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("🚀 Starting Training...")
model.fit(X, y, epochs=10, batch_size=32)

# 4. SAVE EVERYTHING
model.save('hybrid_lstm_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("\n✨ Done! Model and Scaler saved successfully.")
print("Now you can run your Streamlit app.py")