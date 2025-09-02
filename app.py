import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("📈 Stock Trend Prediction App")

# Sidebar input
st.sidebar.header("Configuration")
user_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# Download data
df = yf.download(user_input, start=start_date, end=end_date)

# Show raw data
st.subheader("Raw Data")
st.write(df.tail())

# Visualization
st.subheader("Closing Price Over Time")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'])
st.pyplot(fig)

# Prepare data for prediction
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

training_size = int(len(scaled_data) * 0.70)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Load trained model (make sure you saved your model as "model.h5")
model = load_model("model.h5")

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Plot predictions vs actual
st.subheader("Predictions vs Actuals")

look_back = time_step
trainPredictPlot = np.empty_like(scaled_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(scaled_data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2):len(scaled_data), :] = test_predict

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(scaler.inverse_transform(scaled_data), label="Original")
ax2.plot(trainPredictPlot, label="Train Prediction")
ax2.plot(testPredictPlot, label="Test Prediction")
ax2.legend()
st.pyplot(fig2)

# Forecasting the next 30 days
st.subheader("Next 30 Days Forecast")

x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input = list(x_input[0])
output = []

for i in range(30):  
    x_input = np.array(temp_input[-time_step:]).reshape(1,time_step,1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    output.append(yhat[0][0])

forecast = scaler.inverse_transform(np.array(output).reshape(-1,1))

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(forecast, label="Forecast")
ax3.legend()
st.pyplot(fig3)

