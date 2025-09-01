# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # Title
# st.title("📈 Stock Trend Prediction App")

# # Sidebar input
# st.sidebar.header("Configuration")
# user_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# # Download data
# df = yf.download(user_input, start=start_date, end=end_date)

# # Show raw data
# st.subheader("Raw Data")
# st.write(df.tail())

# # Visualization
# st.subheader("Closing Price Over Time")
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(df['Close'])
# st.pyplot(fig)

# # Prepare data for prediction
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# training_size = int(len(scaled_data) * 0.70)
# train_data = scaled_data[:training_size]
# test_data = scaled_data[training_size:]

# def create_dataset(dataset, time_step=60):
#     X, y = [], []
#     for i in range(time_step, len(dataset)):
#         X.append(dataset[i-time_step:i, 0])
#         y.append(dataset[i, 0])
#     return np.array(X), np.array(y)

# time_step = 60
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, y_test = create_dataset(test_data, time_step)

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Load trained model (make sure you saved your model as "model.h5")
# model = load_model("model.h5")

# # Predictions
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)

# # Plot predictions vs actual
# st.subheader("Predictions vs Actuals")

# look_back = time_step
# trainPredictPlot = np.empty_like(scaled_data)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# testPredictPlot = np.empty_like(scaled_data)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2):len(scaled_data), :] = test_predict

# fig2, ax2 = plt.subplots(figsize=(12,6))
# ax2.plot(scaler.inverse_transform(scaled_data), label="Original")
# ax2.plot(trainPredictPlot, label="Train Prediction")
# ax2.plot(testPredictPlot, label="Test Prediction")
# ax2.legend()
# st.pyplot(fig2)

# # Forecasting the next 30 days
# st.subheader("Next 30 Days Forecast")

# x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
# temp_input = list(x_input[0])
# output = []

# for i in range(30):  
#     x_input = np.array(temp_input[-time_step:]).reshape(1,time_step,1)
#     yhat = model.predict(x_input, verbose=0)
#     temp_input.append(yhat[0][0])
#     output.append(yhat[0][0])

# forecast = scaler.inverse_transform(np.array(output).reshape(-1,1))

# fig3, ax3 = plt.subplots(figsize=(12,6))
# ax3.plot(forecast, label="Forecast")
# ax3.legend()
# st.pyplot(fig3)






























# # # stock_prediction_app.py
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import yfinance as yf
# # import streamlit as st
# # from keras.models import load_model
# # from sklearn.preprocessing import MinMaxScaler

# # # ------------------- STREAMLIT UI -------------------
# # st.title("Stock Trend Prediction App")

# # # Sidebar input
# # st.sidebar.header("Configuration")
# # user_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
# # start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
# # end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

# # # ------------------- DATA -------------------
# # df = yf.download(user_input, start=start_date, end=end_date)

# # st.subheader("Raw Data")
# # st.write(df.tail())

# # # ------------------- Historical Closing Price -------------------
# # st.subheader("Closing Price Over Time")
# # fig, ax = plt.subplots(figsize=(12,6))
# # ax.plot(df['Close'], label="Close Price")
# # ax.set_xlabel("Date")
# # ax.set_ylabel("Price USD")
# # ax.legend()
# # st.pyplot(fig)

# # # ------------------- Data Preprocessing -------------------
# # scaler = MinMaxScaler(feature_range=(0,1))
# # scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# # training_size = int(len(scaled_data) * 0.70)
# # train_data = scaled_data[:training_size]
# # test_data = scaled_data[training_size:]

# # def create_dataset(dataset, time_step=100):
# #     X, y = [], []
# #     for i in range(time_step, len(dataset)):
# #         X.append(dataset[i-time_step:i, 0])
# #         y.append(dataset[i, 0])
# #     return np.array(X), np.array(y)

# # time_step = 100
# # X_train, y_train = create_dataset(train_data, time_step)
# # X_test, y_test = create_dataset(test_data, time_step)

# # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # # ------------------- Load Model -------------------
# # model = load_model("model.h5")  # Make sure this exists

# # # ------------------- Predictions -------------------
# # train_predict = model.predict(X_train)
# # test_predict = model.predict(X_test)

# # train_predict = scaler.inverse_transform(train_predict)
# # test_predict = scaler.inverse_transform(test_predict)

# # # ------------------- Plot Train vs Test -------------------
# # st.subheader("Predictions vs Actuals")

# # look_back = time_step
# # trainPredictPlot = np.empty_like(scaled_data)
# # trainPredictPlot[:, :] = np.nan
# # trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# # testPredictPlot = np.empty_like(scaled_data)
# # testPredictPlot[:, :] = np.nan
# # testPredictPlot[len(train_predict)+(look_back*2):len(scaled_data), :] = test_predict

# # fig2, ax2 = plt.subplots(figsize=(12,6))
# # ax2.plot(scaler.inverse_transform(scaled_data), label="Original")
# # ax2.plot(trainPredictPlot, label="Train Prediction")
# # ax2.plot(testPredictPlot, label="Test Prediction")
# # ax2.legend()
# # st.pyplot(fig2)

# # # ------------------- Next 30 Days Forecast -------------------
# # st.subheader("Next 30 Days Forecast")

# # x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
# # temp_input = list(x_input[0])
# # output = []

# # for i in range(30):
# #     x_input = np.array(temp_input[-time_step:]).reshape(1,time_step,1)
# #     yhat = model.predict(x_input, verbose=0)
# #     temp_input.append(yhat[0][0])
# #     output.append(yhat[0][0])

# # forecast = scaler.inverse_transform(np.array(output).reshape(-1,1))

# # fig3, ax3 = plt.subplots(figsize=(12,6))
# # ax3.plot(forecast, label="Forecast")
# # ax3.set_xlabel("Days")
# # ax3.set_ylabel("Price USD")
# # ax3.legend()
# # st.pyplot(fig3)




































# # stock_prediction_app.py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import yfinance as yf
# import streamlit as st
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# # ------------------- STREAMLIT UI -------------------
# st.title("📈 Stock Trend Prediction App")

# # Sidebar input
# st.sidebar.header("Configuration")
# user_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
# start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
# end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
# epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=20, step=5)

# # ------------------- DATA -------------------
# df = yf.download(user_input, start=start_date, end=end_date)
# st.subheader("Raw Data")
# st.write(df.tail())

# # ------------------- Historical Closing Price -------------------
# st.subheader("Closing Price Over Time")
# fig, ax = plt.subplots(figsize=(12,6))
# ax.plot(df['Close'], label="Close Price")
# ax.set_xlabel("Date")
# ax.set_ylabel("Price USD")
# ax.legend()
# st.pyplot(fig)

# # ------------------- Data Preprocessing -------------------
# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# training_size = int(len(scaled_data) * 0.7)
# train_data = scaled_data[:training_size]
# test_data = scaled_data[training_size:]

# # Function to create sequences
# def create_dataset(dataset, time_step=60):
#     X, y = [], []
#     for i in range(time_step, len(dataset)):
#         X.append(dataset[i-time_step:i, 0])
#         y.append(dataset[i, 0])
#     return np.array(X), np.array(y)

# time_step = 60
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, y_test = create_dataset(test_data, time_step)

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # ------------------- Build & Train Model -------------------
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))

# model.compile(optimizer="adam", loss="mean_squared_error")
# st.text("Training LSTM model... ⏳")
# model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
# st.success("Training completed! ✅")

# # ------------------- Predictions -------------------
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)

# # ------------------- Plot Train vs Test -------------------
# st.subheader("Predictions vs Actuals")
# look_back = time_step

# trainPredictPlot = np.empty_like(scaled_data)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# testPredictPlot = np.empty_like(scaled_data)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2):len(scaled_data), :] = test_predict

# fig2, ax2 = plt.subplots(figsize=(12,6))
# ax2.plot(scaler.inverse_transform(scaled_data), label="Original")
# ax2.plot(trainPredictPlot, label="Train Prediction")
# ax2.plot(testPredictPlot, label="Test Prediction")
# ax2.set_xlabel("Time")
# ax2.set_ylabel("Price USD")
# ax2.legend()
# st.pyplot(fig2)

# # ------------------- Forecast Next 30 Days -------------------
# st.subheader("Next 30 Days Forecast")
# x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
# temp_input = list(x_input[0])
# output = []

# for i in range(30):
#     x_input = np.array(temp_input[-time_step:]).reshape(1,time_step,1)
#     yhat = model.predict(x_input, verbose=0)
#     temp_input.append(yhat[0][0])
#     output.append(yhat[0][0])

# forecast = scaler.inverse_transform(np.array(output).reshape(-1,1))

# fig3, ax3 = plt.subplots(figsize=(12,6))
# ax3.plot(forecast, label="Forecast")
# ax3.set_xlabel("Days")
# ax3.set_ylabel("Price USD")
# ax3.legend()
# st.pyplot(fig3)


































# stock_prediction_app_improved.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ------------------- STREAMLIT UI -------------------
st.title("📈 Improved Stock Trend Prediction App")

# Sidebar input
st.sidebar.header("Configuration")
user_input = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=20, step=5)

# ------------------- DATA -------------------
df = yf.download(user_input, start=start_date, end=end_date)
st.subheader("Raw Data")
st.write(df.tail())

# Use multiple features
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features]

# ------------------- Plot Closing Price -------------------
st.subheader("Closing Price Over Time")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df['Close'], label="Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("Price USD")
ax.legend()
st.pyplot(fig)

# ------------------- Data Preprocessing -------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data.values)

# Train/test split
training_size = int(len(scaled_data) * 0.7)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

# Create sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, :])
        y.append(dataset[i, 3])  # Close price is the target (index 3)
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# ------------------- Build & Train LSTM -------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

st.text("Training LSTM model... ⏳")
history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stop], verbose=1)
st.success("Training completed! ✅")

# ------------------- Predictions -------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform for plotting
close_scaler = MinMaxScaler()
close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]
train_predict_inv = close_scaler.inverse_transform(train_predict)
y_train_inv = close_scaler.inverse_transform(y_train.reshape(-1,1))
test_predict_inv = close_scaler.inverse_transform(test_predict)
y_test_inv = close_scaler.inverse_transform(y_test.reshape(-1,1))

# ------------------- Evaluate -------------------
rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict_inv))
st.write(f"Test RMSE: {rmse:.2f}")

# ------------------- Plot Predictions -------------------
st.subheader("Predictions vs Actuals")
look_back = time_step
trainPredictPlot = np.empty_like(scaled_data[:,3])
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back] = train_predict_inv[:,0]

testPredictPlot = np.empty_like(scaled_data[:,3])
testPredictPlot[:] = np.nan
testPredictPlot[len(train_predict)+(look_back*2):len(scaled_data)] = test_predict_inv[:,0]

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(close_scaler.inverse_transform(scaled_data[:,3].reshape(-1,1)), label="Original Close")
ax2.plot(trainPredictPlot, label="Train Prediction")
ax2.plot(testPredictPlot, label="Test Prediction")
ax2.set_xlabel("Time")
ax2.set_ylabel("Price USD")
ax2.legend()
st.pyplot(fig2)

# ------------------- Forecast Next 30 Days -------------------
st.subheader("Next 30 Days Forecast")
x_input = test_data[-time_step:].reshape(1,time_step,X_test.shape[2])
temp_input = list(x_input[0])
output = []

for i in range(30):
    x_input = np.array(temp_input[-time_step:]).reshape(1,time_step,X_test.shape[2])
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(np.zeros(X_test.shape[2]))
    temp_input[-1][3] = yhat[0][0]  # update Close price
    output.append(yhat[0][0])

forecast = close_scaler.inverse_transform(np.array(output).reshape(-1,1))

fig3, ax3 = plt.subplots(figsize=(12,6))
ax3.plot(forecast, label="Forecast")
ax3.set_xlabel("Days")
ax3.set_ylabel("Price USD")
ax3.legend()
st.pyplot(fig3)
