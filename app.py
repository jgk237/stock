import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler



st.title('stock Analysis')

start = st.date_input(
    "Select the start date",
    datetime.date(2018, 1,1))
end = st.date_input(
    "Select the end date",
    datetime.date(2023, 6, 6))
user_input = st.text_input('Enter Stock Ticker', 'AFL')
df = yf.download(user_input, start, end)

st.subheader('Data from 2012 - 2022')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(m100)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA and 200MA')
m100 = df.Close.rolling(100).mean()
m200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(m100)
plt.plot(m200)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])



scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i - 100: i])
    y_train.append(data_training_array[i: 0])

x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('model.h5')
# try:
#     model = load_model('keras_model.h5')
# except FileNotFoundError:
#     st.error("Model file not found. Please make sure the file path is correct.")
# except:
#     st.error("An error occurred while loading the model.")

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100: i])
    y_test.append(data_training_array[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predicted Stock Price')
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Orginal Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.write(
    "<div style='text-align: center; margin-top: 70px; font-size: 18px;'>Build by | Sagar Sugunan | Sajjad Saheer Ali | Nishan P | ",
    unsafe_allow_html=True)
