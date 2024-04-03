import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import datetime
import yfinance as yfin

from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trading Prediciton Model')
yfin.pdr_override()
user_input = st.text_input("Enter Ticker", "TSLA")
df = pdr.get_data_yahoo(user_input,start,end)
df.head()

#Describing
st.subheader('Data from 2010 to 2020')
st.write(df.describe())

#Plots
st.subheader('Closing Price & Time Chart Comparison')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price & Time Chart Comparison - 100 Moving-Day Average')
fig = plt.figure(figsize=(12,6))
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price & Time Chart Comparison - 200 Moving-Day Average')
fig = plt.figure(figsize=(12,6))
ma200 = df.Close.rolling(200).mean()
plt.plot(ma100, 'r')
plt.plot()
plt.plot(ma200, 'b')
st.pyplot(fig)

#Training and Testing Datasets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
#Testing Data
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

x_train = []
y_train = []

data_training_array = scaler.fit_transform(data_training)

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])

x_train , y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.keras')

#Testing
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing])
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
y_predicted = model.predict(x_test)
scalar = scaler.scale_

scale_factor = 1/scalar[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Prediction Model Vs Original Model')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Model Price')
plt.plot(y_predicted, 'r', label = 'Prediction Model Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)