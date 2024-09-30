import numpy as np
import pandas as pd
import yfinance as yf 
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2023-12-29'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Describing Data
st.subheader('Date from 2010 - 2023')
st.write(df.describe())

#Visualizations

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

df['ma100'] = df['Close'].rolling(100).mean()  # Add parentheses to .mean()
df['ma200'] = df['Close'].rolling(200).mean()  # Add parentheses to .mean()

# First Subheader: 100 Moving Average
st.subheader('Closing Price vs Time Chart with 100MA')
fig1 = plt.figure(figsize=(12,6))
plt.plot(df['ma100'], label='100-day MA')
plt.plot(df['Close'], label='Closing Price')
plt.title('Closing Price vs Time with 100MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

# Second Subheader: 100 Moving Average & 200 Moving Average
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
fig2 = plt.figure(figsize=(12,6))
plt.plot(df['ma100'], label='100-day MA')
plt.plot(df['ma200'], label='200-day MA')
plt.plot(df['Close'], label='Closing Price')
plt.title('Closing Price vs Time with 100MA & 200MA')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Load my model

model =  load_model('keras_model.h5')

#Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Prdicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)

# Footer
st.markdown("""
---
Made with by [Aniket Tambe](https://github.com/tambeaniket)
""")

