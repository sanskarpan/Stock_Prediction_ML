import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

model = load_model("Prediction_Model.keras")

url = "https://stockanalysis.com/stocks/"
st.header('Stock Market Predictor')
st.write("You can visit to this [website](%s) to check the code of your desired stock." % url)

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = date.today()

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

data_tr = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

prev_100 = data_tr.tail(100)
data_test = pd.concat([prev_100, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MAVG50')
mavg50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(mavg50, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MAVG50 vs MAVG100')
mavg100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(mavg50, 'r')
plt.plot(mavg100, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MAVG100 vs MAVG200')
mavg200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(mavg100, 'r')
plt.plot(mavg200, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)
predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
