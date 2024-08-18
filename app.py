import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date
from sklearn.preprocessing import MinMaxScaler
import os
import requests

# Set environment variables
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define the model file path
model_path = "Prediction_Model.keras"

# Check if the model file exists, and if not, download it
if not os.path.exists(model_path):
    st.warning("Model file not found locally. Attempting to download...")
    model_url = "https://raw.githubusercontent.com/sanskarpan/Stock_Prediction_ML/main/Prediction_Model.keras"
    response = requests.get(model_url)
    
    if response.status_code == 200:
        with open(model_path, 'wb') as f:
            f.write(response.content)
        st.success("Model downloaded successfully.")
    else:
        st.error("Failed to download the model. Please check the URL or your internet connection.")
        st.stop()

# Load the model
try:
    model = load_model(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Streamlit app header and description
url = "https://stockanalysis.com/stocks/"
st.header('Stock Market Predictor')
st.write(f"You can visit this [website]({url}) to check the code of your desired stock.")

# Input field for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = date.today()

# Download stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Prepare data for prediction
data_tr = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
scaler = MinMaxScaler(feature_range=(0, 1))

prev_100 = data_tr.tail(100)
data_test = pd.concat([prev_100, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plotting price vs different moving averages
st.subheader('Price vs NEXT50')
next_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(next_50, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price vs NEXT50 vs NEXT100')
next100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(next_50, 'r')
plt.plot(next100, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs NEXT100 vs NEXT200')
next200 = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(next100, 'r')
plt.plot(next200, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

# Prepare data for prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)
predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Plotting original price vs predicted price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
