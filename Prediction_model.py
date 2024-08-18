import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
import tensorflow as tf

# Download stock data
start = '2012-01-01'
end = '2024-03-20'
stock = 'GOOG'
data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Prepare data
data.dropna(inplace=True)
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

x = []
y = []

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i - 100:i])
    y.append(data_train_scale[i, 0])

x, y = np.array(x), np.array(y)

# Building the model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=50, batch_size=32, verbose=1)

# Saving the model
model.save('Prediction_Model.keras')

# Loading the model
model = load_model('Prediction_Model.keras')

# Predicting on test data
prev100 = data_train.tail(100)
data_test = pd.concat([prev100, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

x1 = []
y1 = []
for i in range(100, data_test_scale.shape[0]):
    x1.append(data_test_scale[i - 100:i])
    y1.append(data_test_scale[i, 0])

x1, y1 = np.array(x1), np.array(y1)

y_predict = model.predict(x1)
scale = 1 / scaler.scale_
y_predict = y_predict * scale
y1 = y1 * scale

# Plotting results
plt.figure(figsize=(10, 8))
plt.plot(y_predict, 'r', label='Predicted Price')
plt.plot(y1, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()