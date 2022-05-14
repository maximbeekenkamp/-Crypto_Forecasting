import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()


warnings.filterwarnings('ignore')
pd.options.display.float_format = '${:,.2f}'.format

today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'
btc_df = yf.download('BTC-USD',start_date, today)
btc_df.tail()

btc_df.reset_index(inplace=True)
btc_df.columns
# print(btc_df.columns)

df_plot = btc_df[["Date", "Open"]]

# print(btc_df)

new_names = {
    "Date": "ds", 
    "Open": "y",
}

df_plot.rename(columns=new_names, inplace=True)



x = df_plot["ds"]
y = df_plot["y"]

fig0, ax0 = plt.subplots()
ax0.set_title("Time series plot of BTC-USD Open Price")
ax0.set_xlabel("Date") 
ax0.set_ylabel("Open price") 

ax0.plot(x, y, label="BTC-USD Open Price")
ax0.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')


btc_df.drop(['Date'], 1, inplace=True)
prediction_days = 30

df_train= btc_df[:len(btc_df)-prediction_days]
df_test= btc_df[len(btc_df)-prediction_days:]

training_set = df_train.values
training_set = min_max_scaler.fit_transform(training_set)

x_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]
x_train = np.reshape(x_train, (len(x_train), 1, 1))


num_units = 4
activation_function = "relu"
optimizer = "adam"
loss_function = "mean_squared_error"
batch_size = 5
num_epochs = 100

# Initialize the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = num_units, activation = activation_function, input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = optimizer, loss = loss_function)

# Using the training set to train the model
regressor.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs)

test_set = df_test.values

inputs = np.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted_price = regressor.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)

plt.figure(figsize=(25, 25), dpi=80, facecolor = 'w', edgecolor = 'k')

plt.plot(test_set[:, 0], color='red', label='Real BTC Price')
plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')

plt.title('BTC Price Prediction', fontsize = 40)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize = 40)
plt.legend(loc = 'best')
plt.show()






