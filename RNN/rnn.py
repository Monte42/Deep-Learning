import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 


# ====================
# DATA PRE-PROCESSING
# ====================
# Import Train Set
dataset_train = pd.read_csv('datasets/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values
# Import Test Set
dataset_test = pd.read_csv('datasets/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0,1), copy=True) # these are default and dont need to add
scaled_train_set = sc.fit_transform(training_set)

# NEW - Setting Number of Time Steps
x_train = []
y_train = []
for i in range(60,len(scaled_train_set)):
    x_train.append(scaled_train_set[i-60:i, 0])
    y_train.append(scaled_train_set[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1)) # (batch_size,time_steps,predictors)
# batch_size = # of rows, time_steps = # of cols, predictors = # of values in cell



# =================
# BUILDING THE RNN
# =================
# Initialize RNN
regressor = Sequential() # Regressor vs rnn name because its a continuing value

# Adding first LSTM layer
regressor.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(.2))

# Adding additional LSTM layers
regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(.2))
regressor.add(LSTM(units=60, return_sequences=True))
regressor.add(Dropout(.2))

# Adding final LSTM layer
regressor.add(LSTM(units=60))
regressor.add(Dropout(.2))

# Oputput Layer
regressor.add(Dense(units=1))

# Complie RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')



# =================
# TRAINING THE RNN
# =================
regressor.fit(x_train, y_train, epochs=100, batch_size=32)



# ===============================================
# MAKING PREDICTIONS AND VISUALIZING THE RESULTS
# ===============================================
# combine datasets
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# Get 60 days prior to test as np array
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60 : ].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

predicted_stock_price = regressor.predict(x_test)

# reverse scaling
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
print(predicted_stock_price)


# Plot
plt.plot(real_stock_price, color="blue", label="Actual Stock Prices")
plt.plot(predicted_stock_price, color="green", label="Predicted Stock Prices")
plt.title('Google Stock Price Jan 2017')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()