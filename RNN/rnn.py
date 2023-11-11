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