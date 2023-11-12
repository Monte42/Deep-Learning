# ========
# IMPORTS
# ========
import numpy as np
import pandas as pd
import tensorflow as tf
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from pylab import bone, pcolor, colorbar, plot, show


# ====================
# SELF ORGANIZING MAP
# ====================
# Data Preprocessing
dataset = pd.read_csv('Credit_Card_Applications.csv')
customer_details = dataset.iloc[:,:-1]
customer_acceptance = dataset.iloc[:,-1]
# Feature Scaling
sc = MinMaxScaler()
customer_details = sc.fit_transform(customer_details)
# Init and run S.O.M.
som = MiniSom(10,10,input_len=15)
som.random_weights_init(customer_details)
som.train_random(customer_details, 100)
# Get cell(s) with with highest raiting "frauds"
heat_map = som.distance_map().T
hot_cells = []
for row in heat_map:
    for cell in row:
        if cell > .80 and som.winner(cell) not in hot_cells:
            hot_cells.append(som.winner(cell))
# Show 2D Map
bone()
pcolor(heat_map)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,j in enumerate(customer_details):
    w = som.winner(j)
    plot(
        w[0]+.5,
        w[1]+.5,
        markers[customer_acceptance[i]],
        markeredgecolor = colors[customer_acceptance[i]],
        markerfacecolor = 'None',
        markersize = 10,
        markeredgewidth = 2
    )
show()
# Extract Customers likely to have commited fraud
mappings = som.win_map(customer_details)
frauds = mappings[hot_cells[0]]
frauds = sc.inverse_transform(frauds)





# ===========================
# ARTIFICIAL NEURAL NETWORKS
# ===========================

customers = dataset.iloc[:,1:].values
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1


# Build train/test sets
detail_train, detail_test, accept_train, accept_test = train_test_split(customers, is_fraud, test_size=.2, random_state=0)
# Feature Scaling
stsc = StandardScaler()
detail_test = stsc.fit_transform(detail_test)
detail_train = stsc.fit_transform(detail_train)
# BUILD ANN
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))
ann.add(tf.keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Train ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(detail_train,accept_train,batch_size=1,epochs=3)
# Predict Results
fraud_predict = ann.predict(customers)
fraud_predict = np.concatenate((dataset.iloc[:,0:1].values, fraud_predict), axis=1)
fraud_predict = fraud_predict[fraud_predict[:,1].argsort()]
print(fraud_predict)




# Confusion Matrix
# cm = confusion_matrix(accept_test, fraud_predict)


