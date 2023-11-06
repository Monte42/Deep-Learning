import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# ===================
# DATA PREPROCESSING
# ===================
# IMPORT DATASET
df = pd.read_csv('dataset.csv')
# EXTRACT FREATURES(needed cols) & 
x  = df.iloc[:,3:-1].values
y  = df.iloc[:,-1].values
# print(x)
# print(y)
# ENCODE CATERGORICAL DATA
# Gender
le =  LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])
# print(x)
# Country / One Hot Encodeing No Order
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)
# SPLIT DATA INTO TRAIN/TEST SETS
# Unpacking
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# FEATURE SCALING
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# print(x_train)
# print(x_test)



# ======================================
# BUILDING THE ARTIFICIAL NEURAL NETWORK
# ======================================
# INITIALIZE THE ANN
ann = tf.keras.models.Sequential()

# ADD IPUT LAYER AND FIRST HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

# ADD SECOND HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

# ADD OUTPUT LAYER
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) # softmax if more than 1 neuron



# ================================
# TRAIN ARTIFICIAL NEURAL NETWORK
# ================================
# COMPILING ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # loss=category_crossentropy if output non-binary

# TRAIN ANN ON TRAINSET
ann.fit(x_train, y_train, batch_size=32, epochs=40)



# =========
# HOMEWORK
# =========
# Make a singler prediction for the one customer below using the trained ANN model 

# 'Geography': 'France',
# 'CreditScore': 600,
# 'Gender': 'Male',
# 'Age': 40,
# "Tenure": 3,
# 'Balance': 6000,
# 'NumberOfPrducts': 2,
# 'HasCrCard': 1,
# 'IsActiveMember': 1,
# 'EstimatedSalary': 50000,

data  = [[1,0,0,600,1,40,3,60000,2,1,1,50000]]

data = sc.transform(data)

print(f'Probabilty of customer leaving: {ann.predict(data)}')
print('Customer will leave') if ann.predict(data) > .5 else print('Customer will stay')

