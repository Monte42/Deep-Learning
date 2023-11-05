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

# ADD IPUT LAYER AND FIRST HIDDEN LAYER

# ADD SECOND HIDDEN LAYER

# ADD OUTPUT LAYER
