import numpy as np
import pandas as pd
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


# ===================
# DATA PREPROCESSING
# ===================
# View full Data sets
# movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
# users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
# ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# IMPORT DATA
training_set = pd.read_csv('../Boltzmann/ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('../Boltzmann/ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# FIND Max Num users and Moveis for dataset
num_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
num_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Convert datasets into matrix of user/movie and rating = USER(row),MOVIE(header),RATING(cell)
def convert(data):
    dataset = []
    for user_id in range(1, num_users+1):
        movies_rated = data[:,1][data[:,0]==user_id]
        user_ratings = data[:,2][data[:,0]==user_id]
        ratings = np.zeros(num_movies)
        ratings[movies_rated-1] = user_ratings
        dataset.append(list(ratings))
    return dataset

training_set = convert(training_set)
test_set = convert(test_set)

# Convert Matrixs into Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)
