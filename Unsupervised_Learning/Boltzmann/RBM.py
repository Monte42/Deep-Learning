import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.parallel
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
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
# print(type(training_set), len(training_set))
training_set = np.array(training_set, dtype='int')
# print(type(training_set), len(training_set))
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
# print(type(test_set), len(test_set))
test_set = np.array(test_set, dtype='int')
# print(type(test_set), len(test_set))

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
# print(len(training_set), len(training_set[0]))
# print(len(test_set), len(test_set[0]))

# Convert Matrixs into Torch Tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# covert rating to 1 for liked 0 for not liked
training_set[training_set==0] = -1
training_set[training_set==1] = 0
training_set[training_set==2] = 0
training_set[training_set>=3] = 1
test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set==2] = 0
test_set[test_set>=3] = 1




# ================
# BUILDING THE NN
# ================
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)

    def sample_h(self,x):
        wx  = torch.mm(x,self.W.t())
        activation = wx = self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self,y):
        wy  = torch.mm(y,self.W)
        activation = wy = self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.b += torch.sum((v0-vk),0)
        self.a += torch.sum((ph0-phk),0)

rbm = RBM(len(training_set[0]), 100)