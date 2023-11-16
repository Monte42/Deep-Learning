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



# ============================
# Design Stacked Auto Encoder
# ============================
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.full_connection_1 = nn.Linear(num_movies, 24)
        self.full_connection_2 = nn.Linear(24, 10)
        self.full_connection_3 = nn.Linear(10, 24)
        self.full_connection_4 = nn.Linear(24, num_movies)
        self.activation = nn.Sigmoid()
    def forward_prop(self, x):
        x = self.activation(self.full_connection_1(x))
        x = self.activation(self.full_connection_2(x))
        x = self.activation(self.full_connection_3(x))
        x = self.full_connection_4(x)
        return x

# Initialize SAE
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.012, weight_decay = 0.05)
# Training tha SAE
ephocs = 300
for e in range(1, ephocs+1):
    train_loss = 0
    users_who_rated_movies = 0.
    for user_id in range(num_users):
        input_vector = Variable(training_set[user_id]).unsqueeze(0)
        target = input_vector.clone()
        if torch.sum(target.data > 0) > 0:
            output_vector = sae.forward_prop(input_vector)
            target.require_grad = False
            output_vector[target == 0] = 0
            loss = criterion(output_vector, target)
            mean_corrector = num_movies/float(torch.sum(target.data > 0)+ 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data.item() * mean_corrector)
            users_who_rated_movies += 1.
            optimizer.step()
    print(f'Epoch: {str(e)} - Loss: {str(train_loss/users_who_rated_movies)}')

test_loss = 0
users_who_rated_movies = 0
for user_id in range(num_users):
    input_vector = Variable(training_set[user_id]).unsqueeze(0)
    target = Variable(test_set[user_id]).unsqueeze(0)
    if torch.sum(target.data > 0) > 0:
        output_vector = sae.forward_prop(input_vector)
        target.require_grad = False
        output_vector[target == 0] = 0
        loss = criterion(output_vector, target)
        mean_corrector = num_movies/float(torch.sum(target.data > 0)+ 1e-10)
        test_loss += np.sqrt(loss.data.item())
        users_who_rated_movies += 1.
print(f'Test Loss: {str(test_loss/users_who_rated_movies)}')