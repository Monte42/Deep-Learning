import numpy as np  
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show


# Import Data
dataset = pd.read_csv('Credit_Card_Applications.csv')
# extract data
x = dataset.iloc[:,:-1].values # customer details
y = dataset.iloc[:,-1].values # if customer approved

# Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
x = sc.fit_transform(x)


# Train
# x,y make output grid // input len is # of cols // sigma is radius of affected neurons
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=.5) # Learn rate and rate of wieght change
som.random_weights_init(x)
som.train_random(data = x, num_iteration = 100)

# Visualize
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,j in enumerate(x):
    win_node = som.winner(j)
    plot(
        win_node[0]+.5, # Places centers x axis
        win_node[1]+.5, # Places centers y axis
        markers[y[i]], # checks y @ index / adds correct marker
        markeredgecolor = colors[y[i]],# checks y @ index / sets correct color
        markerfacecolor = 'None',
        markersize = 10,
        markeredgewidth = 2
    )
show()

mappings = som.win_map(x)
frauds = np.concatenate((mappings[(8,1)],mappings[(8,1)]), axis=0)
frauds = sc.inverse_transform(frauds)
print(frauds)