# Boltzmann Machines
# RBM can apparently be seen as a probabilistic graphical model

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nnBM
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
# training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = pd.read_csv('ml-1m/training_set.csv')
training_set = np.array(training_set, dtype = 'int')
# test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = pd.read_csv('ml-1m/test_set.csv')
test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
# training_set = torch.FloatTensor(training_set)
# test_set = torch.FloatTensor(test_set)

# to implement on GPU use: (https://pytorch.org/docs/stable/tensors.html)
training_set = torch.cuda.FloatTensor(training_set)
test_set = torch.cuda.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nv, nh)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W)
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W.t())
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Creating the architecture of the Neural Network
class RBM_gpu():
    def __init__(self, nv, nh):
        self.W = torch.cuda.FloatTensor(nv, nh).normal_()
        self.a = torch.cuda.FloatTensor(1, nh).normal_()  # p(h)|v
        self.b = torch.cuda.FloatTensor(1, nv).normal_()  # p(v)|h
    def sample_h(self, x):
        wx = torch.mm(x, self.W)
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)  # probability of a hidden node being activated given v
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W.t())
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        # v0 = visible node values at k = 0
        # ph0 = probability of hidden node = 1 at k = 0
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# apparently you can: model.cuda() in pytorch where model is a subclass of nn.Module
# but we are creating our own model I think?

nv = len(training_set[0])
nh = 100
batch_size = 100
# rbm = RBM(nv, nh)
rbm = RBM_gpu(nv, nh)


# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0] # to remove -1 values (ie films not rated) or set back to -1 essentially
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]  # need the training data to activate the neurons...
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:  # just one step (blind step, not random walk from before) - MCMC
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)  # one step gibs sampling
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))  # mean of the absolute distances
        s += 1.
print('test loss: '+str(test_loss/s))