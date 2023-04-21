import os
import scipy.stats as st
import numpy as np

# Parameters
num_epochs = 30
num_agents_sqr = 3
num_agents = num_agents_sqr ** 2
batch_size = 10
num_classes = 2
num_multiclasses = 10

N_TEST = 1000
N_TEST_CYCLE = 1000
N_CYCLES = 5
N_MC = 5000
SEED_GRAPH = 1


##### First Setting #####
hidden_size = 64
learning_rate = 0.001 
train_size = 200
train_repetitions = 5
SEED = 21


##### Second Setting #####
learning_rate_simple = 0.0001 
hidden_size_simple = 10
train_size_simple = 40
test_horizon = 20
SEED_SIMPLE = 5

##### Third Setting #####
N_TEST_MULTI = 100
train_size_multi = 1000


##### Mnist data #####
s1 = (28 // num_agents_sqr + 28 % num_agents_sqr)
s2 = 28 // num_agents_sqr
mnist_input =  [s1**2,
                s2*s1,
                s2*s1,
                s2*s1,
                s2*s2,
                s2*s2,
                s2*s1,
                s2**2,
                s2**2]

##### Gaussian example #####
Train_sizes = [100, 300, 500, 1000, 1500]

N_ITER_g = 10000
N_TRAIN_g = 100
N_TEST_g = 10000
N_MC_g = 10

num_agents_g = 4
num_classes_g = 2
dim_g = 2
hidden_size_g = 10
num_epochs_g = 1000
learning_rate_g = 0.0001

Cov = np.eye(2)
Mean =  np.zeros(2)

Cov2 = np.eye(2) * 1.5
Mean2 =  np.zeros(2)

Q1 = st.multivariate_normal(Mean, Cov)
Q2 = st.multivariate_normal(Mean2, Cov2)

Q = [Q1, Q2]


A_g = 0.5 * np.array([[1,1,0,0],
                [0,1,1,0],
                [0,0,1,1],
                [1,0,0,1]])

FIGS_PATH = 'figs/'
if not os.path.isdir(FIGS_PATH):
    os.makedirs(FIGS_PATH)

DATA_PATH = 'data/'
if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

MODELS_PATH = 'models/'
if not os.path.isdir(MODELS_PATH):
    os.makedirs(MODELS_PATH)

    