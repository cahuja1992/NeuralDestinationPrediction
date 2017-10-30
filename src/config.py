import os

ROOT_DIR = os.getenv("DESTPRED_HOME")
DATA_DIR = os.getenv('DESTPRED_DATA', '/data/')

# HyperParameters for Optimizer
lr = 0.01
momentum = 0.9
clip_value = 1.
batch_size = 200
n_epochs = 1

# HyperParameters for MLP
embedding_dim = 10


# HyperParamters for LSTM