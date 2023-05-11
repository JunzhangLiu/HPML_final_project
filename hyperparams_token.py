import torch.nn as nn
from hyperparams_spec import num_embedding as embeddings 
from module import *
#Musics
n_fft = 256
hop_len = 700
sr = 44100
num_samples_per_sec = 63

#Training
batch_size = 8
grad_accu_step = 4

lr = 0.00001

embed_dim=768
