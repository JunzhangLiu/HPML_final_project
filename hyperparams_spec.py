import torch.nn as nn
from module import *
#Musics
n_mels = 256
hop_len = 700
sr = 44100
num_samples_per_sec = int(sr/hop_len)

#Training
batch_size = 1024
num_epoch = 80
grad_accu_step = 1
scheduler = lambda :None
lr = 0.0001
loss = 0
num_worker = 8
num_musics_to_sample = 20
sample_step_size = 6

#Model
frame_op_enc,frame_op_dec=0,1
normalization=0
activation=0
optim=0
num_blocks = 10
input_frames = num_samples_per_sec
num_heads = 8

normalize_frame = 0

num_embedding = 2048
hidden_channel = n_mels+1 if normalize_frame else n_mels


config = dict()
config['num_musics_to_sample'] = num_musics_to_sample
config['hop_len'] = hop_len
config['n_mels'] = n_mels
config['lr'] = lr
config['sample_step_size'] = sample_step_size
config['num_embeddings']=num_embedding
config['input_channels']=hidden_channel
config['hidden_channels'] = hidden_channel
config['num_blocks'] = num_blocks
config['input_frames']=input_frames
config['num_worker'] = num_worker
config['num_heads'] = num_heads
