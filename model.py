import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from module import *
import random

class Spectrogram_generator(nn.Module):
    def __init__(self,config):
        super(Spectrogram_generator,self).__init__()
        num_embeddings = config['num_embeddings']
        input_channels = config['input_channels']
        hidden_channels = config['hidden_channels']
        num_blocks = config['num_blocks']
        train_metrics = config['train_metrics']
        self.use_mask = config['use_mask']
        
        self.num_embedding = num_embeddings
        self.hidden_channels=hidden_channels
        self.encode_one_frame = nn.Sequential(
            (nn.Linear(input_channels,hidden_channels)),
            nn.LayerNorm(hidden_channels),nn.LeakyReLU(inplace=True),
            config['positional_embed_enc']\
                (config['context_length']+1,hidden_channels),
            *[config['frame_operation_encoder'](config) for i in range(num_blocks)],
            (nn.Linear(hidden_channels,hidden_channels)),
            
        )

        self.decode_one_frame = nn.Sequential(
            config['positional_embed_dec']\
                (config['context_length']+1,hidden_channels),
            *[config['frame_operation_decoder'](config) for i in range(num_blocks)],
            (nn.Linear(hidden_channels,input_channels)),
        )
        self.noise = Gaussian_noise()
        self.embeddings = torch.nn.Parameter(torch.normal(0,0.02,(self.num_embedding,hidden_channels)))
        self.counter = torch.nn.Parameter(torch.zeros(self.num_embedding))
        
        self.counter.requires_grad=False
        self.train_metrics = Loss_func(train_metrics)
        self.l2_loss = nn.MSELoss()
    def get_code_indices(self, inputs):
        similarity = torch.matmul(inputs, self.embeddings.T)
        distances = (
            torch.sum(inputs ** 2, axis=1, keepdims=True)
            + torch.sum(self.embeddings ** 2, axis=1)
            - 2 * similarity
        )

        encoding_indices = torch.argmin(distances, axis=1)
        return encoding_indices


    def forward(self,inputs:torch.Tensor,preprocessing=lambda x:x,show_img=False):
        inputs = torch.transpose(inputs,1,2)
        x = inputs
        batch_sz,sample_sz,num_samples = x.shape
        
        y = inputs[:,-1]
        
        encoded_frames = self.encode_one_frame(x)
        vecs_to_sample = encoded_frames.view((-1,encoded_frames.shape[-1]))
        encoding_indices = self.get_code_indices(vecs_to_sample)
        encodings = nn.functional.one_hot(encoding_indices, self.num_embedding).float()
        quantized = torch.matmul(encodings, self.embeddings)

        commitment_loss = torch.mean((quantized.detach() - vecs_to_sample) ** 2)
        codebook_loss = torch.mean((quantized - vecs_to_sample.detach()) ** 2)
        quantized = vecs_to_sample + (quantized - vecs_to_sample).detach()

        quantized = quantized.view(encoded_frames.shape)
            
        dec = self.decode_one_frame(quantized)[:,-1]
        dec_loss = self.train_metrics(dec,y)

        out = {'scalars':{'dec_loss':dec_loss,'commitment_loss':commitment_loss,'codebook_loss':codebook_loss},
            'imgs':dict()}

        out['scalars']['opt_loss'] = dec_loss+0.02*commitment_loss+codebook_loss
        with torch.no_grad():
            self.counter-=1
            self.counter[encoding_indices]=1024
            dead_codebook = self.counter<=0
            num_dead = dead_codebook.sum()
            if num_dead>0:
                num_to_reset = random.randint(1,min(batch_sz,num_dead))
                codebook_to_reset = torch.nonzero(dead_codebook).squeeze(-1)[random.sample(range(num_dead),num_to_reset)]

                self.embeddings[codebook_to_reset] = self.noise(vecs_to_sample[random.sample(range(batch_sz),num_to_reset)],
                                                                0,0.05)
                self.counter[codebook_to_reset]=512
            
            
            out['scalars']['min_counter'] = max(self.counter.min(),0)
            out['scalars']['num_dead_codebooks'] = num_dead
            decoded = preprocessing.revert_preprocessing(torch.transpose(dec.unsqueeze(0),1,2))
            decoded[decoded<0]=0
            target = torch.transpose(preprocessing.revert_preprocessing(y.unsqueeze(0)),1,2)
            out['scalars']['dec_loss'] = self.l2_loss(decoded,target)
            
            if show_img:
                out['imgs']['dec_img']= decoded
                out['imgs']['diff']=torch.abs(decoded-target)
                out['imgs']['target_img']=target
                out['num_imgs']=decoded.shape[0]
        return out
                
    
    def inference_forward(self,inputs:torch.Tensor)->torch.Tensor:
        inputs = torch.transpose(inputs,1,2)
        x = inputs
        
        y = inputs[:,-1]
        
        encoded_frames = self.encode_one_frame(x)
        vecs_to_sample = encoded_frames.view((-1,encoded_frames.shape[-1]))
        encoding_indices = self.get_code_indices(vecs_to_sample)
        encodings = nn.functional.one_hot(encoding_indices, self.num_embedding).float()
        quantized = torch.matmul(encodings, self.embeddings)

        quantized = vecs_to_sample + (quantized - vecs_to_sample).detach()

        quantized = quantized.view(encoded_frames.shape)
            
        dec = self.decode_one_frame(quantized)[:,-1]
        return dec,y
    def encode(self,inputs:torch.Tensor)->torch.Tensor:
        inputs = torch.transpose(inputs,1,2)
        x = inputs        
        encoded_frames = self.encode_one_frame(x)
        vecs_to_sample = encoded_frames.view((-1,encoded_frames.shape[-1]))
        encoding_indices = self.get_code_indices(vecs_to_sample)
        return encoding_indices
    def decode(self,inputs:torch.Tensor)->torch.Tensor:
        encodings = nn.functional.one_hot(inputs, self.num_embedding).float()
        quantized = torch.matmul(encodings, self.embeddings)
        dec = self.decode_one_frame(quantized)[:,-1]
        return torch.transpose(dec,0,1)

class Token_generator(nn.Module):
    def __init__(self,embeddings,embed_dim):
        super(Token_generator,self).__init__()
        self.embedding = nn.Embedding(embeddings,embed_dim)
        dropout = 0.1
        self.embeddings=embeddings
        config=dict()
        config['hidden_channels']=embed_dim
        config['activation1'] = nn.LeakyReLU
        config['activation2'] = nn.LeakyReLU
        config['normalization'] = nn.LayerNorm
        config['dropout']=dropout
        config['num_heads'] = 12
        self.layers = nn.Sequential(Positional_embedding(630,embed_dim),
                                         *[Attention_block(config) for _ in range(35)],
                                         Time_distributed(nn.Linear(embed_dim,embeddings))
                                        )
        self.noise = None
        self.l1_loss = nn.L1Loss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
    def add_noise(self):
        self.noise = Gaussian_noise()
    def train(self,mode=True):
        super().train(mode=mode)
    def forward(self,inputs):
        x,y,time_x,time_y = inputs
        batch_sz = y.shape[0]
        
        y_pred = self.layers(self.embedding(x))
        y_pred = y_pred[:,-1]
        token_loss = self.ce_loss(y_pred,y.long())

        with torch.no_grad():
            ans = torch.argmax(y_pred,-1)
            correct = ans==y
            accuracy = correct.sum()/(batch_sz)
        out = {'scalars':{'opt_loss':token_loss,'accuracy':accuracy},
            'imgs':{},
            'num_imgs':batch_sz}

        return out

    def set_opt(self,opt,lr,scheduler=None):
        self.opt = opt(self.parameters(),lr=lr)
        if scheduler is not None:
            self.scheduler = scheduler(self.opt)
        else:
            self.scheduler = None

    def step(self):
        self.opt.step()
        self.opt.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()

    def inference_forward(self,inputs:torch.Tensor)->torch.Tensor:
        x,y,time_x,time_y = inputs
        batch_sz = y.shape[0]
        
        y_pred = self.layers(self.embedding(x))
        y_pred = y_pred[:,-1]
        # y_true = nn.functional.one_hot(y,y_pred.shape[-1])
        # y_true[:,y] = 1
        token_loss = self.ce_loss(y_pred,y.long())

        with torch.no_grad():
            ans = torch.argmax(y_pred,-1)
            correct = ans==y
            accuracy = correct.sum()/(batch_sz)
        return token_loss,accuracy
    def decode(self,inputs):
        x = self.embedding(inputs)
        
        y_pred = self.layers(x)
        y_pred = y_pred[:,-1]
        ans = torch.argmax(y_pred,-1)
        return ans