import os
from torch.utils.data import Dataset,IterableDataset
import numpy as np
import torch
from typing import List
import random
import math
from process_audio import mp3_2_spectrogram
def worker_init_fn(worker_id):
    
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    dataset = worker_info.dataset
    data_lst=dataset.all_data
    data_lst.sort()
    seed = torch.initial_seed()-worker_id
    random.seed(seed)
    # print(seed)
    # data_lst = data_lst[:4]
    random.shuffle(data_lst)
    
    num_worker = worker_info.num_workers
    num_data = len(data_lst)
    per_worker = int(math.ceil((num_data / num_worker)))
    start = per_worker*worker_id
    end = start+per_worker
    dataset.data_lst=data_lst[start:end]
    # print(dataset.data_lst)

def move_data(batch):
    for i in range(len(batch)):
        if isinstance(batch[i],torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking = True)
    return batch

class Music_dataset_iter(IterableDataset):
    def __init__(self, data_path,all_data,config,mode='train'):
        self.all_data = all_data
        self.data_lst = all_data
        self.data_path=data_path
        self.hop_len = config['hop_len']
        self.n_mels=config['n_mels']
        self.context_length = config['context_length']
        if mode == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.step_size = config['sample_step_size']
        self.num_musics_to_sample=config['num_musics_to_sample']
    def get_nmels(self):
        return self.n_mels
    def __iter__(self):
        num_data = len(self.data_lst)
        data_idx=[]
        current_data=[]
        for i in range(num_data+1):
            if len(current_data)<self.num_musics_to_sample and i<num_data:
                fn = self.data_lst[i]
                d = mp3_2_spectrogram(self.data_path+fn,hop_len = self.hop_len, n_mels=self.n_mels)[0].astype(np.float16)
                if self.is_train:
                    d = np.pad(d,((0,0),(0,random.randrange(0,self.step_size))))
                current_data.append(d)
                data_idx += [(len(current_data)-1,j) for j in range(self.context_length+1,d.shape[-1],self.step_size)]
                continue
            random.shuffle(data_idx)
            for idx,j in data_idx:
                data = current_data[idx][:,j-1-self.context_length:j].astype(np.float32)
                yield torch.tensor(data)
            current_data = []
            data_idx=[]
class Token_dataset_iter(IterableDataset):
    def __init__(self, data_path,data_split,context_length = 630,step_sz=126,mode='train'):
        self.data_path=data_path
        self.data_split = data_split
        self.context_length=context_length
        self.step_sz = step_sz
        # self.data_lst = os.listdir(data_path)
        self.mode=mode
        self.load_data()
    def load_data(self):
        self.data = []
        max_time = 0
        for fn in self.data_split:
            d = np.load(self.data_path+fn).astype(np.uint8)
            if self.mode =='train':
                offst = random.randint(0,self.step_sz//2)
            else:
                offst = 0
            for i in range(self.context_length+1+offst,d.shape[-1],self.step_sz):
                start = i-self.context_length-1
                end = i-1
                x = d[start:end]
                y = d[i]
                time_x = np.array(range(start,end),dtype=np.uint16)
                time_y = np.array([i],dtype=np.uint16)
                max_time = max(max_time,i)
                self.data.append((x,y,time_x,time_y))
        self.max_time = max_time
    def __iter__(self):
        random.shuffle(self.data)
        for x,y,time_x,time_y in self.data:
            yield torch.tensor(x.astype(np.int32)),torch.tensor(y.astype(np.int64)),\
                    torch.tensor(time_x.astype(np.int32)),torch.tensor(time_y.astype(np.int32))

def collate_token(batch):
    return torch.stack(batch,0)

class Batch_collator(object):
    def __call__(self,batch):
        if isinstance(batch,torch.Tensor):
            return batch 
        if isinstance(batch[0],torch.Tensor):
            return torch.stack(batch,0)
        data = [[] for _ in range(len(batch[0]))]
        for b in batch:
            for idx,d in enumerate(b):
                data[idx].append(d)

        batched_data = [torch.stack(i) for i in data]
        return batched_data


class Preprocessing:
    def __init__(self,normalize_frame=False):
        self.normalize_frame=normalize_frame

    def __call__(self,batches:torch.Tensor):
        batches = batches.cuda(non_blocking=True)
        if self.normalize_frame:
            return self.normalize_and_concat(batches)
        else:
            return batches
    def revert_preprocessing(self,data):
        if self.normalize_frame:
            return self.un_normalize(data)
        else:
            return data.clone()
    
    def normalize_and_concat(self,data):
        col_l2_norm = torch.sqrt(torch.sum(data**2,1,keepdim=True))
        data = torch.cat((col_l2_norm,\
                                  data),dim=1)
        small_norm = (col_l2_norm<=0.1).squeeze(1)
        data[:,1:]/=col_l2_norm
        data = torch.transpose(data,1,2)

        data[small_norm]=0
        data = torch.transpose(data,1,2)
        data[:,0] = torch.log(data[:,0]+1)
        return data
    def un_normalize(self,data):
        scale = data[:,0]
        scale = torch.exp(scale)-1

        orig = data*scale.unsqueeze(1)
        orig[:,0]=0
        return orig
    
