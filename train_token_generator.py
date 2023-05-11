from model import *
from process_audio import *
import numpy as np
import torch
import torch.nn as nn
from data import *
from torch.utils.data import DataLoader
import time
from typing import List
from tensorboardX import SummaryWriter
from collections import defaultdict
import random
import sys
import warnings
from hyperparams_token import *
from utils import *
warnings.simplefilter("ignore")


torch.manual_seed(100)
random.seed(100)
def validate(model,loader):
    model.eval()
    tot_loss = 0
    num_samples = 0
    tot_acc = 0
    with torch.no_grad():
        for step,batch in enumerate(loader):
            batch = to_cuda(batch)
            loss,accuracy = model.inference_forward(batch)
            batch_sz = batch[0].shape[0]
            tot_loss += loss*batch_sz
            tot_acc += accuracy*batch_sz
            num_samples+=batch_sz
    return tot_loss/num_samples,tot_acc/num_samples
def to_cuda(batch:List[torch.Tensor]):
    for i in range(len(batch)):
        if isinstance(batch[i],torch.Tensor):
            batch[i] = batch[i].cuda(non_blocking = True)
    return batch
def stats(start,times,time_table):
    t0 = start
    for name,t1 in times:
        time_table[name]+=t1-t0
        t0 = t1
def print_stats(epoch:int,step:int,total_losses:dict,time_stats:dict,print_step:int):
    print('Epoch[{0}] step [{1}]'.format(epoch,step),end = '\t')
    for name,val in total_losses.items():
        print('{0}:{1:.3f}'.format(name,val/print_step), end = ' ')
        total_losses[name]=0
    for name,val in time_stats.items():
        print('{0}:{1:.3f}'.format(name,val/print_step), end=' ')
        time_stats[name]=0
    print()
    sys.stdout.flush()
def display_img(writer:SummaryWriter,title,tensor:torch.Tensor,step:int):
    t = tensor
    writer.add_image(title,t,step)
def train(model,dataloader,writer:SummaryWriter =None,\
          epochs=100,grad_accu_step=1,save_freq = 1,print_step = 1,\
          global_step=0,starting_epoch=0,show_img_step = 250,val_loader=None):
    time_stats = defaultdict(lambda :0)
    loss_types = defaultdict(lambda :0)
    for e in range(starting_epoch,epochs):
        torch.cuda.empty_cache()
        start = time.time()
        model.train()
        for step,batch in enumerate(dataloader):
            data_loading_time = time.time()
            data_moving_time = time.time()
            batch = to_cuda(batch)
            out = model(batch)
            
            loss = out['scalars']['opt_loss']/grad_accu_step

            forward_time = time.time()
            loss.backward()
            backward_time = time.time()
            tot_norm = nn.utils.clip_grad_norm_(model.parameters(),100)
            out['scalars']['tot_norm'] = tot_norm
            if (step+1)%grad_accu_step==0:
                model.step()
            
            for key,val in out['scalars'].items():
                loss_types[key]+=val.item() if isinstance(val,torch.Tensor) else val
            
            stats(start,[('data_loading_time',data_loading_time),('data_moving_time',data_moving_time),
                        ('forward_time',forward_time),('backward_time',backward_time)],time_stats)
            if global_step%print_step==0:
                print_stats(e,step,loss_types,time_stats,print_step)
                writer.flush()

            if writer is not None:
                for key,val in out['scalars'].items():
                    writer.add_scalar(key,val,global_step)

            if (global_step)%show_img_step==0:
                # img_to_show = random.randrange(0,out['imgs'][key][img_to_show].shape[0])
                for key,val in out['imgs'].items():
                    # if 'img' in key:
                    display_img(writer,key,out['imgs'][key].unsqueeze(0),global_step)
                    # else:
                    #     display_img(writer,key,out['imgs'][key][img_to_show].unsqueeze(0),global_step)
            start = time.time()
            global_step+=1
        if val_loader is not None:
            avg_loss,acc = validate(model,val_loader)
            writer.add_scalar('val_loss',avg_loss,e)
            writer.add_scalar('val_acc',acc,e)
            print(e,avg_loss,acc)
        if e%save_freq==0:
            torch.save(model.state_dict(), experiment_dir+'trained_model/'+str(e)+'_'+str(global_step))
if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    random.seed(0)

    experiment_dir = './experiments/auto_enc/'
    model_dir = experiment_dir+'trained_model/'
    logdir = experiment_dir+'tb/'

    num_epoch = 200
    all_data = os.listdir('./tokens')
    all_data.sort()
    random.shuffle(all_data)
    train_split = all_data[:int(0.9*len(all_data))]
    val_split = all_data[int(0.9*len(all_data)):]
    
    
    train_dataset = Token_dataset_iter('./tokens/',data_split = train_split)
    val_dataset = Token_dataset_iter('./tokens/',data_split = val_split,mode='val')

    collator = Batch_collator() 

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=False,num_workers=1, drop_last=False,
                            pin_memory=True,
                            collate_fn=collator)
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,num_workers=1, drop_last=False,
                            pin_memory=True,
                            collate_fn=collator)
    
    model = Token_generator(embeddings,embed_dim)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    starting_epoch,global_step = load_model(model,model_dir)
    

    writer = SummaryWriter(logdir = logdir)

    model.set_opt(torch.optim.Adam,lr=lr)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model,train_loader,writer=writer,\
          epochs=num_epoch,grad_accu_step=grad_accu_step,save_freq = 1,print_step = grad_accu_step*10,\
          global_step=global_step,starting_epoch=starting_epoch,val_loader=val_loader)