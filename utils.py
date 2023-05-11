import torch
import sys
from tensorboardX import SummaryWriter
from typing import List
from hyperparams_spec import *
from model import *
from data import *
from torch.utils.data import DataLoader

def validate(model,loader,preprocessing:Preprocessing):
    model.eval()
    tot_loss = 0
    num_samples = 0
    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        for step,batch in enumerate(loader):
            if isinstance(batch,list):
                data = [preprocessing(b) for b in batch]
            else:
                data = preprocessing(batch)
            batch_sz = batch.shape[0] if isinstance(batch, torch.Tensor) else batch[0].shape[0]
            dec,ans = model.inference_forward(data)
            dec = preprocessing.revert_preprocessing(dec)
            ans = preprocessing.revert_preprocessing(ans)
            loss = loss_func(dec,ans)
            tot_loss += loss*batch_sz
            num_samples+=batch_sz
    return tot_loss/num_samples
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
def print_stats(epoch:int,step:int,total_losses:dict,time_stats:dict,print_step:int,flush=False):
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

def creat_config(config,frame_op_enc,frame_op_dec,normalization,activation,loss,normalize_frame,context_length,\
                    optim):
    frame_operation_choices = [lambda config:Time_distributed(Residual_block(config)),Attention_block]

    normalization_choices = [nn.LayerNorm,nn.BatchNorm1d]
    activation_choices = [nn.LeakyReLU,nn.ReLU]

    positional_embed_choices = [lambda ht,wid:nn.Sequential(),Positional_embedding]
    loss_choices = [[nn.MSELoss],[nn.MSELoss,nn.L1Loss]]
    normalize_frame_choices = [False,True]
    optim_choices = [torch.optim.Adam,torch.optim.SGD]
    
    config['positional_embed_enc'] = positional_embed_choices[frame_op_enc]
    config['positional_embed_dec'] = positional_embed_choices[frame_op_dec]
    config['train_metrics'] = loss_choices[loss]
    config['normalize_frame'] = normalize_frame_choices[normalize_frame]
    
    config['normalization']=normalization_choices[normalization]
    config['activation1']=activation_choices[activation]
    config['activation2']=activation_choices[activation]
    config['optim'] = optim_choices[optim]
    config['frame_operation_encoder'] = frame_operation_choices[frame_op_enc]
    config['frame_operation_decoder'] = frame_operation_choices[frame_op_dec]
    config['context_length'] = context_length
    
    config['use_mask'] = 1

    
def init_run(config):
    all_data = os.listdir('./data/')
    all_data.sort()
    random.shuffle(all_data)
    train_split = all_data[:int(0.9*len(all_data))]
    val_split = all_data[int(0.9*len(all_data)):]

    train_dataset = Music_dataset_iter('./data/',train_split,
                                 config)
    
    val_dataset = Music_dataset_iter('./data/',val_split,
                                 config,mode='val')
    
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=False,num_workers=config['num_worker'], drop_last=False,
                            pin_memory=True,worker_init_fn=worker_init_fn,collate_fn=Batch_collator())
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,num_workers=config['num_worker'], drop_last=False,
                            pin_memory=True,worker_init_fn=worker_init_fn,collate_fn=Batch_collator())
    model = Spectrogram_generator(config)
    optim = config['optim'](model.parameters(),lr = config['lr'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    preprocessing = Preprocessing(config['normalize_frame'])
    return model,optim,preprocessing,train_loader,val_loader
    
def load_model(model,path):
    trained_model = os.listdir(path)
    if trained_model and trained_model!=['0_0']:
        model_idx = sorted([(int(i.split('_')[0]),idx) for idx,i in enumerate(trained_model)],key=lambda x:x[0])[-1][-1]
        fn = trained_model[model_idx]
        model.load_state_dict(torch.load(path+fn))
        starting_epoch,global_step = int(fn.split('_')[0])+1,int(fn.split('_')[1])
    else:
        starting_epoch,global_step=0,0
    return starting_epoch,global_step