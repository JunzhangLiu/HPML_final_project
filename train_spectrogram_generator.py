from model import *
from process_audio import *
from hyperparams_spec import *
import torch
import torch.nn as nn
from data import *
import time
from tensorboardX import SummaryWriter
from collections import defaultdict
import shutil
import random
from utils import *
import warnings
warnings.simplefilter("ignore")


def train(model,optim,dataloader,preprocessing,writer:SummaryWriter =None,\
          epochs=100,grad_accu_step=1,save_freq = 1,print_step = 1,\
          global_step=0,starting_epoch=0,show_img_step = 250,val_loader=None):
    time_stats = defaultdict(lambda :0)
    loss_types = defaultdict(lambda :0)
    min_val_loss = float('inf')
    for e in range(starting_epoch,epochs):
        torch.cuda.empty_cache()
        epoch_start = time.time()
        tot_data_time = 0
        start = time.time()
        model.train()
        for step,batch in enumerate(dataloader):
            data_loading_time = time.time()
            if isinstance(batch,list):
                data = [preprocessing(b) for b in batch]
            else:
                data = preprocessing(batch)
            data_moving_time = time.time()
            out = model(data,preprocessing,show_img=(global_step)%show_img_step==0)
            
            loss = out['scalars']['opt_loss']

            forward_time = time.time()
            loss.backward()
            backward_time = time.time()
            tot_norm = nn.utils.clip_grad_norm_(model.parameters(),100)
            out['scalars']['tot_norm'] = tot_norm
            if (step+1)%grad_accu_step==0:
                optim.step()
                optim.zero_grad()
            
            for key,val in out['scalars'].items():
                loss_types[key]+=val.item() if isinstance(val,torch.Tensor) else val
            
            stats(start,[('data_loading_time',data_loading_time),('data_moving_time',data_moving_time),
                        ('forward_time',forward_time),('backward_time',backward_time)],time_stats)
            tot_data_time += data_loading_time-start
            if global_step%print_step==0:
                print_stats(e,step,loss_types,time_stats,print_step)

            if writer is not None:
                for key,val in out['scalars'].items():
                    writer.add_scalar(key,val,global_step)

                if (global_step)%show_img_step==0:
                    img_to_show = random.randrange(0,out['num_imgs'])
                    for key,val in out['imgs'].items():
                        display_img(writer,key,out['imgs'][key][img_to_show].unsqueeze(0),global_step)
                writer.flush()
                
            start = time.time()
            global_step+=1
        if val_loader is not None:
            avg_loss = validate(model,val_loader,preprocessing)
            if writer is not None:
                writer.add_scalar('val_loss',avg_loss,e)
            min_val_loss = min(min_val_loss,avg_loss)
            print(e,avg_loss)
        if e%save_freq==0:
            torch.save(model.state_dict(), model_dir+str(e)+'_'+str(global_step))
        
        print('Epoch_time',time.time()-epoch_start,'Tot_data_time',tot_data_time)
    return min_val_loss
    
def main():
    torch.manual_seed(10)
    random.seed(10)

    
    context_length = input_frames
    creat_config(config,frame_op_enc,frame_op_dec,normalization,activation,loss,normalize_frame,context_length,optim)
    print(config)
    
    model,optimizer,preprocessing,train_loader,val_loader = init_run(config)
    trained_models = os.listdir(model_dir)
    if trained_models:
        model_idx = sorted([(int(i.split('_')[0]),idx) for idx,i in enumerate(trained_models)],key=lambda x:x[0])[-1][-1]
        fn = trained_models[model_idx]
        model.load_state_dict(torch.load(model_dir+fn))
        starting_epoch,global_step = int(fn.split('_')[0])+1,int(fn.split('_')[1])
        print("starting run at",starting_epoch)
    else:
        starting_epoch,global_step = 0,0
    writer = SummaryWriter(logdir = logdir)

    train(model,optimizer,train_loader,preprocessing,writer=writer,\
        epochs=40,grad_accu_step=grad_accu_step,save_freq = 1,print_step = grad_accu_step*20,\
        global_step=global_step,starting_epoch=starting_epoch,val_loader=val_loader)


if __name__ == '__main__':
    
    experiment_dir = './experiments/frame_emcoder/'
    model_dir = experiment_dir+'trained_model/'
    logdir = experiment_dir+'tb/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # if os.path.exists(logdir):
    #     shutil.rmtree(logdir)
    #     os.mkdir(logdir)
    main()