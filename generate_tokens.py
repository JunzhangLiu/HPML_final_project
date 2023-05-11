from model import *
from utils import *
from hyperparams_spec import *
from process_audio import *
from data import Preprocessing
import os
from tqdm import tqdm
import random

if __name__ == '__main__':
    random.seed(0)
    data_list = './data/'
    generate_music = './generated_music/seg_enc/'
    experiment_dir = './experiments/frame_emcoder/'
    model_dir = experiment_dir+'trained_model/'
    if not os.path.exists('./tokens/'):
        os.mkdir('./tokens/')
    grad_accu_step = 1
    
    frame_op_enc,frame_op_dec=0,1
    context_length=63
    creat_config(config,frame_op_enc,frame_op_dec,normalization,activation,loss,normalize_frame,context_length,optim)
    
    model = Spectrogram_generator(config)
    load_model(model,model_dir)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    musics = os.listdir(data_list)
    random.shuffle(musics)
    for fn in tqdm(musics):
        spec = torch.tensor(mp3_2_spectrogram(data_list+fn,hop_len = 700,n_mels=256)[0])
        music_name = fn.split('.')[0]
        spec=spec.to(device)
        data_len = spec.shape[1]
        with torch.no_grad():
            tokens = model.encode(spec.unsqueeze(0))
        np.save("./tokens/"+music_name+'.npy',tokens.detach().cpu().numpy())
        