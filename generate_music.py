import torch
import torch.nn as nn
from hyperparams_spec import *
from utils import *
from process_audio import *
import time
from model import *
class Spectrogram_enc_dec(nn.Module):
    def __init__(self, encoder,decoder, context_length):
        super(Spectrogram_enc_dec, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.context_length = context_length

    def forward(self, spec : torch.Tensor):
        tokens = self.encoder(spec.unsqueeze(0))
        length = tokens.shape[0]
        tokens = torch.stack([tokens[i:i+self.context_length+1] for i in range(length-self.context_length-1)])
        pred = self.decoder(tokens)
        return pred
class Token_gen(nn.Module):
    def __init__(self, model):
        super(Token_gen, self).__init__()
        self.model = model

    def forward(self, inputs : torch.Tensor):
        return self.model.decode(inputs)
    
class Spec_enc(nn.Module):
    def __init__(self, model):
        super(Spec_enc, self).__init__()
        self.model = model

    def forward(self, inputs : torch.Tensor):
        return self.model.encode(inputs)
    
class Spec_dec(nn.Module):
    def __init__(self, model):
        super(Spec_dec, self).__init__()
        self.model = model

    def forward(self, inputs : torch.Tensor):
        return self.model.decode(inputs)
    
class Music_generator(nn.Module):
    def __init__(self, spec_enc,spec_dec,token_generator, s_context_length, t_context_length):
        super(Music_generator, self).__init__()
        self.spec_enc = spec_enc
        self.spec_dec = spec_dec
        self.token_generator=token_generator
        self.s_context_length = s_context_length
        self.t_context_length = t_context_length

    def forward(self, spec : torch.Tensor, target_length : int):
        tokens = self.spec_enc(spec.unsqueeze(0))
        spec_inputs = []
        for i in range(target_length):
            pred_token = self.token_generator(tokens[i:i+self.t_context_length].unsqueeze(0))
            tokens = torch.cat([tokens,pred_token])
            spec_inputs.append(tokens[-1-self.s_context_length:])
        spec_inputs = torch.stack(spec_inputs)
        return self.spec_dec(spec_inputs)

def load_models(frame_enc_dir,auto_enc_dir):
    frame_op_enc,frame_op_dec=0,1
    context_length=63
    config = creat_config(frame_op_enc,frame_op_dec,normalization,activation,loss,normalize_frame,context_length,optim)
    config['lr'] = lr
    config['num_embeddings']=num_embedding
    config['input_channels']=257 if config['normalize_frame'] else 256
    config['hidden_channels'] = hidden_channel
    config['num_blocks'] = 10
    config['input_frames']=63
    config['num_worker'] = 5
    config['num_heads'] = 8
    spec_generator = Spectrogram_generator(config)
    trained_models = os.listdir(frame_enc_dir)
    if trained_models:
        model_idx = sorted([(int(i.split('_')[0]),idx) for idx,i in enumerate(trained_models)],key=lambda x:x[0])[-1][-1]
        fn = trained_models[model_idx]
        spec_generator.load_state_dict(torch.load(frame_enc_dir+fn))
    spec_generator.to(device)
    token_generator = Token_generator(2048,768)
    trained_model = os.listdir(auto_enc_dir)
    if trained_model and trained_model!=['0_0']:
        model_idx = sorted([(int(i.split('_')[0]),idx) for idx,i in enumerate(trained_model)],key=lambda x:x[0])[-1][-1]
        fn = trained_model[model_idx]
        token_generator.load_state_dict(torch.load(auto_enc_dir+fn))
    token_generator.to(device)
    return spec_generator,token_generator,config

def create_traced_model(spec_generator,token_generator):
    
    test_spec = torch.zeros((1,256,64)).to(device)
    test_token1 = torch.zeros((1,64),dtype=torch.long).to(device)
    test_token2 = torch.zeros((1,63*10),dtype=torch.long).to(device)
    encoder = Spec_enc(spec_generator)
    decoder = Spec_dec(spec_generator)
    token_gen = Token_gen(token_generator)
    traced_encoder = torch.jit.trace(encoder,(test_spec,))
    traced_decoder = torch.jit.trace(decoder,(test_token1,))
    traced_token_gen = torch.jit.trace(token_gen,(test_token2,))  
    
    return traced_encoder,traced_decoder,traced_token_gen,encoder,decoder,token_gen
def get_spectrogram(config):
    data_list = './data/'
    musics = os.listdir(data_list)
    random.shuffle(musics)
    
    fn = musics[0]
    audio,sr = load_audio(data_list+fn)
    num_secs = 60
    start = 34*sr
    length = sr*num_secs
    audio = audio[start-(config['input_frames']//(sr//hop_len))*sr:start+length]
    spec = torch.tensor(audio_2_spectrogram(audio,sr,hop_len,256)).to(device)
    return spec

def generate_music(model,args,model_name,fn = None):
    start = time.time()
    with torch.no_grad():
        pred = model(*args)
    spec_generating_time = time.time()
    if fn is not None:
        mel_to_audio(pred.detach().cpu().numpy(),'./generated_music/'+fn+'.wav',44100,hop_len=700)
        audio_generating_time = spec_generating_time- time.time()
    else:
        audio_generating_time='N/A'
    print("Model: {}\n\
        total time {}\n\
        time to generate spectrogram {}\n\
        time to generate audio {}".format(model_name,audio_generating_time-start,spec_generating_time-start,audio_generating_time))
    
    
if __name__ == "__main__":
    if not os.path.exists('./generated_music/'):
        os.mkdir('./generated_music/')
    random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    spec_generator,token_generator,config = load_models('./experiments/frame_emcoder/trained_model/',\
                                                        './experiments/auto_enc/trained_model/')
    
    
    traced_encoder,traced_decoder,traced_token_gen,encoder,decoder,token_gen = create_traced_model(spec_generator,token_generator)
    m1 = Spectrogram_enc_dec(traced_encoder,traced_decoder,63)
    m1 = torch.jit.script(m1)    
    
    m2 = Spectrogram_enc_dec(encoder,decoder,63)
    
    m3 = Music_generator(traced_encoder,traced_decoder,traced_token_gen,63,63*10)
    m3 = torch.jit.script(m3)
    m4 = Music_generator(encoder,decoder,token_gen,63,63*10)
    
    spec = get_spectrogram(config)
    mel_to_audio(spec.detach().cpu().numpy(),'./generated_music/original.wav',44100,hop_len=700)
    print('torch script generated')
    generate_music(m1,(spec,),'Scripted frame encoder',"scripted_frame_encoder")
    generate_music(m2,(spec,),'Frame encoder')
    generate_music(m3,(spec,63*60),'Scripted music generator','scripted_music_generator')
    generate_music(m4,(spec,63*60),'Music generator')