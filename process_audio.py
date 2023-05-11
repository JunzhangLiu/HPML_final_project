import os
import librosa
import soundfile as sf
import random
import shutil
import numpy as np
from tqdm import tqdm
import pickle

os.environ["PATH"]='./env/ffmpeg/bin;'+os.environ["PATH"]
def load_audio(path_to_file,mono=True)->np.ndarray:
    audio, sr = librosa.load(path_to_file,sr=None,mono=mono)
    return audio,sr
def audio_2_spectrogram(audio,sr,hop_len = 512, n_mels=256,power=1):
    spec= librosa.feature.melspectrogram(y=audio, sr=sr,hop_length=hop_len,n_mels=n_mels,power=power)
    return spec
def mp3_2_spectrogram(path_to_file, hop_len = 512, n_mels = 256, mono=True):
    audio,sr = load_audio(path_to_file,mono=mono)
    spec = audio_2_spectrogram(audio,sr,hop_len = hop_len, n_mels = n_mels)
    return spec,sr
def mel_to_audio(spectrogram,fn,sr,hop_len=512,n_itr=64,power=1):
    res = librosa.feature.inverse.mel_to_audio(spectrogram,sr=sr,hop_length=hop_len,n_iter=n_itr,power=power)
    sf.write(fn, res, sr, 'PCM_24')
def sample(n=5):
    musics = os.listdir('./data/')
    sampled = random.sample(range(len(musics)), n)
    for i in range(n):
        shutil.copy('./data/'+musics[sampled[i]],'./test/{}.mp3'.format(i))
if __name__=='__main__':
    # sample()
    random.seed(10)
    hoplen = 700
    n_mels = 512
    one_sec = 44100//hoplen
    save_dir = './training_data/'
    musics = os.listdir('./data/')
    max_val= float('-inf')
    
    N = 0
    processed_files = set()    
    if not os.path.exists('processed_files.txt'):
        with open('processed_files.txt','wb+') as f:
            pickle.dump(processed_files,f)
    with open('processed_files.txt','rb') as f:
        processed_files = pickle.load(f)
    if not os.path.exists('non_0_idx.npy'):
        non_0_idx = np.zeros(n_mels,dtype=bool)
    else:
        non_0_idx = np.load('non_0_idx.npy')
    if not os.path.exists('tot_len.txt'):
        with open('tot_len.txt','w+') as f:
            f.write(str(0))

    with open('tot_len.txt','r') as f:
        N=int(f.readline())

    for fn in tqdm(musics):
        if fn in processed_files:
            continue
        file_path = './data/'+fn
        processed_files.add(fn)
        try:
            audio,sr = load_audio(file_path)
            if audio.shape[-1]//sr//60>=10:
                continue
            spec = audio_2_spectrogram(audio,sr,hop_len = hoplen, n_mels=n_mels,power=1)
            num_sec=spec.shape[-1]//one_sec+1
            if num_sec>600 or num_sec<60:
                os.remove(file_path)
                continue
            non_0_idx[np.nonzero(spec.sum(1))]=1

        except Exception as e:
            print(e)
            os.remove(file_path)
        N += spec.shape[-1]
    print(N)
    
    with open('processed_files.txt','wb+') as f:
        pickle.dump(processed_files,f)

    non_0_idx = np.save('non_0_idx.npy',non_0_idx)

    with open('tot_len.txt','w+') as f:
        f.write(str(N))