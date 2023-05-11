from pytube import YouTube,Search
import moviepy.audio.io.AudioFileClip
import re
import os
import time
import threading
from tqdm import tqdm
key_words = ['music','playlist','mix']
determine_genre = re.compile('[a-zA-Z]+wave')
duration = re.compile('([0-9][0-9]*:)*[0-9][0-9]*:[0-9][0-9]*')
def is_tutorial(title,desc):
    return 'tutorial' in title or 'tutorial' in desc
def is_music(title,desc):
    for w in key_words:
        if w in title:
            return True
        if w in desc:
            return True
    return False
    
def download_audio(vid_id,yt:YouTube):    
    destination = './raw_data/'
    if os.path.isfile(destination+vid_id+".mp3"):
        return
    # yt = YouTube('https://www.youtube.com/watch?v='+vid_id)
    # extract only audio
    audio_files = yt.streams.filter(only_audio=True)

    best_bit_rate = 0
    best_audio = None
    for a in audio_files:
        bit_rate = int(a.abr[:-4])
        print('found bit rate', bit_rate)
        if bit_rate>best_bit_rate:
            best_audio = a
            best_bit_rate = bit_rate
    out_file = best_audio.download(output_path=destination)
    
    # save the file
    os.rename(out_file, destination+vid_id+".mp3")

def parse_time(t_str):
    t_str = t_str.split(':')
    t_str.reverse()
    time_in_sec = 0
    sec = 1
    for t in t_str:
        time_in_sec += int(t)*sec
        sec*=60
    return time_in_sec

def cut_audio(clips,audio):
    clips.append((audio.duration,None))
    for i in range(len(clips)-1):
        start,song_name = clips[i]
        targetname = './data/'+song_name+'.mp3'
        if os.path.isfile(targetname):
            continue
        # start = parse_time(start)
        # end = parse_time(clips[i+1][0])-0.1 if clips[i+1][1] is not None else clips[i+1][0]
        end,_ = clips[i+1]
        if (end - start)>600 or (end - start)<40:
            continue

        clip = audio.subclip(start,end)
        clip.write_audiofile(targetname)

def parse_link(vid_id,yt:YouTube):
    name = re.compile('')
    # yt = YouTube('https://www.youtube.com/watch?v='+vid_id)
    description = yt.description
    if description is None:
        return []
    description=description.split('\n')
    musics=[]
    i = 0
    for d in description:
        m = duration.search(d)
        if m is not None:
            i+=1
            start = m.group(0)
            name = vid_id+'@'+str(i)
            musics.append((parse_time(start),name))
            if len(musics)>1:
                if musics[i-1][0]-musics[i-2][0]<=0:
                    print('conflicting timestamp found!')
                    return []
    if not len(musics)>=5:
        if len(musics)==0 and 60<yt.length<600:
            return [(0,vid_id+'@0')]
        return []
    return musics
def process_vid(result:YouTube):
    title = result.title
    description = result.description
    vid_id = result.video_id
    try:
        result.check_availability()
    except Exception as e:
        print('video unavaliable')
        return 0,0

    loaded = False
    for _ in range(5):
        try:
            length = result.length
            loaded = True
            break
        except Exception as e:
            time.sleep(1)
        
    musics = parse_link(vid_id, result)
    title_lower,description_lower = str(title).lower(),str(description).lower()
    tutorial_vid,music_vid = is_tutorial(title_lower,description_lower),is_music(title_lower,description_lower)
    if not loaded or len(musics)==0 or tutorial_vid or not music_vid:
        if not loaded:
            message = 'failed to load'
        elif len(musics) == 0:
            message = 'no table of content found'
        else:
            message = 'not a music video'
            for _,song_name in musics:
                if os.path.isfile('./data/'+song_name+'.mp3'):
                    os.remove('./data/'+song_name+".mp3")                
        print(title, message)
        return 0,0

    if os.path.isfile('./data/'+musics[-1][1]+'.mp3'):
        print(title, 'already exists')
        return 0,0
    print('Downloading',title)
    download_audio(vid_id,result)
    audio = moviepy.audio.io.AudioFileClip.AudioFileClip('./raw_data/'+vid_id+".mp3")
    try:
        cut_audio(musics,audio)
    except Exception as e:
        print(e)
    audio.close()
    os.remove('./raw_data/'+vid_id+".mp3")
    return len(musics),length
    
def examine():
    sus = []
    corrupted = []
    num_musics,total_len = 0,0
    for music in tqdm(os.listdir('./data/')):
        try:
            audio = moviepy.audio.io.AudioFileClip.AudioFileClip('./data/'+music)
            length = audio.duration
        except Exception as e:
            corrupted.append(music)
            os.remove('./data/'+music)
            continue
        audio.close()
        if length<=30 or length//60>=10:
            sus.append(music)
            os.remove('./data/'+music)
        num_musics+=1
        total_len+=length
    with open('corrupted_audio_files.txt','a') as f:
        for c in corrupted:
            f.write(c+'\n')
    print(sus)
    total_len = int(total_len)
    return num_musics,total_len
    
def search_vid(keywords):
    s = Search(keywords)
    i = 0
    num_musics,total_len = len(os.listdir('./data/')),0
    # num_musics,total_len = examine()
    while i < len(s.results) and num_musics < 10000:
        # t = threading.Thread(target=process_vid,args=(s.results[i],))
        # t.start()
        r = s.results[i]
        vid_id = r.video_id
        video_url = 'https://www.youtube.com/watch?v='+vid_id
        vid = YouTube(video_url)
        new_musics,new_len = process_vid(vid)
        num_musics += new_musics
        total_len += new_len
        print('num music {}, total length {}:{}:{}'.format(num_musics,total_len//3600,(total_len%3600)//60,(total_len%3600)%60))
        i+=1
        if i>= len(s.results):
            s.get_next_results()
        while threading.active_count()>100:
            time.sleep(1)
if __name__=='__main__':
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('raw_data'):
        os.mkdir('raw_data')
        
    search_vid('synthwave | retrowave | sovietwave | vaporwave')
    num_musics,total_len = examine()
    print(num_musics)