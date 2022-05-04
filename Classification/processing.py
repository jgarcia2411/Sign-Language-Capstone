import os
import torch
import torchvision
import torchvision.transforms as T
import numpy as np
from multiprocessing import Pool

H2S_TR = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train_videos')][0:1000] #l1
H2S_t = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/test_videos')][0:1000] #l2
H2S_ev = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/val_videos')][0:100] #l3

TKT_SIGN = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train/sign')] #~222 #l4
TKT_NO_SIGN = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train/no-sign')] #l5


frames_num = 100
OR_PATH = '/home/ubuntu/ASSINGMENTS/Final_project'

transforms = [T.Resize((100,100))]
frame_transform = T.Compose(transforms)

def preprocess(batch):
    """ Process frames"""
    transforms = T.Compose(
        [
            T.Resize((100,100)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),
        ]
    )
    batch = transforms(batch)
    return batch


def process_target(x):
    if x=='sign':
        return np.array(1)
    else:
        return np.array(0)

def get_frames(video):
    num = frames_num
    if video.shape[0] < num:
        while video.shape[0] < num:
            video = torch.cat((video, video[-1].unsqueeze(0)))
    elif video.shape[0] == num +1:
        video = video[:num]
    else:
        lower = round((video.shape[0]-num)/2) #
        upper = lower + num
        video = video[lower:upper]

    return video

def process_l1(video, path= '/home/ubuntu/ASL/train_videos'):
    v, a, _ = torchvision.io.read_video(path+f'/{video}.mp4')
    del a, _
    v = frame_transform(v.view(v.shape[0], v.shape[-1], v.shape[1], v.shape[2]))
    v = get_frames(v)
    torchvision.io.write_video(f'/home/ubuntu/ASL/clips/sign/{video}.mp4',
                               v.view(v.shape[0], v.shape[2], v.shape[-1], v.shape[1]),
                               fps=25)

def process_l2(video, path= '/home/ubuntu/ASL/test_videos'):
    v, a, _ = torchvision.io.read_video(path + f'/{video}.mp4')
    del a, _
    v = frame_transform(v.view(v.shape[0], v.shape[-1], v.shape[1], v.shape[2]))
    v = get_frames(v)
    torchvision.io.write_video(f'/home/ubuntu/ASL/clips/sign/{video}.mp4',
                               v.view(v.shape[0], v.shape[2], v.shape[-1], v.shape[1]),
                               fps=25)

def process_l3(video, path= '/home/ubuntu/ASL/val_videos'):
    v, a, _ = torchvision.io.read_video(path + f'/{video}.mp4')
    del a, _
    v = frame_transform(v.view(v.shape[0], v.shape[-1], v.shape[1], v.shape[2]))
    v = get_frames(v)
    torchvision.io.write_video(f'/home/ubuntu/ASL/clips/sign/{video}.mp4',
                               v.view(v.shape[0], v.shape[2], v.shape[-1], v.shape[1]),
                               fps=25)

def process_l4(video, path= '/home/ubuntu/ASL/train/sign'):
    v, a, _ = torchvision.io.read_video(path + f'/{video}.mp4')
    del a, _
    v = frame_transform(v.view(v.shape[0], v.shape[-1], v.shape[1], v.shape[2]))
    v = get_frames(v)
    torchvision.io.write_video(f'/home/ubuntu/ASL/clips/sign/{video}.mp4',
                               v.view(v.shape[0], v.shape[2], v.shape[-1], v.shape[1]),
                               fps=25)

def process_l5(video, path= '/home/ubuntu/ASL/train/no-sign'):
    v, a, _ = torchvision.io.read_video(path + f'/{video}.mp4')
    del a, _
    v = frame_transform(v.view(v.shape[0], v.shape[-1], v.shape[1], v.shape[2]))
    v = get_frames(v)
    torchvision.io.write_video(f'/home/ubuntu/ASL/clips/no-sign/{video}.mp4',
                               v.view(v.shape[0], v.shape[2], v.shape[-1], v.shape[1]),
                               fps=25)

p = Pool(8)
p.map(process_l1, H2S_TR)
p.map(process_l2, H2S_t)
p.map(process_l3, H2S_ev)
p.map(process_l4, TKT_SIGN)
p.map(process_l5, TKT_NO_SIGN)




#video, audio, _ = torchvision.io.read_video(f'/home/ubuntu/ASL/train_videos/{video_path}.mp4')
#del audio, _
#video = frame_transform(video.view(video.shape[0], video.shape[-1], video.shape[1], video.shape[2]))
#video = get_frames(video)
#torchvision.io.write_video('/home/ubuntu/ASL/clips/video.mp4', video.view(video.shape[0],video.shape[2], video.shape[-1], video.shape[1]),
#                           fps=25)