import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from random import sample
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# _________________________________________________________________________________________
random.seed(1234)
training_videos = 1000
eval_videos = 50
test_videos = 100
channels = 3 #Gray scale frames
frames_num = 60
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
VIDEOS_DIRECTORY = '/home/ubuntu/ASL/clips'
SIGN = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/clips/sign')]
NO_SIGN = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/clips/no-sign')]

df1 = pd.DataFrame({
    'VIDEO': pd.Series(sample(SIGN, training_videos)),
    'DIRECTORY': '/home/ubuntu/ASL/clips/sign/',
    'CLASS': 'sign'
})

df2 = pd.DataFrame({
    'VIDEO': pd.Series(NO_SIGN),
    'DIRECTORY': '/home/ubuntu/ASL/clips/no-sign/',
    'CLASS': 'no_sign'
})

annotations = pd.concat([df1, df2], ignore_index=True)
train, test = train_test_split(annotations, test_size=0.2, random_state=1234)
train['SPLIT'] = 'train'
test['SPLIT'] = 'test'
annotations = pd.concat([train,test], ignore_index=True)
annotations.reset_index(inplace=True, drop=True)


#H2S_TR = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train_videos')]
#H2S_t = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/test_videos')]
#H2S_ev = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/val_videos')]

#TKT_SIGN = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train/sign')] #~222
#TKT_NO_SIGN = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train/no-sign')] #~981

#df1 = pd.DataFrame({
#    'VIDEO': pd.Series(H2S_TR[:training_videos-len(TKT_SIGN)]),
#    'DIRECTORY': '/home/ubuntu/ASL/train_videos/',
#    'CLASS': 'sign'
#})

# df2 = pd.DataFrame({
#     'VIDEO': pd.Series(TKT_SIGN),
#     'DIRECTORY': '/home/ubuntu/ASL/train/sign/',
#     'CLASS': 'sign'
# })
#
# df3 = pd.DataFrame({
#     'VIDEO': pd.Series(TKT_NO_SIGN[:len(TKT_NO_SIGN)-eval_videos-test_videos]),
#     'DIRECTORY': '/home/ubuntu/ASL/train/no-sign/',
#     'CLASS': 'no_sign'
# })

# train_df = pd.concat([df1, df2, df3], ignore_index=True)
# train_df = shuffle(train_df)
# train_df.reset_index(inplace=True, drop=True)
# _____________________________________________________________________________________________________________________

# df4 = pd.DataFrame({
#     'VIDEO': pd.Series(H2S_ev[:eval_videos]),
#     'DIRECTORY': '/home/ubuntu/ASL/val_videos/',
#     'CLASS': 'sign'
# })
#
# df5 = pd.DataFrame({
#     'VIDEO': pd.Series(TKT_NO_SIGN[len(TKT_NO_SIGN)-eval_videos-test_videos: len(TKT_NO_SIGN)-test_videos]),
#     'DIRECTORY': '/home/ubuntu/ASL/train/no-sign/',
#     'CLASS': 'no_sign'
# })
#
# eval_df = pd.concat([df4, df5], ignore_index=True)
# eval_df = shuffle(eval_df)
# eval_df.reset_index(inplace=True, drop=True)
# # ______________________________________________________________________________________________________________________
#
# df6 = pd.DataFrame({
#     'VIDEO': pd.Series(H2S_t[:test_videos]),
#     'DIRECTORY': '/home/ubuntu/ASL/test_videos/',
#     'CLASS': 'sign'
#
# })
# df7 = pd.DataFrame({
#     'VIDEO': pd.Series(TKT_NO_SIGN[len(TKT_NO_SIGN)-test_videos:]),
#     'DIRECTORY': '/home/ubuntu/ASL/train/no-sign/',
#     'CLASS': 'no_sign'
# })
#
# test_df = pd.concat([df6, df7], ignore_index=True)
# test_df = shuffle(test_df)
# test_df.reset_index(inplace=True, drop=True)
# # ______________________________________________________________________________________________________________________

transforms = [T.Normalize(0.5,0.5)]
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

# ______________________________________________________________________________________________________________________

class dataloader(Dataset):
    def __init__(self, keyword, annotations=annotations,transform=None):
        self.keyword = keyword
        self. annotations = annotations
        # if self.keyword == 'train':
        #     #self.annotations = train_df
        # elif self.keyword == 'eval':
        #     self.annotations = eval_df
        # elif self.keyword == 'test':
        #     self.annotations = test_df
        # else:
        #     print('SPECIFY KEYWORD: train, eval or test')
        self.annotations = self.annotations[self.annotations['SPLIT'] == self.keyword]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #get path
        video_path = os.path.join(self.annotations.iloc[index, 1],
                                  self.annotations.iloc[index, 0])
        y_label = self.annotations.iloc[index, -2]
        y_label = process_target(y_label)

        #read video
        vid, audio,_ = torchvision.io.read_video((video_path+'.mp4'))
        del audio
        del _
        #transformation
        if self.transform:
            frames_processed = self.transform(vid.view(vid.shape[0], vid.shape[-1], vid.shape[1], vid.shape[2]).float())
            #frames_processed = get_frames(frames_processed)
            frames_processed = frames_processed/torch.max(frames_processed)
        else:
            #frames = [f for f in vid]
            #frames_processed = preprocess(torch.stack(frames))
            #frames_processed = get_frames(frames_processed)
            frames_preprocessed = vid.view(vid.shape[0], vid.shape[-1], vid.shape[1], vid.shape[2])
            frames_preprocessed = frames_preprocessed.float()/torch.max(frames_preprocessed.float())

        return frames_processed ,torch.tensor(y_label).float()



class collate_batch:
    def __init__(self, frames_idx, device):
        self.frames_ids = frames_idx
        self.device = device

    def __call__(self, batch):
        self.frames_list = []
        self.labels_list = []
        for (_frames, _labels) in batch:
            self.frames_list.append(_frames)
            self.labels_list.append(_labels)

        self.frames_list = pad_sequence(self.frames_list, batch_first=False, padding_value=self.frames_ids)
        return self.frames_list, torch.stack([i for i in self.labels_list])

def get_loader(
        keyword,
        transform=frame_transform,
        batch_size=1,
):
    dataset = dataloader(keyword, transform=transform)
    frames_ids = 1.0 #
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        num_workers= 0)
                        #collate_fn=collate_batch(frames_idx=frames_ids, device=device), num_workers=4)
    if keyword == 'train':
        return loader, dataset
    else:
        return loader

#batch_size = 2
#loader, dataset = get_loader(keyword='train', batch_size=batch_size, transform=frame_transform)
#for batch_idx, (inputs, labels) in enumerate(loader):
#    print(f'Batch number {batch_idx} \n Inputs Shape {inputs.shape} \n Labels Shape {labels.shape}')





