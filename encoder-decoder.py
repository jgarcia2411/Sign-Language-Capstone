import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from utils import *
from torch.utils.data import DataLoader

#_______________________________Helpers_______---_____________________________
spacy_eng = spacy.load('en_core_web_sm')

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

# Text Processing pipeline
english = Field(
    tokenize=tokenize_eng, lower=True, init_token='<sos>', eos_token="<eos>")

num_epochs = 4
learning_rate = 0.001
batch_size = 16

# Model hyper-parameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#_______________________________Data_______---_____________________________
# get the data then split
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'
#os.chdir("..") # Change to the parent directory if code is saved in different f.
#PATH = os.getcwd()
#DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
DATA_DIR = '/home/ubuntu/ASL'
#sep = os.path.sep
#os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory
#use entire training sentences to build vocabulary dimension

train_annotations = pd.read_csv(OR_PATH+'/how2sign_realigned_train 2.csv')

english.build_vocab(train_annotations['SENTENCE'].tolist(), max_size=10000, min_freq=2)

train_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos",
                            transform= None)
test_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_test.csv', root_dir=DATA_DIR+'/test_videos',
                            transform= None)
val_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_val.csv', root_dir=DATA_DIR+'/val_videos',
                            transform= None)

train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=True)
#test_loader

train_iterator, test_iterator, val_iterator = BucketIterator.splits(
    (train_loader, test_loader, val_loader),
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device=device
)


#_______________________________Model_______---_____________________________
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        """input_size: fixed length of 1662 keypoints extracted on each frame
            embedding_size: (100-300) recommended range
            hidden_size: 1024 predefined in training hyper-parameters
            num_layers: predefined in training hyper-parameters
            p: dropout improves network by preventing co-adaptation.. pytorch"""
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        #x shape: (seq_lenght= num of frames, N=batch size)
        embedding = self.dropout(self.embedding(x)) #shape: (seq_length, N, embedding_size
        outputs, (hidden, cell) = self.rnn(embedding) #shape: (seq_length, N, hidden_size)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout= nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        predictions = self.fc(outputs)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[[t]] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


#_____________________________________________ Training hyper-parameters__________________________________________

input_size_encoder = 1662 #length of keypoints vector on each frame
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300 #(100-300) standard
decoder_embedding_size = 300
hidden_size = 1024 # Look for this value in papers
num_layers =2
enc_dropout = 0.5
dec_dropout = 0.5

#To get training loss:
writer = SummaryWriter(f'runs/loss_plot')
step = 0

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)

vectors = process_video(DATA_DIR+'/test_videos/-g0iPSnQt6w_2-1-rgb_front.mp4') # I hope you are having fun

for epoch in range(num_epochs):
    print(f'[Epoch {epoch} / {num_epochs}')
    checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    #save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, vectors, english, device, max_length=50
    )

    print(f'Translated example sentence: \n {translated_sentence}')
    model.train()

for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1




