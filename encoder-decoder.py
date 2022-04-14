import random

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, BucketIterator
import torchtext
import numpy as np
import spacy
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from utils import *
from torch.utils.data import DataLoader
from torchtext.legacy.datasets import Multi30k
os.system("export CUDA_VISIBLE_DEVICES=''")

#_______________________________Helpers_______---_____________________________

#_________________I used this section only to build a vocabulary the same way in the video__________________________
spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

# This is what we should use to tokenize sentence
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

#
SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field( tokenize = tokenize_en,
            init_token='<sos>',eos_token='<eos>',
             lower=True)

## ____________________________________temporal files to build vocab_______________\
vocab_data, non_vocab, temporal = Multi30k.splits(exts =('.de', '.en'),
                                                  fields = (SRC, TRG))
TRG.build_vocab(vocab_data, min_freq = 2) ## HERE IS WHERE WE BUILD A VOCABULARY: SIZE ~5388 words. This missmatch
# with tokens ids > 5388. Token ids are obtained in the utils.py script using "bert". We should fix that processing part
# using spacy model.

print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
# ___________________________________________________________________________________

# ______________________________________________Training configuration__________________________________________

num_epochs = 4
learning_rate = 0.001
batch_size = 1

# Model hyper-parameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#_______________________________Data_______---_____________________________
# get the data then split
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'

DATA_DIR = '/home/ubuntu/ASL'

train_annotations = pd.read_csv(OR_PATH+'/how2sign_realigned_train 2.csv')

# signvideosDataset is a class defined in utils script. This is how pytorch works to get and process data efficiently.
# we can modify OR_PATH and DATA_DIR to work in local computer with a small sample of videos.

train_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/",
                            transform= None)
test_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_test.csv', root_dir=DATA_DIR+'/test_videos/',
                            transform= None)
val_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_val.csv', root_dir=DATA_DIR+'/val_videos/',
                            transform= None)
# Data Loader is a pytorch function that give us data distributed in batches-> this is useful to don't crash memory
train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=True)

#_______________________________Model Architecture ___________________________________________________________________
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        """input_size: fixed length of 1662 keypoints extracted on each frame
            embedding_size: (100-300) recommended range
            hidden_size: 1024 predefined in training hyper-parameters
            num_layers: predefined in training hyper-parameters
            p: dropout improves network by preventing co-adaptation.. pytorch"""
        super(Encoder, self).__init__()
        #Input one vector of 1662 keypoints from a sequence of #frames
        self.input_size = input_size

        #Output size of embedding
        self.embedding_size = embedding_size

        # Dimension of the NNs inside the lstm cell/ (hs,cs)'s dimension
        self.hidden_size = hidden_size

        # Regularization parameter
        self.dropout = nn.Dropout(p)
        self.tag = True

        # Number of layers in the LSTM
        self.num_layers = num_layers

        # [input size, output size]---> 1662, 300
        #self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers, dropout=p)

    # Shape [sequence_length: #frames, batch_size]
    def forward(self, x):
        # Shape----> (#sequencelength, batch_size, embeddings dims)
        outputs, (hidden, cell) = self.rnn(x)
        # outputs = [sen_len, batch_size, hid_dim*]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        # input_size : size english vocabulary
        # output_size: same input_size
        super(Decoder, self).__init__()
        self.dropout= nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size) # english word -> embedding
        # embedding gives shape: (1,batch_Size,embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, cell):
        # x = [batch_size]
        # hidden = [n_layers*n_dir, batch_size, hid_dim]
        # cell = [n_layers*n_dir, batch_size, hid_dim]

        x = x.unsqueeze(0) # x = [1, , batchsize]

        embedding = self.dropout(self.embedding(x)) # embedding = [1, batch_size, emb_dim]

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs = [seq_len, batch_size, hid_dime * n_dir]
        # hidden = [n_layers*n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        predictions = self.fc(outputs.squeeze(0)) #shape: (1, Batch_Size, length_target_vocab)
        # predictions = [batch_size, output_dim]

        #predictions = predictions.squeeze(0) #shape: (N, length_target_vocab)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        # src = [sen_len, batch_size]
        # trg = [sen_len, bach_size]
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(TRG.vocab) #len(english.vocab) -> to 5000 just to try

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)#.to(device)

        hidden, cell = self.encoder(source) #vector context 1024 dimension.

        x = target[0, :]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_force_ratio

            best_guess = output.argmax(1)

            x = target[t] if teacher_force else best_guess

        return outputs


#_____________________________________________ Training hyper-parameters__________________________________________

input_size_encoder = 1662
input_size_decoder = len(TRG.vocab)
output_size = len(TRG.vocab)
encoder_embedding_size = 300 #(100-300) standard
decoder_embedding_size = 300
hidden_size = 1024 # Look for this value in papers
num_layers =1
enc_dropout = 0.5
dec_dropout = 0.5

#To get training loss:
writer = SummaryWriter(f'runs/loss_plot')
step = 0

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
)##.to(device)

decoder_net = Decoder(
    input_size_decoder,
    decoder_embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout
)#.to(device)

model = Seq2Seq(encoder_net, decoder_net)#.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)


for epoch in range(num_epochs):
    print(f'[Epoch {epoch} / {num_epochs}')
    checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader): #train_iterator
            # Quick reshape input_data to LSTM:
            inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[0], inputs.shape[-1]))
            labels = torch.reshape(labels, (labels.shape[-1], labels.shape[0]))
            # Get input and targets and get to cuda
            inp_data = inputs#.to(device)
            target = labels#.to(device)

            # Forward prop

            #try:
            #print(">>>", inp_data.shape, target.shape)
            print(torch.max(inp_data),torch.min(inp_data),torch.max(target),torch.min(target))
            output = model(inp_data, target)

            # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
            # doesn't take input in that form. For example if we have MNIST we want to have
            # output to be: (N, 10) and targets just (N). Here we can view it in a similar
            # way that we have output_words * batch_size that we want to send in into
            # our cost function, so we need to do some reshapin. While we're at it
            # Let's also remove the start token while we're at it
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim) #output[1:]
            target = target[1:].view(-1) #target[1:]

            optimizer.zero_grad()


            loss = criterion(output, target) ######last bug is here, shapes of target and output

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
            print("Yay")
            #except:
                ### log the video that generated the error
                #print(":-(", )




