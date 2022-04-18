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

loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", batch_size=batch_size)

#__________________________________________Past version to load images__________________________________________________
# signvideosDataset is a class defined in utils script. This is how pytorch works to get and process data efficiently.
# we can modify OR_PATH and DATA_DIR to work in local computer with a small sample of videos.

#train_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/",
#                            transform= None)
#test_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_test.csv', root_dir=DATA_DIR+'/test_videos/',
#                           transform= None)
#val_dataset = signvideosDataset(csv_file=OR_PATH+'/how2sign_realigned_val.csv', root_dir=DATA_DIR+'/val_videos/',
#                            transform= None)
# Data Loader is a pytorch function that give us data distributed in batches-> this is useful to don't crash memory
#train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)
#val_loader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=True)

#_______________________________Model Architecture ___________________________________________________________________

#
#  Attention & Transformation
"""
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(query)  # (N, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        # n:batch size, q:queue, h:heads, d: heads'dimension, k: key length
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        # Triangular Matrix
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)  # Other Options: BachNorm
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
"""
#Encoder & Decoder
class Encoder(nn.Module):
    def __init__(self,
                 input_size,     #input_size: fixed length of 1662 keypoints/landmarkers from openCV
                 embedding_size, #100-300 recommended range
                 hidden_size,    #1024 predefined in training hyper-parameters, according to papers
                 num_layers,     #predefined in training hyper-parameters
                 p):             #dropout improves network by preventing co-adaptation.. pytorch

        super(Encoder, self).__init__()
        #Input one vector of 1662 keypoints/landmarkers from a sequence of each frame
        self.input_size = input_size

        #Output size of embedding
        self.embedding_size = embedding_size

        # Dimension of the NNs inside the lstm cell/ (hs,cs)'s dimension
        self.hidden_size = hidden_size



        # Number of layers in the LSTM
        self.num_layers = num_layers

        # [input size, output size]---> 1662, 300
        #self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers, bidirectional = True)
        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

        # Regularization parameter
        self.dropout = nn.Dropout(p)
        self.tag = True


    # Shape [sequence_length: #frames, batch_size]
    def forward(self, x): #encoder_states = outputs

        # Shape----> (#sequencelength, batch_size, embeddings dims)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        # input_size : size english vocabulary
        # output_size: same input_size
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size)  # english word -> embedding
        # embedding gives shape: (1,batch_Size,embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2, embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states, hidden, cell):
        # x = [batch_size]
        # hidden = [n_layers*n_dir, batch_size, hid_dim]
        # cell = [n_layers*n_dir, batch_size, hid_dim]

        x = x.unsqueeze(0)  # x = [1, , batchsize]

        embedding = self.dropout(self.embedding(x))  # embedding = [1, batch_size, emb_dim]

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))
        # energy: (seq_length, N, 1)
        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        attention = attention.permute(1, 2, 0)
        # attention: (seq_length, N, 1), snk

        encoder_states = encoder_states.permute(1, 0, 2)
        # encoder_states: (seq_length, N, hidden_size*2), snl

        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        # outputs = [seq_len, batch_size, hid_dime * n_dir]
        # hidden = [n_layers*n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]

        predictions = self.fc(outputs.squeeze(0))  # shape: (1, Batch_Size, length_target_vocab)
        # predictions = [batch_size, output_dim]

        # predictions = predictions.squeeze(0) #shape: (N, length_target_vocab)

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
        target_vocab_size = len(dataset.vocab) #len(english.vocab) -> to 5000 just to try

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source) #vector context 1024 dimension.

        x = target[0, :]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x,encoder_states, hidden, cell)
            outputs[t] = output

            teacher_force = random.random() < teacher_force_ratio

            best_guess = output.argmax(1)

            x = target[t] if teacher_force else best_guess

        return outputs


#_____________________________________________ Training hyper-parameters__________________________________________

input_size_encoder = 1662
input_size_decoder = len(dataset.vocab)
output_size = len(dataset.vocab)
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

pad_idx = dataset.vocab.stoi['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)


for epoch in range(num_epochs):
    print(f'[Epoch {epoch} / {num_epochs}')
    checkpoint = {'state_dict': model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.train()

    for batch_idx, (inputs, labels) in enumerate(loader): #train_iterator
            # Quick reshape input_data to LSTM:
            inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[0], inputs.shape[-1]))
            #inputs = torch.cat(inputs).view(len(inputs), -1, 1)
            #labels = torch.reshape(labels, (labels.shape[-1], labels.shape[0]))
            # Get input and targets and get to cuda
            inp_data = inputs.to(device)
            target = labels.to(device)

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

            print(f'Batch # {batch_idx}, Training Loss = {loss}')

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




