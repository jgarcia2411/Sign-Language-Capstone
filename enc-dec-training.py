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
import math

from torch.utils.data import DataLoader
from torchtext.legacy.datasets import Multi30k
os.system("export CUDA_VISIBLE_DEVICES=''")

#_______________________________Helpers_______---_____________________________

# ______________________________________________Training configuration__________________________________________

num_epochs = 2
learning_rate = 0.001
batch_size = 20

# Model hyper-parameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#_______________________________Data_______---_____________________________
# get the data then split
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'

DATA_DIR = '/home/ubuntu/ASL'

glove_path = '/home/ubuntu/ASSINGMENTS'

loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", keyword='train', batch_size=batch_size)

val_loader, val_dataset = get_loader(OR_PATH+'/how2sign_realigned_val.csv', root_dir=DATA_DIR+"/val_videos/", keyword='val', batch_size=batch_size)

test_loader, test_dataset = get_loader(OR_PATH+'/how2sign_realigned_test.csv', root_dir=DATA_DIR+"/test_videos/", keyword= 'test', batch_size=1)

print(f'your vocabulary size is: {len(dataset.vocab)}')
print(f'your vocabulary size is: {len(val_dataset.vocab)}')
print(f'your vocabulary size is: {len(test_dataset.vocab)}')

#__________________________________________Helper funcion to use Word2Vec_______________________________________________
#idea taken from: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
# https://androidkt.com/pre-train-word-embedding-in-pytorch/


words = []
idx = 0
word2idx = {}
glove = pd.read_csv(f'{glove_path}/glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove_embedding = {key: val.values for key, val in glove.T.items()}

matrix_len = len(dataset.vocab)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

for i, word in enumerate(dataset.vocab):
    try:
        weights_matrix[i] =



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
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(dataset.vocab) #len(english.vocab) -> to 5000 just to try

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

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
input_size_decoder = len(dataset.vocab)
output_size = len(dataset.vocab)
encoder_embedding_size = 300 #(100-300) standard
decoder_embedding_size = 300
hidden_size = 256 # Look for this value in papers
num_layers =1
enc_dropout = 0.5
dec_dropout = 0.5


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

sentences, translations = translate_video(model, test_loader, device, dataset)
for i in sentences:
    for j in translations:
        print(f'Real sentence: \n {i} \n Translated:sentence: \n {j}')



def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch_idx, (inputs, labels) in enumerate(iterator):
        #inputs = inputs.view(inputs.shape[1], inputs.shape[0], inputs.shape[-1])
        inp_data = inputs.to(device)
        target = labels.to(device)
        optimizer.zero_grad()

        output = model(inp_data, target)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)  # output[1:]
        target = target[1:].view(-1)  # target[1:]
        loss = criterion(output,target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss+= loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(iterator):
            #inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[0], inputs.shape[-1]))
            inp_data = inputs.to(device)
            target = labels.to(device)

            output = model(inp_data, target, 0)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)  # output[1:]
            target = target[1:].view(-1)  # target[1:]
            loss = criterion(output, target)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


CLIP = 1

best_valid_loss = float('inf')

for epoch in range(num_epochs):
    print(f'Epoch {epoch}  training')

    train_loss = train(model, loader, optimizer, criterion, CLIP)
    print(f'Epoch {epoch}  evaluating')
    valid_loss = evaluate(model, val_loader, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model_{}.pt'.format('SIGN2TEXT'))
        print("The model has been saved!")
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


sentences, translations = translate_video(model, test_loader, device, dataset)
for i in sentences:
    for j in translations:
        print(f'Real sentence: \n {i} \n Translated:sentence: \n {j}')







