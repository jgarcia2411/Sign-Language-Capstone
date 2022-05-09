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
