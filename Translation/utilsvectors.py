import numpy as np
from torchtext.legacy.data import Field
import spacy
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchtext.data.metrics import bleu_score
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import contractions


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# ________________________________________Data+Paths__________________________________________________
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage/'
DATA_DIR = '/home/ubuntu/ASL/vectors/'
glove_path = '/home/ubuntu/ASSINGMENTS'
training_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train_videos')]

# ____________________________________Processing functions ____________________________________
spacy_en = spacy.load('en_core_web_sm')
punctuations = """!()-[]{};:\,<>./@#$%^&*_~...0123456789âÂ+-¦Ã©´¾½\x80\x99\x9c\x9d"""
def cleaner(sentence, punctuations):
    no_punc = ''
    for char in sentence:
        if char not in punctuations:
            no_punc = no_punc + char

    return no_punc

def clean_contractions(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    return expanded_text




# ___________________________________Vocabulary________________________________________________
class Vocabulary:
    def __init__(self, annotations_path=OR_PATH+'/how2sign_realigned_train 2.csv'):
        """freq_threshold: frequency of words to build vocabulary
        itos: index to string
        stoi: string to index"""
        self.itos ={0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>": 3}
        self.dataframe = pd.read_csv(annotations_path)
    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def build_vocabulary(self):
        self.dataframe['clean_text'] = self.dataframe['SENTENCE'].apply(lambda x: clean_contractions(x))
        self.dataframe['clean_text'] = self.dataframe['clean_text'].apply(lambda x: cleaner(x,punctuations))
        sentences_list = self.dataframe['clean_text'].tolist()
        tokens_list = [self.tokenizer_eng(i) for i in sentences_list]
        idx = 4  # 3:<unk>
        frequencies = {}
        for sentence in tokens_list:
            for token in sentence:
                if token not in frequencies:
                    frequencies[token] = 1
                else:
                    frequencies[token] += 1
                if frequencies[token] ==3:
                #if token not in self.stoi:
                    self.itos[idx] = token
                    self.stoi[token] = idx
                    idx += 1
                else:
                    pass
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


# _________________________________Data Loader ___________________________________________

class signvideosDataset(Dataset):
    def __init__(self, keyword, csv_file=OR_PATH+'/how2sign_realigned_train 2.csv'): #vocabulary_list
        # read entire annotations files to build vocabulary:
        self.annotations = pd.read_csv(csv_file)
        self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(training_videos)]
        self.annotations['clean_text'] = self.annotations['SENTENCE'].apply(lambda  x: clean_contractions(x))
        self.annotations['clean_text'] = self.annotations['clean_text'].apply(lambda x: cleaner(x, punctuations))
        self.vocab = Vocabulary()
        self.vocab.build_vocabulary()
        self.keyword = keyword
        self.train_df, self.test_df = train_test_split(self.annotations, test_size=0.2, random_state=1234)
        self.train_df.reset_index(inplace=True, drop=True)
        self.test_df.reset_index(inplace=True, drop=True)
        if self.keyword == 'train':
            self.df = self.train_df
        else:
            self.df = self.test_df.sample(100)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        #Read path to process with processing function:
        #vectors_path = os.path.join(self.root_dir, self.df.iloc[index, 3])
        vectors_path = self.df.iloc[index, 3]
        coordinates = np.load(DATA_DIR+vectors_path+'.npy')
        np.random.seed(55)
        if coordinates.shape[0]>100:
            compressed = np.array([coordinates[i] for i in sorted(np.random.choice(np.array([i for i in range(coordinates.shape[0])]), 100, replace=False))])
        else:
            compressed = coordinates

        # Reads sentence to process with tokenizer
        y_label = self.df.iloc[index, 7]
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(y_label) #word->index
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        return torch.from_numpy(compressed).float() , torch.tensor(numericalized_caption)

class collate_batch:
    def __init__(self, pad_idx, coord_idx, device):
        self.pad_idx = pad_idx
        self.coord_ids = coord_idx

        self.device = device

    def __call__(self, batch):
        self.text_list = []
        self.coordinates_list = []
        for (_coordinates, _text) in batch:

            self.text_list.append(_text)
            self.coordinates_list.append(torch.tensor(_coordinates).float())


        self.text_list = pad_sequence(self.text_list, batch_first=False, padding_value=self.pad_idx)
        self.coordinates_list = pad_sequence(self.coordinates_list, batch_first=False, padding_value=self.coord_ids)
        return self.coordinates_list, self.text_list

def get_loader(
        keyword,
        batch_size = 1
):
    dataset = signvideosDataset(keyword)
    pad_idx = dataset.vocab.stoi['<PAD>']
    coord_ids = 1.0
    loader = DataLoader(dataset=dataset,
        batch_size= batch_size,
        collate_fn = collate_batch(pad_idx = pad_idx, coord_idx=coord_ids, device=device), num_workers=4)
    if keyword == 'train':
        return loader, dataset
    else:
        return loader

def translate_video(model, iterator, device, dataset, max_length=50):
    translations = []
    sentences = []
    model.load_state_dict(torch.load('model_{}.pt'.format('attention'), map_location=device))
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(iterator):
            idx = [i.item() for i in labels]
            sentences.append([dataset.vocab.itos[i] for i in idx][1:])
            inp_data = inputs.to(device)
            target = labels.to(device)
            output = model(inp_data, target, 0)
            translated_sentence = [dataset.vocab.itos[idx.argmax().item()] for idx in output]
            translations.append(translated_sentence[1:])

    return sentences, translations

def translate(model, iterator, device, dataset, max_lenght=50):
    translations = []
    sentences = []
    model.load_state_dict(torch.load('model_{}.pt'.format('attention'), map_location=device))
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(iterator):
        idx = [i.item() for i in labels]
        sentences.append([dataset.vocab.itos[i] for i in idx][1:])
        with torch.no_grad():
            outputs_encoder, hidden, cells = model.encoder(inputs.to(device))
        outputs = [dataset.vocab.stoi["<SOS>"]]
        for _ in range(max_lenght):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)
            with torch.no_grad():
                output, hiddens, cells = model.decoder(
                    previous_word, outputs_encoder, hidden, cells
                )
                best_guess = output.argmax(1).item()
            outputs.append(best_guess)
            if output.argmax(1).item()==dataset.vocab.stoi['<EOS>']:
                break
        translated_sentence = [dataset.vocab.itos[idx] for idx in outputs]
        translations.append(translated_sentence)
    return sentences, translations



