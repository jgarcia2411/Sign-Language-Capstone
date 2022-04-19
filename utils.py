import cv2
import numpy as np
import mediapipe as mp
import multiprocessing
from multiprocessing import Pool
from torchtext.legacy.data import Field
import spacy
from torchtext.vocab import Vectors
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from torchtext.data.metrics import bleu_score
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

#text processing
import re
import spacy
import string
spacy_model = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
from nltk.stem import *
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# _________________________un-comment this to use GPU_________________________
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.system("export CUDA_VISIBLE_DEVICES=''")

#
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'
DATA_DIR = '/home/ubuntu/ASL'
spacy_en = spacy.load('en_core_web_sm')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%____________________________________________Video Processing_______________________________________________________
# This function reads a video frame by frame and returns a sequence of (#frames, dime=1662) vectors
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB- required in Mediapipe
    image.flags.writeable = False  # Image is no longer writeable, saves memory
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)  # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    vectors = np.concatenate([pose, face, lh, rh])
    return vectors


def process_video(video):  # we can try to implement tqdm process bar here
    cap = cv2.VideoCapture(video)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = []
        for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Read feed
            ret, frame = cap.read()
            if ret == True:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                # print(results)

                # Draw landmarks
                # draw_styled_landmarks(image, results)

                # Export Keypoints
                vectors = extract_keypoints(results)
                sequence.append(vectors)

    cap.release()
    cv2.destroyAllWindows()
    return np.vstack(sequence)


punctuations = """!()-[]{};:'"\,<>./?@#$%^&*_~"""

################################
# TEXT PREPROCESSING FUNCTIONS #
################################
def _remove_punctuation(text, step):
    if step == 'initial':
        return [
            token for token in text if re.sub(r'[\W_]+', ' ', token.text)
            not in string.punctuation
            and re.sub(r'([\W_])+', ' ', token.text) != ' '
            and re.sub(r'([\W_])+', ' ', token.text) != ''
        ]
    elif step == 'last':
        return [re.sub(r'[\W_]+', ' ', token) for token in text]

def _remove_stop_words(text):
    return [token for token in text if not token.is_stop]

def _lemmatize(text):
    return [token.lemma_ for token in text]

def preprocess_text(text, is_search_space=True):
    if is_search_space:
        # Remove the upper header part of the text.
        # We only need to do this for the search
        # space, not for the query string.
        step_1 = ' '.join(text.split('\n\n')[1:])
    else:
        step_1 = text

    # Lowercase text and remove extra spaces.
    step_2_3 = ' '.join(
        [word.lower() for word in str(step_1).split()]
    )

    # Tokenize text with spaCy.
    step_4 = spacy_model(step_2_3)

    # Remove punctuation.
    step_5 = _remove_punctuation(step_4, step = 'initial')

    # Remove stop words.
    step_6 = _remove_stop_words(step_5)

    # Lemmatize text.
    step_7 = _lemmatize(step_6)

    # Remove punctuation again.
    step_8 = _remove_punctuation(step_7, step = 'last')

    # Remake sentence with new cleaned up tokens.
    return ' '.join(step_8)


# %%____________________________________________Input Target Processing_________________________________________________

class Vocabulary:
    def __init__(self, freq_threshold):
        """freq_threshold: frequency of words to build vocabulary
        itos: index to string
        stoi: string to index"""

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod

    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4  # 3:<unk>
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]


# __________________________________________________Class to feed in Pytorch data loader_______________________________
class signvideosDataset(Dataset):
    def __init__(self, csv_file, root_dir, keyword, transform=None):
        # read entire annotations files to build vocabulary:
        self.annotations = pd.read_csv(csv_file)
        self.annotations['no_punc'] = self.annotations['SENTENCE'].apply(lambda x: cleaner(x, punctuations))
        self.vocab = Vocabulary(1)
        self.vocab.build_vocabulary(self.annotations['no_punc'].tolist())

        # Tet with 100 videos for training, validating and testing
        if keyword == "train":
            self.annotations = self.annotations.head(500)
        elif keyword == "test":
            self.annotations = self.annotations.sample(20)
        elif keyword == 'val':
            self.annotations = self.annotations.sample(20)
        else:
            print('PLEASE SPECIFY KEYWORD')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        # Read path to process with processing function:
        video_path = os.path.join(self.root_dir, self.annotations.iloc[index, 3])
        # coordinates = torch.from_numpy(process_video(video_path+'.mp4')).float()
        coordinates = process_video(video_path + '.mp4')
        if self.transform:
            coordinates = self.transform(coordinates)

        # Reads sentence to process with tokenizer
        y_label = self.annotations.iloc[index, 6]
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(y_label)  # word->index
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return coordinates, torch.tensor(numericalized_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        coordinates = [item[0] for item in batch]
        # coordinates=np.stack(coordinates, axis=1).astype(np.float)
        coordinates = torch.from_numpy(np.array(coordinates)).float()
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return coordinates, targets


class collate_batch:
    def __init__(self, pad_idx, frames_idx, device):
        self.pad_idx = pad_idx
        self.frames_ids = frames_idx

        self.device = device

    def __call__(self, batch):
        self.text_list = []
        self.coordinates_list = []
        for (_coordinates, _text) in batch:
            self.text_list.append(_text)
            self.coordinates_list.append(torch.tensor(_coordinates).float())

        self.text_list = pad_sequence(self.text_list, batch_first=False, padding_value=self.pad_idx)
        self.coordinates_list = pad_sequence(self.coordinates_list, batch_first=False, padding_value=self.frames_ids)
        return self.coordinates_list, self.text_list


def get_loader(
        csv_file,
        root_dir,
        keyword,
        batch_size=2,
        transform=None
):
    dataset = signvideosDataset(csv_file, root_dir, keyword, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    frames_ids = 1.0
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        collate_fn=collate_batch(pad_idx=pad_idx, frames_idx=frames_ids, device=device))
    return loader, dataset


# batch_size = 2
# loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", keyword='train', batch_size=batch_size)

# for batch_idx, (inputs, labels) in enumerate(loader):
#    print(f'Batch number {batch_idx} \n Inputs Shape {inputs.shape} \n Labels Shape {labels.shape}')


# def get_loader(
#        csv_file,
#        root_dir,
#        keyword,
#        batch_size = 32,
#        transform=None
# ):
#    dataset = signvideosDataset(csv_file, root_dir, keyword, transform=transform)
#    pad_idx = dataset.vocab.stoi['<PAD>']
#    loader = DataLoader(dataset=dataset,
#        batch_size= batch_size,
#        collate_fn = MyCollate(pad_idx = pad_idx))
#    return loader, dataset


# ___________Uncomment this to test data loader___________________________________

# loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", batch_size=2)

# for idx, (inputs,labels) in enumerate(loader):
#    print(inputs.shape)

# _______________________________________________________________________________________________________________________

def translate_video(model, iterator, device, dataset, max_length=50):
    translations = []
    sentences = []
    model.load_state_dict(torch.load('model_{}.pt'.format('SIGN2TEXT'), map_location=device))
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
    # inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[0], inputs.shape[-1]))
    # for sentence in labels:
    #        idx = [i.item() for i in sentence]
    # sentences.append([dataset.vocab.itos[i] for i in idx][1:])
    # inp_data = inputs.to(device)
    # target = labels.to(device)

    # Pass video through encoder
    # hidden, cell = model.encoder(inp_data)
    # Pass video through decoder
    # outputs = [dataset.vocab.stoi["<SOS>"]]

    # for _ in range(max_length):
    #    previous_word = torch.tensor([outputs[-1]]).to(device)
    #    previous_word = torch.tensor(np.array([outputs[-1]])).to(device)

    #    with torch.no_grad():
    #        output, hidden, cell = model.decoder(previous_word, hidden, cell)
    #        best_guess = output.argmax(1).item()

    #    outputs.append(best_guess)

    # Model predicts it's the end of the sentence
    #    if output.argmax(1).item() == dataset.vocab.stoi["<EOS>"]:
    #        break

    # translated_sentence = [dataset.vocab.itos[idx.argmax().item()] for idx in output]

    # translations.append(translated_sentence[1:])


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
