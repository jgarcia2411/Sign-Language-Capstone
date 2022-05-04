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
from torch.nn.utils.rnn import  pad_sequence
from torch.utils.data import DataLoader

# _________________________un-comment this to use GPU_________________________
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.system("export CUDA_VISIBLE_DEVICES=''")

#
OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'
DATA_DIR = '/home/ubuntu/ASL'
glove_path = '/home/ubuntu/ASSINGMENTS'
spacy_en = spacy.load('en_core_web_sm')
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
training_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/train_videos')]
test_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/test_videos')]
val_videos = [i[:-4] for i in os.listdir('/home/ubuntu/ASL/val_videos')]

# %%____________________________________________Video Processing_______________________________________________________
# This function reads a video frame by frame and returns a sequence of (#frames, dime=1662) vectors
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB- required in Mediapipe
    image.flags.writeable = False                  # Image is no longer writeable, saves memory
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    vectors = np.concatenate([pose, face, lh, rh]) 
    return vectors

def process_video(video): #we can try to implement tqdm process bar here
    cap = cv2.VideoCapture(video)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = []
        for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Read feed
            ret, frame = cap.read()
            if ret == True:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                #print(results)
                
                # Draw landmarks
                #draw_styled_landmarks(image, results)

                #Export Keypoints
                vectors = extract_keypoints(results)
                sequence.append(vectors)

    cap.release()
    cv2.destroyAllWindows()
    return np.vstack(sequence)

#vectors = process_video('/home/ubuntu/ASL/train_videos/03SgrpiJkSw_0-8-rgb_front.mp4')

punctuations = """!()-[]{};:\,<>./@#$%^&*_~...0123456789âÂ+-¦Ã©´¾½\x80\x99\x9c\x9d"""
def cleaner(sentence, punctuations):
    no_punc = ''
    for char in sentence:
        if char not in punctuations:
            no_punc = no_punc + char

    return no_punc


# %%____________________________________________Input Target Processing_________________________________________________



# Own vocabulary
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
        self.dataframe['clean_text'] = self.dataframe['SENTENCE'].apply(lambda x: cleaner(x,punctuations))
        sentences_list = self.dataframe['clean_text'].tolist()
        tokens_list = [self.tokenizer_eng(i) for i in sentences_list]
        idx = 4  # 3:<unk>
        for sentence in tokens_list:
            for token in sentence:
                if token not in self.stoi:
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

#__________________________________________________Class to feed in Pytorch data loader_______________________________
class signvideosDataset(Dataset):
    def __init__(self, csv_file, root_dir, keyword, transform=None): #vocabulary_list
        # read entire annotations files to build vocabulary:
        self.annotations = pd.read_csv(csv_file)
        self.annotations['clean text'] = self.annotations['SENTENCE'].apply(lambda x: cleaner(x, punctuations))
        self.vocab = Vocabulary()
        self.vocab.build_vocabulary()
        self.keyword = keyword

        # Tet with 100 videos for training, validating and testing
        if self.keyword == "train":
            self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(training_videos)]
            self.annotations = self.annotations.sample(5000, random_state=123)
        elif self.keyword == "test":
            self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(test_videos)]
            self.annotations = self.annotations.sample(10, random_state=123)
        elif self.keyword == 'val':
            self.annotations = self.annotations[self.annotations['SENTENCE_NAME'].isin(val_videos)]
            self.annotations = self.annotations.sample(10, random_state=123)
        else:
            print('PLEASE SPECIFY KEYWORD')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        #Read path to process with processing function:
        video_path = os.path.join(self.root_dir, self.annotations.iloc[index, 3])
        #coordinates = torch.from_numpy(process_video(video_path+'.mp4')).float()
        coordinates = process_video(video_path+'.mp4')
        if self.transform:
            coordinates = self.transform(coordinates)

        # Reads sentence to process with tokenizer
        y_label = self.annotations.iloc[index, 7]
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(y_label) #word->index
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        return coordinates , torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        coordinates = [item[0] for item in batch]
        #coordinates=np.stack(coordinates, axis=1).astype(np.float)
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
        batch_size = 1,
        transform=None
):
    dataset = signvideosDataset(csv_file, root_dir, keyword,transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']
    frames_ids = 1.0
    loader = DataLoader(dataset=dataset,
        batch_size= batch_size,
        collate_fn = collate_batch(pad_idx = pad_idx, frames_idx=frames_ids, device=device), num_workers=8)
    if keyword == 'train':
        return loader, dataset
    else:
        return loader


#batch_size = 2
#loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", keyword='train', batch_size=batch_size)

#for batch_idx, (inputs, labels) in enumerate(loader):
#    print(f'Batch number {batch_idx} \n Inputs Shape {inputs.shape} \n Labels Shape {labels.shape}')




#def get_loader(
#        csv_file,
#        root_dir,
#        keyword,
#        batch_size = 32,
#        transform=None
#):
#    dataset = signvideosDataset(csv_file, root_dir, keyword, transform=transform)
#    pad_idx = dataset.vocab.stoi['<PAD>']
#    loader = DataLoader(dataset=dataset,
#        batch_size= batch_size,
#        collate_fn = MyCollate(pad_idx = pad_idx))
#    return loader, dataset


#___________Uncomment this to test data loader___________________________________

#loader, dataset = get_loader(OR_PATH+'/how2sign_realigned_train 2.csv', root_dir=DATA_DIR+"/train_videos/", batch_size=2)

#for idx, (inputs,labels) in enumerate(loader):
#    print(inputs.shape)

#_______________________________________________________________________________________________________________________

def translate_video(model, iterator, device, dataset, max_length=50):
    translations = []
    sentences = []
    model.load_state_dict(torch.load('model_{}.pt'.format('SIGN2TEXT_500'), map_location=device))
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
            #inputs = torch.reshape(inputs, (inputs.shape[1], inputs.shape[0], inputs.shape[-1]))
            #for sentence in labels:
            #        idx = [i.item() for i in sentence]
            #sentences.append([dataset.vocab.itos[i] for i in idx][1:])
            #inp_data = inputs.to(device)
            #target = labels.to(device)

            # Pass video through encoder
            #hidden, cell = model.encoder(inp_data)
            # Pass video through decoder
            #outputs = [dataset.vocab.stoi["<SOS>"]]

            #for _ in range(max_length):
            #    previous_word = torch.tensor([outputs[-1]]).to(device)
            #    previous_word = torch.tensor(np.array([outputs[-1]])).to(device)

            #    with torch.no_grad():
            #        output, hidden, cell = model.decoder(previous_word, hidden, cell)
            #        best_guess = output.argmax(1).item()

            #    outputs.append(best_guess)

            # Model predicts it's the end of the sentence
            #    if output.argmax(1).item() == dataset.vocab.stoi["<EOS>"]:
            #        break

            #translated_sentence = [dataset.vocab.itos[idx.argmax().item()] for idx in output]

            #translations.append(translated_sentence[1:])




def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

