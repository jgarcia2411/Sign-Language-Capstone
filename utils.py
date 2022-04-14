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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.system("export CUDA_VISIBLE_DEVICES=''")

OR_PATH = '/home/ubuntu/ASSINGMENTS/SignLanguage'

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


# %%____________________________________________Input Target Processing_________________________________________________
# Here we are using 'Bert' model to get tokens ids out of 'Bert' vocabulary size = >5000. Here is where we are having
# idex missmatch error.

# Transformers tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def tokenize_text(text):
    return tokenizer.tokenize(text, padding=True) #padding returns sequence of tokens id with similar length

english = Field(
    init_token='<sos>', tokenize=tokenize_text, lower=True) #eos_token='<eos>'

#__________________________________________________Class to feed in Pytorch data loader_______________________________
class signvideosDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #vectors = []
        video_path = os.path.join(self.root_dir, self.annotations.iloc[index, 3])
        coordinates = torch.from_numpy(process_video(video_path+'.mp4')).float()
        #vectors.append(coordinates)
        y_label = self.annotations.iloc[index, 6]
        y_label = english.preprocess(y_label)  # gives tokens
        y_label.insert(0, english.init_token)  # <sos>
        #y_label.append(english.eos_token)  # <eos>
        y_label = tokenizer.convert_tokens_to_ids(y_label) # inserted line-> get idx from vocabulary
        #y_label = english.preprocess(y_label) # gives tokens
        # <sos>
        # <eos>
        #y_label = [english.vocab.stoi[token] for token in y_label] # maps tokens to vocabulary indices
        y_label = torch.tensor(y_label)
        if self.transform:
            coordinates = self.transform(coordinates)
            #vectors.append(coordinates)

        return coordinates, y_label

# Ignore this for now, we will use this to make predictions after training.
def translate_sentence(model, vectors, english, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    # sequence = vectors #changed  [vector for vector in vectors]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    #sequence.insert(0, english.init_token)
    #sequence.append(english.eos_token)

    # Convert to Tensor

    #sentence_tensor = torch.LongTensor(sequence)#.to(device)#.unsqueeze(1)
    #sentence_tensor = torch.Tensor(vectors).float().to(device) # changed np.array(sequence) to only sequence
    #sentence_tensor = torch.from_numpy(vectors).float().to(device)
    #sentence_tensor = sentence_tensor.float()


    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor(np.array([outputs[-1]]))#.to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, vectors, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, vectors, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
