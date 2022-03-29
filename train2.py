from turtle import shape
from poseestimation import *
from utils import *
import os
import pandas as pd
#import keras
import numpy as np
import random
#import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef
#from tensorflow.keras.utils import to_categorical
#from gensim.scripts.glove2word2vec import glove2word2vec
from sentence_transformers import SentenceTransformer
from tensorflow.keras import Input
from tensorflow.keras.layers import Lambda


#Additionals
#from tensorflow.keras.optimizers import Adam
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#from tensorflow.keras.models import Sequential, LSTM, RNN
#from tensorflow.keras import layers
import typing 
from typing import Any, Tuple 
import tensorflow_text as tf_text
from tensorflow.keras import backend as K


## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

## folder "Data" images
## folder "excel" excel file , whatever is there is the file
## get the classes from the excel file
## folder "Documents" readme file

OR_PATH = os.getcwd()
os.chdir("..") # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

n_epoch = 2000
BATCH_SIZE = 64
MODEL_NAME = 'Transformers with attention'
MAX_VOCAB_SIZE = 5000
FEATURES = 1662
N_TIMESTEPS_IN = #number of frames
latentSpaceDimension = 256 #Arbitrary 
model_target = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Text processing
def tf_lower_and_split_punct(text):
    # Split accecented characters.
  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  # Keep space, a to z, and select punctuation.
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  # Add spaces around punctuation.
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
  # Strip whitespace.
  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text

# Process target:
def process_target():
    xdf_data['target'] = xdf_data['SENTENCE'].apply(lambda x: tf_lower_and_split_punct(x))
    xdf_data['target'] = xdf_data['target'].apply(lambda x: model_target.encode(x))
    return xdf_data

    
# Processing path and getting features (vector/frame) and label
def process_path(feature,target):
    vectors = []
    label = target
    file_path = feature+".mp4" #This variable will pass read video cv2
    vector = process_video(file_path)
    vectors.append(vector)
    return vectors, label
#------------------------------------------------------------------------------------------------------------------
# Processing target to transform as numpy array
def get_target():
    xdf_data = process_target()
    y_target = np.array(xdf_data['target'])
    return y_target

#------------------------------------------------------------------------------------------------------------------
# Read data
def read_data():
    ds_inputs = np.array(DATA_DIR+xdf_data['SENTENCE_NAME'])
    ds_targets = get_target

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) 
    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
    return final_ds

#------------------------------------------------------------------------------------------------------------------
# Save model
def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt'.format(MODEL_NAME), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))




#_________________________________________________________________________
def create_hard_coded_decoder_input_model(batch_size):
    #Encoder:
    encoder_inputs = Input(shape=(N_TIMESTEPS_IN, 1662), name='encoder_inputs') # timestemps in is number of frames, 1662 n_features =keypoints
    encoder_lstm = LSTM(latentSpaceDimension, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # initial context vector is the states of the encoder
    states = [state_h, state_c]

    # Set up the decoder layers
    decoder_inputs = Input(shape=(1, 1662)) 
    decoder_lstm = LSTM(latentSpaceDimension, return_sequences=True, 
                      return_state=True, name='decoder_lstm')
    decoder_dense = Dense(MAX_VOCAB_SIZE, activation='softmax',  name='decoder_dense') #output_shape change to dimension of output embeding

    all_outputs = []
    decoder_input_data = np.zeros((batch_size, 1, 1662))
    decoder_input_data[:, 0, 0] = 1 #

    inputs = decoder_input_data
    # decoder will only process one timestep at a time.
    for _ in range(N_TIMESTEPS_IN):
        # Run the decoder on one timestep
        outputs, state_h, state_c = decoder_lstm(inputs,
                                                initial_state=states)
        outputs = decoder_dense(outputs)
        # Store the current prediction (we will concatenate all predictions later)
        all_outputs.append(outputs)
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        inputs = outputs
        states = [state_h, state_c]
    # Concatenate all predictions such as [batch_size, timesteps, features]
    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)



#__________________________________________________________________________________________
xdf_data = pd.read_csv('how2sign_realigned_train.csv')
videos = ['_-adcxjm1R4_0-8-rgb_front', '_-adcxjm1R4_1-8-rgb_front','_0-JkwZ9o4Q_5-5-rgb_front',
             '_0-JkwZ9o4Q_6-5-rgb_front', '_0-JkwZ9o4Q_7-5-rgb_front', '_0-JkwZ9o4Q_8-5-rgb_front',
             '_0-JkwZ9o4Q_9-5-rgb_front', '_0-JkwZ9o4Q_10-5-rgb_front', '_0-JkwZ9o4Q_11-5-rgb_front',
             '_0-JkwZ9o4Q_12-5-rgb_front', '_0fO5ETSwyg_0-5-rgb_front', '_0fO5ETSwyg_1-5-rgb_front',
             '_0fO5ETSwyg_2-5-rgb_front', '_0fO5ETSwyg_3-5-rgb_front', '_2u0MkRqpjA_0-5-rgb_front',
             '_2u0MkRqpjA_1-5-rgb_front']

xdf_data = xdf_data[xdf_data['SENTENCE_NAME'].isin(videos)]

training_ds = read_data()