from poseestimation import *
from utils import *
import os

#import keras
import numpy as np
import random
#import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sys
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score,  matthews_corrcoef
#from tensorflow.keras.utils import to_categorical
#from gensim.scripts.glove2word2vec import glove2word2vec
from sentence_transformers import SentenceTransformer

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

#------------------------------------------------------------------------------------------------------------------

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
#model_target = SentenceTransformer('paraphrase-MiniLM-L6-v2')


#------------------------------------------------------------------------------------------------------------------
# Transform dataset target: 
#def process_target():
 #   targ = xdf_data['SENTENCE']
  #  return targ
    #xdf_data['target'] = xdf_data['SENTENCE'].apply(lambda x: model_target.encode(x))

#------------------------------------------------------------------------------------------------------------------
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
    y_target = np.array(xdf_data['SENTENCE'])
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
#------------------------------------------------------------------------------------------------------------------
# Encoder- decoder RNN with attention
#Input: sequence of vectors. Processing done
#Output: sentence. Processing in the flight:
#output_text_processor = tf.keras.layers.TextVectorization(
#    standardize = tf_lower_and_split_punct,
#    max_tokens = MAX_VOCAB_SIZE
#)
#output_text_processor.adapt(targ)

#Model:
embedding_dim = 256
units = 1024

class Encoder(tf.keras.layers.Layer): #tf.keras.Model
    def __int__(self,input_vocab_size, embedding_dim, enc_units): #adds batch size
        super(Encoder, self).__init__()
        #self.batch_sz = batch_sz
        self.enc_units = enc_units
        #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.input_vocab_size = input_vocab_size

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        #Return the sequence and state
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_inizialiaer='glorot_uniform')
    
    def call(self, in_seq, state=None):#hidden
        shape_checker = ShapeChecker()
        shape_checker(in_seq, ('batch','s'))

        vectors = in_seq
        shape_checker(vectors, ('batch','s','embed_dim'))

        output, state = self.gru(vectors, initial_state=state)
        shape_checker(output, ('batch','s','enc_units'))
        shape_checker(state,('batch','enc_units'))

        return output, state


class BahdanauAttention(tf.keras.layers.Layer): #Attention layer
    def __init__(self,units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()
    
    def call(self,query, value, mask):
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))
        shape_checker(mask, ('batch', 's'))


        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch','t','attn_units'))

        w2_key = self.W2(value)
        shape_checker(w2_key,('batch','s','attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs = [w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores = True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                    embedding_dim)
        
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        
        self.attention = BahdanauAttention(self.dec_units)

        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)

        self.fc = tf.keras.layers.Dense(self.output_vocab_size)
        
#------------------------------------------------------------------------------------------------------------------

if __name__ == '"__main__':
    #File name
    for file in os.listdir(PATH):
        if file[-3:] == 'csv':
            FILE_NAME = PATH + os.path.sep + file
    
    xdf_data = pd.read_csv(FILE_NAME)
    xdf_data = shuffle(xdf_data)
    xdf_data.reset_index(inplace=True, drop=True)
    
    train_ds = read_data()
    
    ENCODER = Encoder()



    



        
    
