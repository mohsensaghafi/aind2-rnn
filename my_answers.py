import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import re


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for t in range(len(series) - window_size):
        X.append(series[t:t+window_size])
        y.append(series[t+window_size])
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    # layer 1: LSTM module with 5 hidden units
    model.add(LSTM(5, input_shape=(window_size, 1)))
    
    # layer 2: fully connected module (one unit)
    model.add(Dense(1))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    # regex that match any character that is not acsii lowercase and not punctuation character.
    regex = '[^'+''.join(punctuation)+'a-z]'
    clean_text = re.sub(regex, ' ', text)
    return clean_text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for index in range(0, len(text) - window_size, step_size):
        inputs.append(text[index : index + window_size])
        outputs.append(text[index + window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    
    model = Sequential()
    # layer 1: LSTM module with 200 hidden units
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    
    # layer 2: linear module, fully connected, with len(chars) hidden units
    model.add(Dense(num_chars))

    # layer 3: softmax activation
    model.add(Activation("softmax"))
    
    return model