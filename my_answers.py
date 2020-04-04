import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    
     #loop over the input series
    for point in range(len(series) - window_size):
        #extract the input points by window size
        X.append(series[point:point+window_size])
        #extract the output point by looking at the next point
        y.append(series[point+window_size])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #Model type
    model = Sequential()
    #LSTM module 5 nodes
    model.add(LSTM(5,  input_shape = (window_size,1)))
    #dense layer for output 1 node
    model.add(Dense(1))
    
    return model



### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    alphabet = list(string.ascii_lowercase)
    valid_chars = set(punctuation + alphabet + [' '])

    all_chars = set(text)
    chars_to_remove = all_chars - valid_chars

    for char_to_remove in chars_to_remove:
        text = text.replace(char_to_remove,' ')

    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
     # containers for input/output pairs  
    inputs = []
    outputs = []
    
    i= 0
    txt_length= len(text)
    while i + window_size < txt_length: # while window upper bound is < entire string
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size
        
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    from keras.layers import Dense, Activation, LSTM
    model=Sequential()
    #lstm hidden layer with 200 nodes.
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    #fully connected dense layer 
    model.add(Dense(num_chars))
    model.add(Activation('softmax')) # softmax for multi-class
    return model
   
