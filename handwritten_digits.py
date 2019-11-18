from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
# load data

def reduce_data(inputs,outputs,class_counts):
    new_inputs = []
    new_outputs = []
    counts = [0 for i in range(10)]
    totals = []
    totals.append(len(inputs)/10)
    for i in range(1,10):
        totals.append(totals[-1]*class_counts[i]/class_counts[i-1])
    for i in range(len(inputs)):
        if counts[outputs[i]] < totals[outputs[i]]:
            new_inputs.append(inputs[i])
            new_outputs.append(outputs[i])
            counts[outputs[i]] +=1
        
    return np.array(new_inputs), np.array(new_outputs)


def get_digit_data(class_counts):

    (train_inputs, train_outputs), (test_inputs, test_outputs) = mnist.load_data()

    (train_inputs, train_outputs), (test_inputs, test_outputs)  = \
           reduce_data(train_inputs,train_outputs,class_counts), \
           reduce_data(test_inputs, test_outputs,class_counts)
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = train_inputs.shape[1] * train_inputs.shape[2]
    train_inputs = train_inputs.reshape((train_inputs.shape[0], num_pixels)).astype('float32')
    test_inputs = test_inputs.reshape((test_inputs.shape[0], num_pixels)).astype('float32')
    # normalize inputs from 0-255 to 0-1
    train_inputs = train_inputs / 255
    test_inputs = test_inputs / 255
    # one hot encode outputs

    train_outputs = np_utils.to_categorical(train_outputs)
    test_outputs = np_utils.to_categorical(test_outputs)


    return (train_inputs, train_outputs), (test_inputs,test_outputs)
