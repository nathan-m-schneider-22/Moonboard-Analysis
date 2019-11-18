from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
"""
Nathan Schneider schnei.nathan@gmail.com
handwritten_digits.py

This program loads and returns numpy arrays of 
testing and training data for the handwritten digits
recognition problem. It reduces the data from an 
even distribution to instead match the distribution
between grades of the climbing data, to replicate the
imablanced state of the climbing classification

"""

#Takes the data, as well as the grade counts for climbs, then 
#returns data that is reduced so it matches the distribution
def reduce_data(inputs,outputs,class_counts):
    new_inputs = []
    new_outputs = []
    
    #the number of inputs for each class so far
    counts = [0 for i in range(10)]

    #Keep an array of total inputs allowed for each class
    totals = []
    totals.append(len(inputs)/10) 
    for i in range(1,10): #The totals in the distribution after the first class
        #are always less, so we can express the totals as proportions compared to the last
        totals.append(totals[-1]*class_counts[i]/class_counts[i-1])

    #Iterate through the inputs, and while there is less than total for that class, add em to the inputs/outputs
    for i in range(len(inputs)):
        if counts[outputs[i]] < totals[outputs[i]]:
            new_inputs.append(inputs[i])
            new_outputs.append(outputs[i])
            counts[outputs[i]] +=1
        
    #Return the numpy arrays
    return np.array(new_inputs), np.array(new_outputs)


def get_digit_data(class_counts):

    #Load the dataset from the mnist keras dataset
    (train_inputs, train_outputs), (test_inputs, test_outputs) = mnist.load_data()


    #Reduce the training and testing sets to mirror the climb distribution
    (train_inputs, train_outputs), (test_inputs, test_outputs)  = \
           reduce_data(train_inputs,train_outputs,class_counts), \
           reduce_data(test_inputs, test_outputs,class_counts)
           
    #Find the total num of pixels
    num_pixels = train_inputs.shape[1] * train_inputs.shape[2]
    #Vectorize the inputs 
    train_inputs = train_inputs.reshape((train_inputs.shape[0], num_pixels)).astype('float32')
    test_inputs = test_inputs.reshape((test_inputs.shape[0], num_pixels)).astype('float32')

    #Normalize inputs from 0-255 to 0-1
    train_inputs = train_inputs / 255
    test_inputs = test_inputs / 255

    #Make the output vals into 0,1,0... vectors
    train_outputs = np_utils.to_categorical(train_outputs)
    test_outputs = np_utils.to_categorical(test_outputs)

    return (train_inputs, train_outputs), (test_inputs,test_outputs)
