import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
"""
Nathan Schneider schnei.nathan@gmail.com
run_model.py
This program takes input and output training numpy arrays
and trains a keras MLP, which can be trained with class
weights inversely proportional to class counts.
It also evaluates this network, providing accuracy,
off by one accuracy, and saving a histogram that
accurately reflects the accuracy of the network classification
"""

#Trains the model, either weighted by the counts of the classes, or directly
def train_model(train_inputs, train_outputs,class_counts = None):

    #Basic sequential Keras MLP
    model = Sequential()
    model.add(Dense(20, input_dim=train_inputs.shape[1], activation="relu"))
    model.add(Dense(train_outputs.shape[1], activation="softmax"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    #Turn the counts into inverse weights, and train
    if class_counts != None:
        class_weights = {}
        for key in class_counts:
            class_weights[key] = 1/class_counts[key]
        model.fit(train_inputs, train_outputs, epochs=10, batch_size=10, verbose=1,class_weight = class_weights)

    else: #Otherwise train it
        model.fit(train_inputs, train_outputs, epochs=10, batch_size=10, verbose=1)

    #Return the trained model
    return model

def evaluate_model(model,test_inputs,test_outputs,title,x_axis_offset = 0,block = False):
    #Keras categorical accuracy
    scores = model.evaluate(test_inputs, test_outputs, verbose=0)
    print("Categorical Accuracy: %.2f%%" % (scores[1]*100))

    #Our own accuracy, for if the networkd can guess close to correct
    off_by_one_correct = 0

    #Build lists of y and t values for tracking accuracy for each category
    guesses = []
    real = []
    correct_guess = []
    wrong_guess = []

    #Predict the y values for our test set
    y_array = model.predict(test_inputs)

    #iterate through each y and t
    for index in range(len(test_inputs)):
        #Find the value
        y_val = np.argmax(y_array[index])
        t_val = np.argmax(test_outputs[index])

        #Add the networkd guess (y) to the guesses
        guesses.append(y_val + x_axis_offset)
        #Add the t value to the real list
        real.append(t_val+ x_axis_offset)

        #Add the guess to the appropriate histogram bin
        if t_val == y_val: correct_guess.append(y_val+ x_axis_offset)
        else: wrong_guess.append(y_val+ x_axis_offset)

        #Track if the network was only off by a grade
        if abs(t_val - y_val)<=1: off_by_one_correct +=1

    print("Off by one Accuracy: %.2f" %(off_by_one_correct/len(test_inputs)*100))


    #Create a histogram to display the categorical accuracy of the network
    plt.hist([real,guesses,correct_guess,wrong_guess],\
        bins=[i+ x_axis_offset for i in range(len(test_outputs[0]))],\
        color= ["Orange","darkblue","Green","Red"],\
        label=("Actual","Network Guess","Correct Guess","Incorrect Guess"))

    plt.legend()
    plt.xticks([i+x_axis_offset for i in range(len(test_outputs[0]))])
    plt.xlabel("Class Category")
    plt.ylabel("Counts")
    plt.title(title)
    plt.savefig(title.replace(" ","_")+".png")
    plt.clf()