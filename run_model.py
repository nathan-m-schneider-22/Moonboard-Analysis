import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

#This class weights is the counts of each category of climbs, used to both calculate the 

def train_model(train_inputs, train_outputs,weighted = False,class_counts = None):

    model = Sequential()
    model.add(Dense(20, input_dim=train_inputs.shape[1], activation="relu"))
    model.add(Dense(train_outputs.shape[1], activation="softmax"))
    model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

    if weighted:

        class_weights = {}
        for key in class_counts:
            class_weights[key] = 1/class_counts[key]
        model.fit(train_inputs, train_outputs, epochs=10, batch_size=10, verbose=1,class_weight = class_weights)

    else:
        model.fit(train_inputs, train_outputs, epochs=10, batch_size=10, verbose=1)

    return model

def evaluate_model(model,test_inputs,test_outputs,title,x_axis_offset = 0,block = False):
    scores = model.evaluate(test_inputs, test_outputs, verbose=0)

    print("Categorical Accuracy: %.2f%%" % (scores[1]*100))
    off_by_one_correct = 0
    guesses = []
    real = []
    correct_guess = []
    wrong_guess = []

    y_array = model.predict(test_inputs)
    for index in range(len(test_inputs)):
        y = y_array[index]
        t = test_outputs[index]
        y_val = np.argmax(y)
        t_val = np.argmax(t)
        guesses.append(y_val + x_axis_offset)
        real.append(t_val+ x_axis_offset)

        if t_val == y_val: correct_guess.append(y_val+ x_axis_offset)
        else: wrong_guess.append(y_val+ x_axis_offset)

        if abs(t_val - y_val)<=1: off_by_one_correct +=1

    print("Off by one Accuracy: %.2f" %(off_by_one_correct/len(test_inputs)*100))

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
    # plt.show(block = True)
    plt.clf()