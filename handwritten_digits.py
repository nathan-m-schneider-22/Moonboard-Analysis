from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
# load data

class_weights = {0:6390,
    1:4536,
    2:2469,
    3:1729,
    4:1834,
    5:790,
    6:272,
    7:91,
    8:20,
    9:8,
    10:0.8
}
def reduce_data(inputs,outputs):
    new_inputs = []
    new_outputs = []
    counts = [0 for i in range(10)]
    totals = []
    totals.append(len(inputs)/10)
    for i in range(1,10):
        totals.append(totals[-1]*class_weights[i]/class_weights[i-1])

    for i in range(len(inputs)):
        if counts[outputs[i]] < totals[outputs[i]]:
            new_inputs.append(inputs[i])
            new_outputs.append(outputs[i])
            counts[outputs[i]] +=1
        
    return np.array(new_inputs), np.array(new_outputs)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs

(X_train, y_train), (X_test, y_test) = reduce_data(X_train, y_train), reduce_data(X_test,y_test)


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# define baseline model
model = Sequential()
model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# build the model
# Fit the model
model.fit(X_train, y_train, epochs=20, batch_size=200, verbose=1,class_weight=class_weights)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))




guesses = []
real = []
correct_guess = []
wrong_guess = []

correct = 0
off_by = 0
outputs = y_test
y_array = model.predict(X_test)
for index in range(len(X_test)):
    y = y_array[index]
    t = outputs[index]
    y_val = np.argmax(y)
    t_val = np.argmax(t)
    guesses.append(y_val)
    real.append(t_val)
    if abs(t_val - y_val) <=off_by: correct+=1

    if t_val == y_val: correct_guess.append(y_val)
    else: wrong_guess.append(y_val)


plt.hist([real,guesses,wrong_guess,correct_guess],\
    bins=[i for i in range(10+1)],\
    color= ["Orange","darkblue","Red","Green"],\
    label=("Actual","Network Guess","Incorrect Guess","Correct Guess"))
plt.legend()
plt.xlabel("Digit")
plt.ylabel("Number of Values")
plt.title("Network Guesses versus Actual Values")
plt.savefig("guesses.png")
plt.show()
