import json
import numpy as np
import matplotlib.pyplot as plt
grade_map = {}
grade_map["6A"] = 3
grade_map["6A+"] = 3
grade_map["6B"] = 4
grade_map["6B+"] = 4
grade_map["6C"] = 5
grade_map["6C+"] = 5
grade_map["7A"] = 6
grade_map["7A+"] = 7
grade_map["7B"] = 8
grade_map["7B+"] = 8
grade_map["7C"] = 9
grade_map["7C+"] = 10
grade_map["8A"] = 11
grade_map["8A+"] = 12
grade_map["8B"] = 13
grade_map["8B+"] = 14
grade_map["8C"] = 15

class_weights = {0:1/6390,
    1:1/4536,
    2:1/2469,
    3:1/1729,
    4:1/1834,
    5:1/790,
    6:1/272,
    7:1/91,
    8:1/20,
    9:1/8,
    10:0.1/8
}
PARTITION_VALUE = .85
NUM_GRADES = 13
            
f = open("data/moonboard_problems_setup_2016.json")
js = json.load(f)

print(len(js))
problem_count = 0
hold_set = set()
grades = []
bad_prob = set()
for key in js:
    problem_count +=1
    grades.append(grade_map[js[key]["Grade"]])
    for hold in js[key]["Moves"]:
        hold_set.add(hold["Description"])
    if js[key]["Repeats"]<1:
        bad_prob.add(key)
        problem_count -=1

for key in bad_prob: del js[key]
hold_count = len(hold_set)
hold_to_index = {}
hold_list= list(hold_set)
for i in range(hold_count): hold_to_index[hold_list[i]] = i
climb_id = '307479'

inputs = np.empty([problem_count,hold_count])
outputs = np.empty([problem_count,NUM_GRADES])
index = 0

for climb_id in js:
    climb = js[climb_id]
    input_vector = [0 for i in range(hold_count)]
    for hold in climb["Moves"]:
        input_vector[hold_to_index[hold["Description"]]] = 1

    output_vector = [0 for i in range(NUM_GRADES)]
    output_vector[grade_map[climb["Grade"]]-4] = 1

    inputs[index] = input_vector
    outputs[index] = output_vector
    index +=1

def partition_vals(inputs,outputs):
    counts = [0 for i in range(NUM_GRADES)]        

    test_inputs= []
    test_outputs=[]
    train_inputs=[]
    train_outputs=[]
    np.random.shuffle(inputs)
    np.random.shuffle(outputs)
    for index in range(len(inputs)):
        grade_index = np.where(outputs[index]==1)[0][0]
        if counts[grade_index]< 2500:
            counts[grade_index] +=1
            if index%5 !=0:
                train_inputs.append(inputs[index])
                train_outputs.append(outputs[index])
            else:
                test_inputs.append(inputs[index])
                test_outputs.append(outputs[index])

    print(counts)
    # grade = 5
    # for index in range(len(inputs)):
    #     if outputs[index][grade-4]==1:

    #     else:
    #         train_inputs.append(inputs[index])
    #         train_outputs.append(outputs[index])

    return np.array(test_inputs),np.array(test_outputs),np.array(train_inputs),np.array(train_outputs)

test_inputs,test_outputs,train_inputs,train_outputs = partition_vals(inputs,outputs)

print(test_inputs,test_outputs,train_inputs,train_outputs)

train_inputs = inputs[0:int(PARTITION_VALUE*problem_count)]
test_inputs = inputs[int(PARTITION_VALUE*problem_count):]

train_outputs = outputs[0:int(PARTITION_VALUE*problem_count)]
test_outputs = outputs[int(PARTITION_VALUE*problem_count):]

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

logfile = open("outputs.log","w")


print(NUM_GRADES,problem_count)

print(train_inputs[0])
print(train_outputs[0])
model = Sequential()
model.add(Dense(20, input_dim=train_inputs.shape[1], activation="relu"))
model.add(Dense(NUM_GRADES, activation="softmax"))
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
# model.fit(train_inputs, train_outputs, epochs=10, batch_size=10, verbose=1)
model.fit(train_inputs, train_outputs,class_weight=class_weights, epochs=10, batch_size=10, verbose=1)


print(model.evaluate(test_inputs, test_outputs, batch_size=128)[1])
# print(model.evaluate(test_inputs, test_outputs, batch_size=128)[2])



def off_one_test(test_inputs,test_out,model):
    guesses = []
    real = []
    correct_guess = []
    wrong_guess = []

    correct = 0
    off_by = 0
    y_array = model.predict(test_inputs)
    for index in range(len(test_inputs)):
        y = y_array[index]
        t = outputs[index]
        y_val = np.argmax(y)
        t_val = np.argmax(t)
        guesses.append(y_val+4)
        real.append(t_val+4)
        # print(y)
        # print(t)
        # print(y_val,t_val)
        if abs(t_val - y_val) <=off_by: correct+=1

        if t_val == y_val: correct_guess.append(y_val+4)
        else: wrong_guess.append(y_val+4)


    print("Off by ",off_by)
    print(correct/index)
    plt.hist([real,guesses,wrong_guess,correct_guess],\
        bins=[i+4 for i in range(NUM_GRADES)],\
        color= ["Orange","darkblue","Red","Green"],\
        label=("Actual","Network Guess","Incorrect Guess","Correct Guess"))
    plt.legend()
    plt.xticks([i+4 for i in range(NUM_GRADES)])
    plt.xlabel("Grade (Difficulty)")
    plt.ylabel("Number of Climbs")
    plt.title("Network Guesses versus Actual Values")
    for i in range(20):
        print(i,real.count(i))
        print(i,guesses.count(i))
    plt.savefig("guesses.png")
    plt.show()

        
off_one_test(test_inputs,test_outputs,model)
off_one_test(train_inputs,train_outputs,model)