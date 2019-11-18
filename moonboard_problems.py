import json
import numpy as np
from random import shuffle

"""
Nathan Schneider schnei.nathan@gmail.com
moonboard_problems.py

This program takes the json of the moonboard climbs,
and returns numpy arrays of training and testing data
for a keras network
"""



#A grade map for the european to V grades
grade_map = {
    "6A": 3,
    "6A+" :  3,
    "6B" :  4,
    "6B+" :  4,
    "6C" :  5,
    "6C+" :  5,
    "7A" :  6,
    "7A+" :  7,
    "7B" :  8,
    "7B+" :  8,
    "7C" :  9,
    "7C+" :  10,
    "8A" :  11,
    "8A+" :  12,
    "8B" :  13,
    "8B+" :  14,
    "8C" :  15
}

#Partition 80% train, 20% test
PARTITION_VALUE = .8

NUM_GRADES = 11

#Take a filename, and a number of required repeats (climb popularity measure)
def get_climb_data(filename,repeats_count = 0):

    #load the data
    infile = open(filename)
    climb_dict = json.load(infile)

    #Create the set of holds
    hold_set = set()
    for key in climb_dict:
        for hold in climb_dict[key]["Moves"]:
            hold_set.add(hold["Description"])

    #Count them, and get the empty vector with maps from holds to index
    hold_count = len(hold_set)
    hold_to_index = {}
    hold_list= list(hold_set)
    for i in range(hold_count): hold_to_index[hold_list[i]] = i

    #List of inputs and outputs to be filled
    inputs = []
    outputs = []

    #Iterate trhough all climbs, adding the vectors to input and output
    for climb_id in climb_dict:
        climb = climb_dict[climb_id]

       
        if climb["Repeats"] >= repeats_count:   #Repeats cutoff
            input_vector = [0 for i in range(hold_count)] #Empty holds vector
            for hold in climb["Moves"]: #change the right 0s to 1s
                input_vector[hold_to_index[hold["Description"]]] = 1

            #Categorize the output vector
            output_vector = [0 for i in range(NUM_GRADES)]
            output_vector[grade_map[climb["Grade"]]-4] = 1

            #Add em
            inputs.append(np.array(input_vector))
            outputs.append(np.array(output_vector))

    #Guarantee random climb order
    shuffle(inputs)
    shuffle(outputs)

    #Split them along the partition line
    train_inputs = np.array(inputs[0:int(PARTITION_VALUE*len(inputs))])
    test_inputs = np.array(inputs[int(PARTITION_VALUE*len(inputs)):])

    train_outputs = np.array(outputs[0:int(PARTITION_VALUE*len(inputs))])
    test_outputs = np.array(outputs[int(PARTITION_VALUE*len(inputs)):])

    return (train_inputs,train_outputs), (test_inputs,test_outputs)