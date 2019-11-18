import json
import numpy as np


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

PARTITION_VALUE = .8
NUM_GRADES = 13
# FILENAME = "data/moonboard_problems_setup_2016.json"
def get_climb_data(filename,repeats_count = 0):

    infile = open(filename)
    climb_dict = json.load(infile)

    hold_set = set()
    bad_prob = set()
    for key in climb_dict:
        for hold in climb_dict[key]["Moves"]:
            hold_set.add(hold["Description"])
        # if js[key]["Repeats"]<1:
        #     bad_prob.add(key)
        #     problem_count -=1

    hold_count = len(hold_set)
    hold_to_index = {}
    hold_list= list(hold_set)
    for i in range(hold_count): hold_to_index[hold_list[i]] = i


    inputs = []
    outputs = []


    for climb_id in climb_dict:
        climb = climb_dict[climb_id]

        if climb["Repeats"] >= repeats_count:
            input_vector = [0 for i in range(hold_count)]
            for hold in climb["Moves"]:
                input_vector[hold_to_index[hold["Description"]]] = 1

            output_vector = [0 for i in range(NUM_GRADES)]
            output_vector[grade_map[climb["Grade"]]-4] = 1

            inputs.append(np.array(input_vector))
            outputs.append(np.array(output_vector))

    # def partition_vals(inputs,outputs):
    #     counts = [0 for i in range(NUM_GRADES)]        

    #     test_inputs= []
    #     test_outputs=[]
    #     train_inputs=[]
    #     train_outputs=[]
    #     np.random.shuffle(inputs)
    #     np.random.shuffle(outputs)
    #     for index in range(len(inputs)):
    #         grade_index = np.where(outputs[index]==1)[0][0]
    #         if counts[grade_index]< 2500:
    #             counts[grade_index] +=1
    #             if index%5 !=0:
    #                 train_inputs.append(inputs[index])
    #                 train_outputs.append(outputs[index])
    #             else:
    #                 test_inputs.append(inputs[index])
    #                 test_outputs.append(outputs[index])

    #     print(counts)
    #     # grade = 5
    #     # for index in range(len(inputs)):
    #     #     if outputs[index][grade-4]==1:

    #     #     else:
    #     #         train_inputs.append(inputs[index])
    #     #         train_outputs.append(outputs[index])

    #     return np.array(test_inputs),np.array(test_outputs),np.array(train_inputs),np.array(train_outputs)

    # test_inputs,test_outputs,train_inputs,train_outputs = partition_vals(inputs,outputs)

    # print(test_inputs,test_outputs,train_inputs,train_outputs)
    import random

    random.shuffle(inputs)
    random.shuffle(outputs)

    train_inputs = np.array(inputs[0:int(PARTITION_VALUE*len(inputs))])
    test_inputs = np.array(inputs[int(PARTITION_VALUE*len(inputs)):])

    train_outputs = np.array(outputs[0:int(PARTITION_VALUE*len(inputs))])
    test_outputs = np.array(outputs[int(PARTITION_VALUE*len(inputs)):])

    return (train_inputs,train_outputs), (test_inputs,test_outputs)