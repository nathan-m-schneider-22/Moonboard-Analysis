from run_model import *
from moonboard_problems import *
from handwritten_digits import *
import time
"""
Nathan Schneider schnei.nathan@gmail.com
moonboard_analysis.py

This program serves as the driver for parsing
data, creating and evaluating neural networks
for the classification of climbing grades for
moonboard problems, as well as a comparison with 
handwritten digit recognition to show the 
validity of the attempted methods. 
"""

#This is a count of the number of climbs of each grade, used to deal with the imbalanced datatset
grade_counts = {0:6390,
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
#For runtime tracking 
start_time = time.time()

#Retrive the data for the climbs from a downloaded moonbaord json. 
(train_x,train_y),(test_x,test_y) = get_climb_data("data/moonboard_problems_setup_2016.json",repeats_count=5)


#Train the unweighted model, and evaluate it, as well as save a histogram for the results
unweighted_climb_model = train_model(train_x,train_y)
evaluate_model(unweighted_climb_model,test_x,test_y,"Climb Categorization (Unweighted Training)",x_axis_offset=4)

#Train and evaluate the weighted trained model
weighted_climb_model = train_model(train_x,train_y,class_counts=grade_counts)
evaluate_model(weighted_climb_model,test_x,test_y,"Climb Categorization (Weighted Training)",x_axis_offset=4,block = True)

#Get the imbalanced data from the handwritten digit dataset
(train_x,train_y),(test_x,test_y) = get_digit_data(grade_counts)

#Train the unweighted model, and evaluate it, as well as save a histogram for the results
unweighted_digit_model = train_model(train_x,train_y)
evaluate_model(unweighted_digit_model,test_x,test_y,"Digit Categorization (Unweighted Training)")

#Train and evaluate the weighted trained model
weighted_digit_model = train_model(train_x,train_y,class_counts=grade_counts)
evaluate_model(weighted_digit_model,test_x,test_y,"Digit Categorization (Weighted Training)")

print("Total runtime: ",time.time()-start_time)