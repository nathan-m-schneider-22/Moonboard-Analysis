# Moonboard-Analysis
This repo holds the nerual net for predicting the V grade of a given moonboard problem, given its stated holds. 
To run this repo, run ` python moonboard_analysis.py `
Make sure you have the necessary keras, matplotlib, tensorflow, numpy libraries. This repo runs on Python 3.6
## Data
The data are a set of climbs, each climb with a list of holds on the climb, and a difficulty grade. These holds are taken from a grid and mapped into a 142 dimensional input vector. The output vector is a onehot encoded grade from 4-14. You can find moonboard climbs here [https://www.moonboard.com/].

The data was taken from another repository[https://github.com/e-sr/moonboard], for building your own LED moonboard with loaded problems. This data was then cleaned, with only climbs that had been repeated by other users a specified (5) number of times to be considered usable. The data was then put into vector form based on the hold set, with 0s corresponding to lacking the hold, and 1s with having the hold. 

For the justification of the method with another dataset, the MNIST handwritten digit dataset was used. This was downloaded through the keras datasets. 

## Algorithm
The chosen algorithm here is a multilayer perceptron, implemented through the keras library. It is 
 - Multilayer, with an input layer, one hidden layer, and output layer
 - Supervised, with training t values for training inputs
 - Non-linear, with nonlinear activation functions
 - Feedforward
 - Dense
 - Non-topographical

Our MLP was implemented with ther keras library, see the train_model function of [run_model.py](./run_model.py). Our analysis was done using the model.predict method. 

## Results 
Other results, such as the results of [Andrew Houghton](https://github.com/andrew-houghton/moon-board-climbing), and this [Stanford Project](http://cs229.stanford.edu/proj2017/final-reports/5232206.pdf) reported success rates of real grade classification around 35%. After examination through the graphics in the stanford paper, and examining firsthand the results of Houghton's repo, found that they were to an extent exploiting imbalances in the data, as there were far more easier climbs than harder, and the networks would consistently guess low grades and be correct more often. I found this to be easy to replicate with my own results. 


![Climb_Categorization_(Unweighted_Training).png](Climb_Categorization_(Unweighted_Training).png)


You can see that the network is over-gussing for V4 grade climbs, almost double of how many there actually are. Doing this, it can achieve the 35% accuracy rate, as this is approximately the percentage of V4s. 

To combat this, I decided to weight the less common grades higher than common grades when it comes to training weights. In doing so, each training iteration had a weight inversely proportional to the number of grades in the data set. While this fix did remove the low grade bias from the system, the accuracy dropped significantly, down to approximately 20% (+-5% depending on the training/testing sets)


![Climb_Categorization_(Weighted_Training).png](Climb_Categorization_(Weighted_Training).png)


As seen, this reduces the low grade bias, and spreads the guesses around to other categories. It appears that this dataset may necessarily lead to prediction with the MLP algorithm. 

To confirm that this was a problem with the dataset and not the algorithm, I examined classifying handwritten digits with the MNIST dataset. I then reduced the dataset to mirror the distribution of climbs withtin the climb dataset, so that there would be many 0s, 1s, and very few high digits. As predicted, the digit classification performed very well (98% accuracy) with the imbalanced dataset 


![Digit_Categorization_(Unweighted_Training).png](Digit_Categorization_(Unweighted_Training).png).


And when the weighting was applied to it, the results remained similar. 


![Digit_Categorization_(Weighted_Training).png](Digit_Categorization_(Weighted_Training).png).


As we can see, the method of MLP, with or without class weight adjustment, is valid and effective on predictable datasets, but less so on the climbing dataset. 
