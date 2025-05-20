
#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
X_training = np.array(pd.read_csv("weather_training.csv").iloc[:, 1:-1])
y_training = np.array(pd.read_csv("weather_training.csv").iloc[:, -1])

#update the training class values according to the discretization (11 values only)


def discretization(training_vals):
    discretized_vals = []

    for vals in training_vals:
        discretized_vals.append(min(classes, key = lambda c: abs(c-vals)))

    return discretized_vals


#reading the test data
X_test = np.array(pd.read_csv("weather_test.csv").iloc[:, 1:-1])
y_test = np.array(pd.read_csv("weather_test.csv").iloc[:, -1])

#update the test class values according to the discretization (11 values only)
y_test = discretization(y_test)

highest_naive_bayes_accuracy = 0
#loop over the hyperparameter value (s)
y_train_discretized = discretization(y_training)

for val in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=val)
    clf = clf.fit(X_training, y_train_discretized)

    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    correct_predictions = 0
    total_predictions = 0

    for x_test_val, y_test_discretized_val in zip(X_test, y_test):

        predictions = clf.predict([x_test_val])[0]

        if y_test_discretized_val == 0:
            if predictions == 0:
                correct_predictions += 1

        else:
            percent_difference = abs(predictions - y_test_discretized_val) / abs(y_test_discretized_val) * 100



            if percent_difference <= 15:

                correct_predictions += 1

        
        total_predictions += 1
    test_accuracy = correct_predictions / total_predictions


    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    if test_accuracy > highest_naive_bayes_accuracy:
        highest_naive_bayes_accuracy = test_accuracy

        
        print(f"Highest Naïve Bayes accuracy so far: {highest_naive_bayes_accuracy}")
        print(f"Parameter: s = {val}")


