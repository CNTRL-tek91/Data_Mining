
#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#11 classes after discretization
classes = [i for i in range(-22, 39, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
train_data = pd.read_csv("weather_training.csv")

#reading the test data
test_data = pd.read_csv("weather_test.csv")

#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
X_training = np.array(train_data.values)[:, 1:-1]
X_testing = np.array(test_data.values)[:, 1:-1]


y_training = np.array(train_data.values)[:, -1]
y_testing = np.array(test_data.values)[:, -1]

def discretization(class_values):

    discretized_values = []

    for value in class_values:
        discretized_values.append(min(classes, key = lambda c: abs(c-value)))

    return discretized_values

discretized_y_training = discretization(y_training)
discretized_y_testing = discretization(y_testing)

highest_knn_accruacy = 0

#loop over the hyperparameter values (k, p, and w) ok KNN

for k in k_values:
    for p in p_values :
        for w in w_values:

            

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, discretized_y_training)

            

            #make the KNN prediction for each test sample and start computing its accuracy
         
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            correct_predictions = 0
            total_predictions = 0

            for x_test_val, y_test_discretized_val in zip(X_testing, discretized_y_testing):

                predictions = clf.predict([x_test_val])[0]

                continuous_val = y_test_discretized_val
                continuous_predictions = predictions

                if continuous_val == 0:
                    if continuous_predictions == 0:
                        correct_predictions += 1
                else:
                    percent_difference = abs(continuous_predictions - continuous_val) / abs(continuous_val) * 100

                    if percent_difference <= 15:
                        correct_predictions += 1
                total_predictions += 1

            knn_accruracy = correct_predictions / total_predictions

                


            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            if knn_accruracy > highest_knn_accruacy:


                highest_knn_accruacy = knn_accruracy

                print(f"Highest KNN accuracy so far: {highest_knn_accruacy}")
                print(f"Parameters: k = {k}, p = {p}, weight = {w}")






