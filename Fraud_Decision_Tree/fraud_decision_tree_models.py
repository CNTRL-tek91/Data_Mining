


#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []
    finalAccuracy = 0

    df = pd.read_csv(ds, sep=',', header=0)   #reading the training data by using Pandas library
    data_training = np.array(df.values)[:,1:] #eliminating the first feature (id)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    

    refund = {
        "Yes": 1,
        "No": 2,
    }

    marital = {
        "Single": [1,0,0],
        "Divorced": [0,1,0],
        "Married": [0,0,1],
    }

    for data in data_training:
        X.append([refund[data[0]], marital[data[1]][0], marital[data[1]][1], marital[data[1]][2], float(data[2].replace("k",""))])

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
   

    cheat = {
        "Yes": 1,
        "No": 2,
    }

    for data in data_training:
        Y.append(cheat[data[3]])

    #loop your training and test tasks 10 times here
    for i in range(10):

        #fitting the decision tree to the data by using Gini index and no max_depth
        clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
        clf = clf.fit(X, Y)

        #plotting the decision tree
        #tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
        #plt.show()

        df = pd.read_csv('cheat_test.csv', sep=',', header=0) #reading the test data by using Pandas library
        data_test = np.array(df.values)[:,1:]                 #eliminating the first feature (id)

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for data in data_test:

            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label

            class_predicted = clf.predict([[refund[data[0]], marital[data[1]][0], marital[data[1]][1], marital[data[1]][2], float(data[2].replace("k",""))]])[0]
            true_label = cheat[data[3]]

            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            if true_label == 1:
               if class_predicted == 1:
                  tp = tp +1
               else:
                  fn = fn +1
            else:
               if class_predicted == 1:
                  fp = fp +1
               else:
                  tn = tn +1

        accuracy = (tp + tn)/(tp + tn + fp + fn)
        #print(accuracy)

        #find the average accuracy of this model during the 10 runs (training and test set)
        finalAccuracy += accuracy

    #print the accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2

    print("finalAccuracy when training on " + ds + ": " + str(finalAccuracy/10))






