""" 
Author: Daishak Patel
UID: U84709314
Class: Affective Computing
Project 2

"""

import sys
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

import os
import matplotlib.pyplot as plt

# Read data from CSV file
# store the data in a list of lists
def readData(path):
    csvData = []
    with open(path, 'r') as f: # open the CSV file
        reader = csv.reader(f) # create a CSV reader object
        for row in reader: # iterate over each row in the CSV file
            subjectID = row[0] # get the subject ID
            dtype = row[1] # get the data type
            cls = row[2] # get the class
            values = [float(x) for x in row[3:]] # convert the values to float
            csvData.append([subjectID, dtype, cls, values]) # append the data to the list
    return csvData


# Extract the features, i.e, mean, variance, min, max
def extractFeatures(csvData, dataType):
    listOfDataTypes = ["dia", "sys", "eda", "res"] # data types
    DataLists = {"dia": [], "sys": [], "eda": [], "res": []} # data array to hold the data for each data type
    # iterate over each data type
    for type in listOfDataTypes: # iterate over each data type
        # get the data for the current data type
        DataLists[type] = [x for x in csvData if type in x[1].lower()]

    # Initialize the features, labels, and subjects which holds the according data type
    features = {"dia": [], "sys": [], "eda": [], "res": [], "all": []} 
    labels = {"dia": [], "sys": [], "eda": [], "res": [], "all": []}
    subjects = {"dia": [], "sys": [], "eda": [], "res": [], "all": []}

    # Compute features
    for type in DataLists: # iterate over each data type
        data = DataLists[type] # get the data for the current data type
        for subjectID, dtype, cls, values in data: # iterate over each row in the data
            mean = np.mean(values) # compute the mean of the values
            # compute the variance of the values
            var = np.var(values)
            # compute the minimum value of the values
            min_val = np.min(values)
            # compute the maximum value of the values
            max_val = np.max(values)
            feature = [mean, var, min_val, max_val] # store the computed features in a list
            features[type].append(feature) # append the features to the features list
            labels[type].append(cls) # append the class to the labels list
            subjects[type].append(subjectID) # append the subject id to the subjects list
     # if the data type is all then combine the features, labels, and subjects for all data types into a single list
    if dataType == "all":
        length = len(features["dia"]) # get the length of the features for the data type dia
        for i in range(length): # iterate over the length
            row = features["dia"][i] + features["sys"][i] + \
                features["eda"][i] + features["res"][i]
            features["all"].append(row) # append the row to the features list
        labels["all"] = labels["dia"] # set the labels for all data types to the labels for the data type dia
        subjects["all"] = subjects["dia"] # set the subjects for all data types to the subjects for the data type dia
    return features, labels[dataType], subjects[dataType]

# Print the evaluation metrics for the classifier using subject-independent cross-validation
def PrintEvalMetrics(X, y, dataType, subjects):
    # Function to perform subject-independent cross-validation
    def subject_independent_cross_validation(X, y, clf, dataType, subjects):
        # set the number of folds to 10
        n_folds = 10

        # get the groups for the cross-validation (in this case, the subjects)
        groups = subjects

        # create a GroupKFold cross-validator with the specified number of folds
        gkf = GroupKFold(n_splits=n_folds)
        gkf.get_n_splits(X, y, groups)

        # initialize lists to store the evaluation metrics for each fold
        confusion_matrix_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []

        # iterate over each fold
        for i in range(10):
            # iterate over each train-test split in the current fold
            for fold_index, (train_index, test_index) in enumerate(gkf.split(X, y, groups)):
                # if the current train-test split belongs to the current fold
                if i == fold_index:
                    # split the data into training and testing sets based on the current train-test split
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # train the classifier on the training data and evaluate it on the testing data
                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test) # predict the class labels for the testing data
                    # compute the confusion matrix, precision, recall, and accuracy for the current fold
                    finalPredictions = pred
                    groundTruth = y[test_index]
                    cm = confusion_matrix(finalPredictions, groundTruth)
                    precision = precision_score(
                        groundTruth, finalPredictions, average='macro')
                    recall = recall_score(
                        groundTruth, finalPredictions, average='macro')
                    accuracy = accuracy_score(
                        groundTruth, finalPredictions)

                    # append the evaluation metrics for the current fold to the respective lists
                    confusion_matrix_scores.append(cm)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    accuracy_scores.append(accuracy)

                    # write the evaluation metrics for the current fold to a file for specific data type under Results folder
                    new_dir = 'Results/'
                    os.makedirs(new_dir, exist_ok=True) # create the Results folder if it does not exist
                    filename = dataType+".txt" # create a file with the data type as the filename
                    try: # try to write the evaluation metrics to the file
                        with open(os.path.join(new_dir, filename), 'a') as file:
                            file.write('Fold ' + str(fold_index + 1)+'\n')
                            file.write("\tConfusion matrix: " +
                                       str(cm) + '\n')
                            file.write("\tPrecision: " +
                                       str(precision) + '\n')
                            file.write("\tRecall: " + str(recall) + '\n')
                            file.write("\tAccuracy: " +
                                       str(accuracy) + '\n')
                    except Exception as e: # handle any errors that occur while writing to the file
                        print("Error while writing to file:", e)

        # return the lists of evaluation metrics for all folds
        return confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores

    # Initialize the classifier
    X = X[dataType] # get the features for the specified data type
    X = np.array(X) # convert the features to a numpy array
    y = np.array(y) # convert the labels to a numpy array
    subjects = np.array(subjects) # convert the subjects to a numpy array
    clf = RandomForestClassifier() # initialize the classifier
    confusion_matrix_scores, precision_scores, recall_scores, accuracy_scores = subject_independent_cross_validation(
        X, y, clf, dataType, subjects) # perform subject-independent cross-validation

    # Compute and return the average score across all folds
    avg_cm = np.mean((confusion_matrix_scores), axis=0) 
    avg_precision = np.mean((precision_scores), axis=0)
    avg_recall = np.mean((recall_scores), axis=0)
    avg_accuracy = np.mean((accuracy_scores), axis=0)

    # Save the average score across all folds in a file
    new_dir = 'Results/'
    os.makedirs(new_dir, exist_ok=True)
    filename = dataType+".txt"
    try:
        with open(os.path.join(new_dir, filename), 'a') as file: # write the average scores to the file
            file.write('\nAverage Scores \n')
            file.write("\tConfusion matrix: "+str(avg_cm) + '\n')
            file.write("\tPrecision: " + str(avg_precision) + '\n')
            file.write("\tRecall: " + str(avg_recall) + '\n')
            file.write("\tAccuracy: " + str(avg_accuracy) + '\n')
    except Exception as e: # handle any errors that occur while writing to the file
        print("Error while writing to file:", e)

    # return the average scores
    return avg_cm, avg_precision, avg_recall, avg_accuracy
# Code for boxplot

# Code for plotting the physiological signals in one line graph
def PhysiologicalSignalsPlot():
    # read the data from the CSV file
    data = readData('Project2Data.csv')

    # get the data for the signals in rows 453 to 456
    # get the data for M027 Pain
    signal1 = data[453] 
    signal2 = data[454]
    signal3 = data[455]
    signal4 = data[456]

    # get the values for the first signal
    values1 = signal1[3]
    values2 = signal2[3]
    values3 = signal3[3]
    values4 = signal4[3]

    # get the classifier for the first signal
    classifier1 = signal1[1]
    classifier2 = signal2[1]
    classifier3 = signal3[1]
    classifier4 = signal4[1]

    # plot the signals
    plt.figure(figsize=(10, 5)) # set the size of the plot
    plt.plot(values1, label=classifier1) # plot the first signal
    plt.plot(values2, label=classifier2)
    plt.plot(values3, label=classifier3)
    plt.plot(values4, label=classifier4) # plot the fourth signal
    plt.legend()
    plt.title('Physiological Signals') # set the title of the plot
    plt.xlabel('Time') # set the x-axis label
    plt.ylabel('Value') # set the y-axis label
    plt.show() # show the plot

# Code for boxplot of the features extracted
def box_plot(features):
    # Initialize the lists to store the mean, variance, min, and max values
    mean = []
    variance = []
    min = []
    max = []

    # Get the features for each data type
    dia = features["dia"]
    sys = features["sys"]
    eda = features["eda"]
    res = features["res"]
    # Combine the features for all data types
    data = np.concatenate((dia, sys, eda, res))
    # Get the mean, variance, min, and max values
    mean = data[:, 0]
    variance = data[:, 1]
    min = data[:, 2]
    max = data[:, 3]
    # Combine data into a single list
    all_data = [mean, variance, min, max]
    # Create a box plot
    fig, ax = plt.subplots()
    ax.boxplot(all_data)

    # Set the tick labels
    ax.set_xticklabels(['Mean', 'Variance', 'Min', 'Max'])

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Command Line Arguments
    dataType = sys.argv[1]
    data_file = sys.argv[2]

    data = readData(data_file)

    features, labels, subjects = extractFeatures(data, dataType)

    PrintEvalMetrics(features, labels, dataType, subjects)

    # Uncomment the following lines to run the box_plot and PhysiologicalSignalsPlot functions
    # which will plot the box plot of the extracted features and the physiological signals respectively
    # Uncommenting both at the same time will run both but it will be displayed one after the other
    # so once the box plot window is closed, the physiological signals plot will be displayed
    
    # box_plot(features)
    # PhysiologicalSignalsPlot()
