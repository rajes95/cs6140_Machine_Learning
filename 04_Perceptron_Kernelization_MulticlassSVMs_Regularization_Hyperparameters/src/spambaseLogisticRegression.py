'''
CS 6140 Machine Learning - Assignment 04

Problem 2.4 - Regularized Logistic Regression - Spambase Dataset

@author: Rajesh Sakhamuru
@version: 7/20/2020
'''

import pandas as pd
import matplotlib.pyplot as plt

import logisticRegressionFunctions as lr


def main():
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    colNames = []
    for i in range(58):
        colNames.append(str(i))

    # import data file
    try:
        spamData = pd.read_csv("datasets/spambase.csv", names=colNames)
    except:
        print("File not found")

    # add constant feature for w0
    spamData.insert(0, 'constant', 1)
    colNames = []
    for i in range(59):
        colNames.append(str(i))

    spamData.columns = colNames
    target = colNames[-1]

    print("Spambase Data:")
    print(spamData, "\n")

    learningRate = 0.001
    tolerance = 0.00005

    resultsTable, trainConfusionMatrix, testConfusionMatrix, (bestFold, bestLogLossList) = \
    lr.tenFoldCrossValidation(spamData, target, learningRate, tolerance, verbose=True)

    print("Results Table:")
    print(resultsTable)

    print("\nColumns are actual value, Rows are predicted value")
    print("Training Confusion Matrix:")
    print(trainConfusionMatrix)

    print("\nTesting Confusion Matrix:")
    print(testConfusionMatrix)

    # plot Log Loss over iteration for the selected fold
    title = "Log-Loss Training Plot for Fold #" + str(bestFold)
    RMSEList = bestLogLossList
    plt.plot(RMSEList, 'bo')
    plt.title(title)
    plt.ylabel("Log-Loss Values")
    plt.xlabel("Gradient Descent Iteration Number")
    plt.show()

"""
















"""
main()
