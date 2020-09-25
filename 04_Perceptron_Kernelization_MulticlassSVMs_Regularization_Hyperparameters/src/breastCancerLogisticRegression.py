'''
CS 6140 Machine Learning - Assignment 04

Problem 2.4 - Regularized Logistic Regression - Breast Cancer Dataset

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
    for i in range(31):
        colNames.append(str(i))

    # import data file
    try:
        cancerData = pd.read_csv("datasets/breastcancer.csv", names=colNames)
    except:
        print("File not found")

    # add constant feature for w0
    cancerData.insert(0, 'constant', 1)
    colNames = []
    for i in range(32):
        colNames.append(str(i))

    cancerData.columns = colNames
    target = colNames[-1]

    print("Breast Cancer Data:")
    print(cancerData, "\n")

    learningRate = 0.005
    tolerance = 0.00005

    resultsTable, trainConfusionMatrix, testConfusionMatrix, (bestFold, bestLogLossList) = \
    lr.tenFoldCrossValidation(cancerData, target, learningRate, tolerance, verbose=True)

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
