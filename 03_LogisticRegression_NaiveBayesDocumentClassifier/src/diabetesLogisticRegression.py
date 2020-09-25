'''
CS 6140 Machine Learning - Assignment 03

Problem 1.3 - Logistic Regression - Pima Indian Diabetes Dataset

@author: Rajesh Sakhamuru
@version: 6/20/2020
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
    for i in range(9):
        colNames.append(str(i))

    # import data file
    try:
        diabetesData = pd.read_csv("datasets/diabetes.csv", names=colNames)
    except:
        print("File not found")

    # add constant feature for w0
    diabetesData.insert(0, 'constant', 1)
    colNames = []
    for i in range(10):
        colNames.append(str(i))

    diabetesData.columns = colNames
    target = colNames[-1]

    print("Pima Indian Diabetes Data:")
    print(diabetesData, "\n")

    learningRate = 0.003
    tolerance = 0.00005

    resultsTable, trainConfusionMatrix, testConfusionMatrix, (bestFold, bestLogLossList) = \
    lr.tenFoldCrossValidation(diabetesData, target, learningRate, tolerance, verbose=True)

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
