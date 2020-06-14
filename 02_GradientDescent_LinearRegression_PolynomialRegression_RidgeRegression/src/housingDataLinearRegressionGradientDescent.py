'''
CS 6140 Machine Learning - Assignment 02

2.1 - Gradient Descent for Linear Regression - Housing Dataset

@author: Rajesh Sakhamuru
@version: 6/13/2020
'''

import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import linearRegressionGradientDescent as gd


def main():
    '''
    Runs Gradient descent for Housing dataset to determine linear regression weights
    '''
    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # column names initialized
    colNames = []
    for i in range(14):
        colNames.append(str(i))

    # import data file
    try:
        housingData = pd.read_csv("datasets/housing.csv", names=colNames)
    except:
        print("File not found")

    # add constant feature for w0
    housingData.insert(0, 'constant', 1)
    colNames = []
    for i in range(15):
        colNames.append(str(i))

    housingData.columns = colNames
    target = colNames[-1]

    # set learning rate and tolerance for gradient descent
    learningRate = 0.0004
    tolerance = 0.005

    print(housingData)

    trainSSEValues, trainRMSEValues, testSSEValues, testRMSEValues, bestRMSEList = \
    gd.tenFoldCrossValidation(housingData, target, learningRate, tolerance)

    # return and visualize data
    resultsSummary = pd.DataFrame()

    resultsSummary['Fold #'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    resultsSummary['Train SSE'] = trainSSEValues
    resultsSummary['Train RMSE'] = trainRMSEValues
    resultsSummary['Test SSE'] = testSSEValues
    resultsSummary['Test RMSE'] = testRMSEValues

    print()
    print(resultsSummary)
    print()
    print("Train data SSE Mean over folds:", sum(trainSSEValues) / len(trainSSEValues))
    print("Train data SSE Standard Deviation over folds:", st.pstdev(trainSSEValues))
    print()
    print("Train data RMSE Mean over folds:", sum(trainRMSEValues) / len(trainRMSEValues))
    print("Train data RMSE Standard Deviation over folds:", st.pstdev(trainRMSEValues))
    print()
    print("Test data SSE Mean over folds:", sum(testSSEValues) / len(testSSEValues))
    print("Test data SSE Standard Deviation per fold:", st.pstdev(testSSEValues))
    print()
    print("Test data RMSE Mean over folds:", sum(testRMSEValues) / len(testRMSEValues))
    print("Test data RMSE Standard Deviation over folds:", st.pstdev(testRMSEValues))

    # plot RMSE over iteration for the selected fold
    title = "RMSE Training Plot for Fold #" + str(bestRMSEList[0])
    RMSEList = bestRMSEList[1]
    plt.plot(RMSEList, 'bo')
    plt.title(title)
    plt.ylabel("RMSE Values")
    plt.xlabel("Iteration Number")
    plt.show()


main()
