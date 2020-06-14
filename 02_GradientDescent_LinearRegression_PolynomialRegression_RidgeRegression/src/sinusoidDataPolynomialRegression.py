'''
CS 6140 Machine Learning - Assignment 02

5 - Polynomial Regression using Normal Equations - Sinusoid Dataset

@author: Rajesh Sakhamuru
@version: 6/13/2020
'''

import pandas as pd
import matplotlib.pyplot as plt
import linearRegressionGradientDescent as gd


def getPolynomialData(originalData, target, p):
    '''
    Add new features for each feature of higher powers until x^p. Used for
    Polynomial Regression.
    
    :param originalData: pandas dataframe
    :param target: target column name
    :param p: max polynomial degree value
    
    '''

    data = originalData.copy()

    colNames = gd.getColNames(data)

    targetCol = data[target]

    newData = pd.DataFrame()

    for col in colNames:
        if col == target:
            continue
        uniques = list(data[col].unique())
        if len(uniques) == 1:
            newData = pd.concat([newData.reset_index(drop=True), data[col].reset_index(drop=True)], axis=1)
            continue
        colExps = pd.DataFrame()

        for i in range(1, p + 1):
            colExps[i] = data[col].pow(i)

        newData = pd.concat([newData.reset_index(drop=True), colExps.reset_index(drop=True)], axis=1)

    newData = pd.concat([newData.reset_index(drop=True), targetCol.reset_index(drop=True)], axis=1)

    colNames = gd.getColNames(newData)
    newColNames = []
    for i in range(len(colNames)):
        newColNames.append(str(i))
    colNames = newColNames
    target = colNames[-1]

    newData.columns = colNames

    return newData, target


def polynomialRegressions(trainData, testData, target, powers):
    '''
    For each polynomial degree, we are testing least squares regression to see
    which degree gives the least error, then the results are graphed.
    
    :param trainData: Training pandas dataframe
    :param testData: testing pandas dataframe
    :param target: target column name
    :param powers: list of polynomial degrees used in LSR
    '''

    trainMeanSSEValues = []
    trainRMSEValues = []
    testMeanSSEValues = []
    testRMSEValues = []

    for p in powers:
        print("\nPolynomial Degree:", p)

        # make dataset polynomial to the value of p
        expTrainData, newTarget = getPolynomialData(trainData, target, p)
        expTestData, _ = getPolynomialData(testData, target, p)

        weightsMatrix, trainSSE, trainRMSE, testSSE, testRMSE = \
        gd.leastSquaresRegression(expTrainData, expTestData, newTarget, verbose=False)

        print("Weights:")
        print(weightsMatrix)
        print('Training Mean SSE:', trainSSE / len(trainData))
        print('Training RMSE:', trainRMSE)
        print('Testing Mean SSE:', testSSE / len(testData))
        print('Testing RMSE:', testRMSE)
        print('-----------------------------------')

        trainMeanSSEValues.append(trainSSE / len(trainData))
        trainRMSEValues.append(trainRMSE)
        testMeanSSEValues.append(testSSE / len(testData))
        testRMSEValues.append(testRMSE)

    resultsSummary = pd.DataFrame()

    resultsSummary['Polynomial Degree'] = powers
    resultsSummary['Train Mean SSE'] = trainMeanSSEValues
    resultsSummary['Train RMSE'] = trainRMSEValues
    resultsSummary['Test Mean SSE'] = testMeanSSEValues
    resultsSummary['Test RMSE'] = testRMSEValues

    print(resultsSummary)

    title = "Training and Testing Mean SSE vs. Polynomial Degree"
    plt.plot(powers, trainMeanSSEValues, 'bo', label="Training Mean SSE")
    plt.plot(powers, testMeanSSEValues, 'g^', label="Testing Mean SSE")
    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel("Mean SSE Values")
    plt.xlabel("Polynomial Degree")
    plt.show()

    return


def main():

    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # column names initialized
    colNames = []
    for i in range(2):
        colNames.append(str(i))

    # import data file
    try:
        trainData = pd.read_csv("datasets/sinData_Train.csv", names=colNames)
        testData = pd.read_csv("datasets/sinData_Validation.csv", names=colNames)
    except:
        print("File not found")

    # add constant feature to training and testing, then reset colNames and target
    trainData.insert(0, 'constant', 1)
    testData.insert(0, 'constant', 1)
    colNames = []
    for i in range(3):
        colNames.append(str(i))
    trainData.columns = colNames
    testData.columns = colNames
    target = colNames[-1]

    # polynomial degrees tested:
    powers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    print("Sinusoid Training Dataset:")
    print(trainData)

    polynomialRegressions(trainData, testData, target, powers)


main()
