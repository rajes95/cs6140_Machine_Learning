'''
CS 6140 Machine Learning - Assignment 02

5 - Polynomial Regression using Normal Equations - Yacht Dataset

@author: Rajesh Sakhamuru
@version: 6/13/2020
'''

import pandas as pd
import math
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


def tenFoldCrossValidation(originalData, target, powers):
    '''
    Over the polynomial degrees in the powers list, we do 10-fold cross
    validation of least squares regression, and then calculate RMSE values across
    each of the different poly-degrees values.
    
    :param originalData: dataframe with all training and testing data 
    :param target: target column name
    :param powers: polynomial degree values that are used for the data
    '''

    # list of SSEs over the folds
    trainMeanRMSEValues = []
    testMeanRMSEValues = []

    # prevent changes to original data
    data = originalData.copy()

    # shuffle data
    dataShuffled = data.sample(frac=1).reset_index(drop=True)
    numRows = len(dataShuffled)

    oneTenthRows = round(numRows / 10)

    for p in powers:
        print("\nPolynomial Degree:", p)

        trainRMSEValues = []
        testRMSEValues = []

        lowestTestRMSE = math.inf
        bestWeights = []

        # 10-fold cross validation of growTree model
        for n in range(10):
            # split into training and testing data
            testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
            trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])

            normTrainData, normParams, _ = gd.zScoreNormalization(trainData, target)
            normTestData, _, _ = gd.zScoreNormalization(testData, target, normParams=normParams)

            expTrainData, newTarget = getPolynomialData(normTrainData, target, p)
            expTestData, newTarget = getPolynomialData(normTestData, target, p)

            weightsMatrix, _, trainRMSE, _, testRMSE = \
            gd.leastSquaresRegression(expTrainData, expTestData, newTarget, verbose=False)

            trainRMSEValues.append(trainRMSE)
            testRMSEValues.append(testRMSE)

            if testRMSE < lowestTestRMSE:
                lowestTestRMSE = testRMSE
                bestWeights = weightsMatrix

        trainMeanRMSE = sum(trainRMSEValues) / len(trainRMSEValues)
        testMeanRMSE = sum(testRMSEValues) / len(testRMSEValues)
        trainMeanRMSEValues.append(trainMeanRMSE)
        testMeanRMSEValues.append(testMeanRMSE)

        print("Weights with lowest Test RMSE (on normalized data):")
        print(bestWeights)
        print("Train Mean RMSE:", trainMeanRMSE)
        print("Test Mean RMSE:", testMeanRMSE)

        print("---------------------------------------------------")

    resultsSummary = pd.DataFrame()

    resultsSummary['Polynomial Degree'] = powers
    resultsSummary['Train RMSE'] = trainMeanRMSEValues
    resultsSummary['Test RMSE'] = testMeanRMSEValues

    print(resultsSummary)

    title = "Training and Testing Mean RMSE vs. Polynomial Degree"
    plt.plot(powers, trainMeanRMSEValues, 'bo', label="Training Mean RMSE")
    plt.plot(powers, testMeanRMSEValues, 'g^', label="Testing Mean RMSE")
    plt.legend(loc='best')
    plt.title(title)
    plt.ylabel("Mean RMSE Values")
    plt.xlabel("Polynomial Degree")
    plt.show()

    return trainRMSEValues, testRMSEValues


def main():

    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # column names initialized
    colNames = []
    for i in range(7):
        colNames.append(str(i))

    # import data file
    try:
        yachtData = pd.read_csv("datasets/yachtData.csv", names=colNames)
    except:
        print("File not found")

    # add constant feature for w0
    yachtData.insert(0, 'constant', 1)
    colNames = []
    for i in range(8):
        colNames.append(str(i))

    yachtData.columns = colNames
    target = colNames[-1]
    powers = [1, 2, 3, 4, 5, 6, 7]

    print("Yacht Dataset:")
    print(yachtData)

    tenFoldCrossValidation(yachtData, target, powers)


main()
