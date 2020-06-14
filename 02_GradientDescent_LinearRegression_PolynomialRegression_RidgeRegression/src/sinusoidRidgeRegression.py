'''
CS 6140 Machine Learning - Assignment 02

7 - Ridge Regression - Sinusoid Dataset

@author: Rajesh Sakhamuru
@version: 6/13/2020
'''

import numpy as np
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


def tenFoldCrossValidation(originalData, target, powers, lambdaValues):
    '''
    Over the polynomial degrees in the powers list then over the lambdaValues for each 
    power, we do 10-fold cross validation of least squares regression, and then 
    calculate RMSE values across each of the different poly-degrees values.
    
    :param originalData: dataframe with all training and testing data 
    :param target: target column name
    :param powers: polynomial degree values that are used for the data
    :param lambdaValues: all lambda values tested at each polynomial Degree
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

    trainGraphData = []
    testGraphData = []

    # loop polynomial degrees (will not graph if len(powers) != 2)
    for p in powers:
        print("\nPolynomial Degree:", p)

        trainMeanRMSEValues = []
        testMeanRMSEValues = []
        
        # loop lambda values
        for lamVal in lambdaValues:

            trainRMSEValues = []
            testRMSEValues = []
            # 10-fold cross validation of growTree model
            for n in range(10):
                # split into training and testing data
                testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
                trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])
                
                # center all data
                cenTrainData, targetMean, meanList = centerData(trainData, target)
                cenTestData, _, _ = centerData(testData, target, meanList=meanList)
                
                # make data polynomial based on degree
                polyTrainData, newTarget = getPolynomialData(cenTrainData, target, p)
                polyTestData, _newTarget = getPolynomialData(cenTestData, target, p)

                # z-score normalize data
                normTrainData, normParams, _ = gd.zScoreNormalization(polyTrainData, newTarget)
                normTestData, _, _ = gd.zScoreNormalization(polyTestData, newTarget, normParams=normParams)

                # because centered data
                w0 = targetMean

                _weightsMatrix, _trainSSE, trainRMSE, _testSSE, testRMSE = \
                ridgeRegression(normTrainData, normTestData, newTarget, w0, lamVal)

                trainRMSEValues.append(trainRMSE)
                testRMSEValues.append(testRMSE)

            trainMeanRMSE = sum(trainRMSEValues) / len(trainRMSEValues)
            testMeanRMSE = sum(testRMSEValues) / len(testRMSEValues)
            trainMeanRMSEValues.append(trainMeanRMSE)
            testMeanRMSEValues.append(testMeanRMSE)

        print()

        resultsSummary = pd.DataFrame()

        resultsSummary['Lambda Values'] = lambdaValues
        resultsSummary['Train RMSE'] = trainMeanRMSEValues
        resultsSummary['Test RMSE'] = testMeanRMSEValues

        print(resultsSummary)
        print("---------------------------------------------------")
        trainGraphData.append(trainMeanRMSEValues)
        testGraphData.append(testMeanRMSEValues)

    if len(powers) == 2:
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(lambdaValues, trainGraphData[0], 'go')
        axs[0, 0].set_title('Training RMSE at Polynomial Degree ' + str(powers[0]))
        axs[0, 1].plot(lambdaValues, testGraphData[0], 'go')
        axs[0, 1].set_title('Testing RMSE at Polynomial Degree ' + str(powers[0]))
        axs[1, 0].plot(lambdaValues, trainGraphData[1], 'bo')
        axs[1, 0].set_title('Training RMSE at Polynomial Degree ' + str(powers[1]))
        axs[1, 1].plot(lambdaValues, testGraphData[1], 'bo')
        axs[1, 1].set_title('Testing RMSE at Polynomial Degree ' + str(powers[1]))

        for ax in axs.flat:
            ax.set(xlabel='Lambda Value', ylabel='Mean RMSE Value')
        fig.tight_layout()
#         for ax in axs.flat:
#             ax.label_outer()

        plt.show()

    return trainGraphData, testGraphData


def centerData(originalData, target, meanList=None):
    '''
    Subtract mean of each column from all values in that column, including feature
    
    :param originalData: pandas dataframe
    :param target: target column name
    :param meanList: list of means that can be used to center testing data, default is None
    '''
    
    data = originalData.copy()

    colNames = gd.getColNames(data)

    testingData = False

    if meanList is None:
        meanList = []
    else:
        testingData = True
    count = 0
    targetMean = 0

    for col in colNames:
        if testingData is False:
            mean = (sum(data[col]) / len(data[col]))
            if col == target:
                targetMean = mean
            data[col] = data[col] - mean
            meanList.append(mean)
        else:
            data[col] = data[col] - meanList[count]
            count += 1

    return data, targetMean, meanList


def ridgeRegression(origTrainData, origTestData, target, w0, lambdaVal):
    '''
    Use closed form solution for ridge regression to calculate the weights matrix,
    if lambdaVal is 0, then it is linear regression.
    
    :param origTrainData: train data pandas dataframe
    :param origTestData: test data pandas dataframe
    :param target: target column name
    :param w0: mean of the target column is w0, for use in calculating SSE
    :param lambdaVal: limits overfitting of data, but if value is too high can underfit
    '''

    trainData = origTrainData.copy()
    testData = origTestData.copy()

    targetCol = trainData[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(trainData.drop(columns=[target]))

    numFeatures = len(gd.getColNames(trainData.drop(columns=[target])))

    idMatrixXLambda = lambdaVal * np.identity(numFeatures)
    
    # closed form solution for calculating weights using ridge regression
    weightsMatrix = np.linalg.inv(np.add((dataMatrix.T.dot(dataMatrix)), idMatrixXLambda)).dot(dataMatrix.T.dot(targetMatrix))

    trainSSE = calculateSSE(trainData, target, weightsMatrix, w0)
    trainRMSE = math.sqrt(trainSSE / len(trainData))

    testSSE = calculateSSE(testData, target, weightsMatrix, w0)
    testRMSE = math.sqrt(testSSE / len(testData))

    return weightsMatrix, trainSSE, trainRMSE, testSSE, testRMSE


def calculateSSE(data, target, weightsMatrix, w0):
    '''
    After gradient descent, this function can be used to determine 
    the SSE indicating how much error the model has over the data and its 
    predictions.
     
    :param data: testing data dataframe object
    :param target: target column name
    :param weightsMatrix: weights matrix not including w0, calculated via ridge regression
    :param w0: mean of target column as y-intercept weight
    '''

    targetCol = data[target].values.tolist()

    testData = data.drop(columns=[target])
    testRowsList = testData.values.tolist()
    SSE = 0

    # calculate squared errors for each row in testing data and sum them for the SSE
    for n in range(len(testRowsList)):
        row = testRowsList[n]
        row = np.array(row)
        # w0 is added to both because both prediction and actual were centered
        prediction = float(w0 + np.sum((weightsMatrix.T * row)))
        actual = float(w0 + targetCol[n])
#         print("prediction:", prediction, "actual:", actual, "  ")

        SSE += (prediction - actual) * (prediction - actual)

    return SSE


def main():

    # data visualization options for console output
    pd.set_option('display.max_columns', None)
#     pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # column names and target column name initialized
    colNames = []
    for i in range(2):
        colNames.append(str(i))
    target = colNames[-1]

    # import data file
    try:
        trainData = pd.read_csv("datasets/sinData_Train.csv", names=colNames)
    except:
        print("File not found")

    powers = [5, 9]

    lambdaValues = []
    for n in range(51):
        lambdaValues.append(n * 0.2)

    print("Sinusoid Training Dataset:")
    print(trainData)

    tenFoldCrossValidation(trainData, target, powers, lambdaValues)


main()

