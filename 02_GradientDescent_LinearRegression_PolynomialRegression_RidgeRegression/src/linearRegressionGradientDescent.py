'''
CS 6140 Machine Learning - Assignment 02

Regression

@author: Rajesh Sakhamuru
@version: 6/13/2020
'''
import numpy as np
import statistics as st
import math


def tenFoldCrossValidation(originalData, target, learningRate, tolerance):
    '''
    This function generates decision trees based on 10 unique folds of the 
    dataset provided. Then the accuracy values over the folds and the confusion 
    matrix are returned.
    
    :param originalData: dataframe with all training and testing data 
    :param target: target column name
    :param learningRate: learning rate for gradient descent
    :param tolerance: tolerence threshold to determine when to stop gradient descent
    '''
    # prevent changes to original data
    data = originalData.copy()

    # shuffle data
    dataShuffled = data.sample(frac=1).reset_index(drop=True)
    numRows = len(dataShuffled)

    oneTenthRows = round(numRows / 10)

    # list of SSEs over the folds
    trainSSEValues = []
    trainRMSEValues = []
    testSSEValues = []
    testRMSEValues = []

    lowestRMSE = math.inf
    bestFold = 0
    bestRMSEList = []

    # 10-fold cross validation of growTree model
    for n in range(10):
        print("\nFold #", n + 1)
        # split into training and testing data
        testData = dataShuffled[(n * oneTenthRows):oneTenthRows * (n + 1)]
        trainData = dataShuffled.drop(dataShuffled.index[(n * oneTenthRows):oneTenthRows * (n + 1)])

        normTrainData, normParams, _ = zScoreNormalization(trainData, target)
        normTestData, _, _ = zScoreNormalization(testData, target, normParams=normParams)

        if n == 0:
            print('------LSR using Normal Equations (for Fold#1)---------')
            leastSquaresRegression(normTrainData, normTestData, target)

        weightsMatrix, RMSEList = gradientDescent(normTrainData, target, learningRate, tolerance)
        print("Weights for normalized data using Gradient Descent:")
        print(weightsMatrix)

        trainSSE = calculateSSE(normTrainData, target, weightsMatrix)
        trainRMSE = math.sqrt(trainSSE / len(normTrainData))
        print('Training SSE:', trainSSE)
        print('Training RMSE:', trainRMSE)

        testSSE = calculateSSE(normTestData, target, weightsMatrix)
        testRMSE = math.sqrt(testSSE / len(normTestData))
        print('Testing SSE:', testSSE)
        print('Testing RMSE:', testRMSE)

        if trainRMSE < lowestRMSE:
            lowestRMSE = trainRMSE
            bestFold = n + 1
            bestRMSEList = RMSEList

        trainSSEValues.append(trainSSE)
        trainRMSEValues.append(trainRMSE)
        testSSEValues.append(testSSE)
        testRMSEValues.append(testRMSE)
        print("---------------------------------------------------")

    return trainSSEValues, trainRMSEValues, testSSEValues, testRMSEValues, (bestFold, bestRMSEList)


def getColNames(data):
    '''
    Returns list of column names of data given
    :param data: pandas dataframe
    '''
    return list(data.columns)


def zScoreNormalization(originalData, target, normParams=None):
    '''
    z-score normalization of features in data parameter. If mean and standard
    deviation are provided for each feature in normParams, then those are used to
    normalize the data.
    
    :param originalData: pandas dataframe
    :param target: target column name
    :param normParams: list of (mean, stdDev) tuples for each feature
    '''

    data = originalData.copy()

    colNames = getColNames(data)

    if target in colNames:
        colNames.remove(target)

    testing = False

    minMaxParams = []

    # if using given normalization parameters to normalize data
    if normParams is None:
        normParams = []
    else:
        testing = True

    count = 0

    for col in colNames:
        if col == '0':
            normParams.append((1, 0))
            count += 1
            continue
        # min and max feature values for feature re-scaling if wanted
        minMaxParams.append((data[col].min(), data[col].max()))

        if len(data[col].unique()) == 1:
            data[col] = 0

            if not testing:
                normParams.append((0, 0))

            count += 1
            continue
        if not testing:
            mean = data[col].mean()
            stdDev = st.pstdev(list(data[col]))
            normParams.append((mean, stdDev))
        else:
            mean, stdDev = normParams[count]

        data[col] = data[col] - mean
        data[col] = round(data[col] / stdDev, 3)

        count += 1

    return data, normParams, minMaxParams


def leastSquaresRegression(origTrainData, origTestData, target, verbose=True):
    '''
    least squares regression using normal equations for linear regression
    
    :param origTrainData: Training Data
    :param origTestData: Testing Data
    :param target: target column name
    :param verbose: True by default, and if true, will print details about LSR
    '''

    trainData = origTrainData.copy()
    testData = origTestData.copy()

    targetCol = trainData[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(trainData.drop(columns=[target]))

    weightsMatrix = np.linalg.inv(dataMatrix.T.dot(dataMatrix)).dot(dataMatrix.T.dot(targetMatrix))

    trainSSE = calculateSSE(trainData, target, weightsMatrix)
    trainRMSE = math.sqrt(trainSSE / len(trainData))

    testSSE = calculateSSE(testData, target, weightsMatrix)
    testRMSE = math.sqrt(testSSE / len(testData))

    if verbose:
        print('------------------')
        print("Weights using Normal Equations:")
        print(weightsMatrix)
        print('Training SSE:', trainSSE)
        print('Training RMSE:', trainRMSE)
        print('Testing SSE:', testSSE)
        print('Testing RMSE:', testRMSE)
        print('------------------')

    return weightsMatrix, trainSSE, trainRMSE, testSSE, testRMSE


def gradientDescent(originalData, target, learningRate, tolerance):
    '''
    Calculates the weights to assign to features using gradient descent. Also
    returns RMSE list of each RMSE per iteration.
    
    :param originalData: pandas dataframe data
    :param target: target column name
    :param learningRate: learning rate to limit gradient descent jumps
    :param tolerance: tolerence threshold to determine when gradient descent
                        has converged
    '''

    data = originalData.copy()

    colNames = getColNames(data)

    featList = colNames.copy()
    if target in colNames:
        featList.remove(target)

    weights = [[0]] * len(featList)

    weightsMatrix = np.array(weights)

    targetCol = data[target].values.tolist()
    targetMatrix = np.array([[n] for n in targetCol])

    dataMatrix = np.array(data.drop(columns=[target]))

    converged = False

    count = 0

    SSE = calculateSSE(data, target, weightsMatrix)
    priorRMSE = math.sqrt(SSE / len(data))
    RMSEList = [priorRMSE]

    while not converged:
        # calculate gradient
        gradient = []
        for feat in range(len(featList)):
            featGrad = 0

            for i in range(len(dataMatrix)):
                featGrad += ((weightsMatrix.T.dot(dataMatrix[i])) - targetMatrix[i]) * dataMatrix[i][feat]
            gradient.append(featGrad)

        gradient = np.array(gradient)
#         print(gradient)

        newWeights = (weightsMatrix - (learningRate * gradient))
        weightsMatrix = newWeights

        SSE = calculateSSE(data, target, weightsMatrix)
        currentRMSE = math.sqrt(SSE / len(data))

        if (abs(priorRMSE - currentRMSE) < tolerance):
            converged = True

        priorRMSE = currentRMSE
        RMSEList.append(priorRMSE)

        count += 1
        if count >= 1000:
            converged = True

    return weightsMatrix, RMSEList


def calculateSSE(data, target, weightsMatrix):
    '''
    After gradient descent, this function can be used to determine 
    the SSE indicating how much error the model has over the data and its 
    predictions.
     
    :param testData: testing data dataframe object
    :param target: target column name
    '''

    targetCol = data[target].values.tolist()

    testData = data.drop(columns=[target])
    testRowsList = testData.values.tolist()
    SSE = 0

    # calculate squared errors for each row in testing data and sum them for the SSE
    for n in range(len(testRowsList)):
        row = testRowsList[n]
        row = np.array(row)

        prediction = float(np.sum((weightsMatrix.T * row)))
        actual = float(targetCol[n])
#         print(row, "\nprediction:", prediction, "actual:", actual, "  ")

        SSE += (prediction - actual) * (prediction - actual)

    return SSE

